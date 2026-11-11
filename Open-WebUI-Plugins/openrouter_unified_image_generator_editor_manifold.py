"""
title: OpenRouter Unified Image Generator & Editor (Manifold)
author: adapted from dfirmans' WAN pipe
version: 3.2.0
description: >
  Unified Pipe for image generation and editing in Open WebUI via OpenRouter.
  Uses the dedicated OpenRouter Image API (`/api/v1/images`) for both
  text-to-image and image-to-image/reference workflows.
  Request parameters are adapted per model using `/api/v1/images/models`
  (unsupported parameters are dropped, values are clamped to what the model accepts).
  This is a manifold pipe: set `models` to a comma-separated list of OpenRouter
  model IDs and each one will appear as a selectable model in Open WebUI.
required_open_webui_version: 0.11.4
requirements: httpx
"""

import re
import json
import math
import time
import random
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

import httpx
from pydantic import BaseModel, Field
from fastapi import Request


class OpenRouterHTTPError(Exception):
    """Non-200 response from OpenRouter, with the status code kept for decisions."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"OpenRouter HTTP {status_code}: {message}")


@dataclass
class ImageRequest:
    model_id: str
    prompt: str
    input_images: List[str]
    aspect_ratio: Optional[str]
    resolution: Optional[str]
    size: Optional[str]
    entry: Optional[Dict[str, Any]] = None
    skip_image_api: bool = False
    session_id: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    sent_ratio: Optional[str] = None
    sent_resolution: Optional[str] = None


class Pipe:
    """
    Unified OpenRouter image generation manifold pipe.

    Set `models` to a comma/newline-separated list of OpenRouter model IDs.
    Each ID appears as its own model in Open WebUI. Selecting one routes the
    request to that model through this pipe.
    """

    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description="OpenRouter API key (get one at https://openrouter.ai/keys)",
        )
        base_url: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenRouter API base URL",
        )
        models: str = Field(
            default=(
                "google/gemini-nano-banana-2.1, "
                "google/gemini-3.1-flash-image, "
                "google/gemini-3.1-flash-lite-image, "
                "google/gemini-2.5-flash-image, "
                "google/gemini-3-pro-image, "
                "openai/gpt-image-2, "
                "black-forest-labs/flux-3-image, "
                "black-forest-labs/flux.2-pro, "
                "bytedance-seed/seedream-5-0-flash, "
                "bytedance-seed/seedream-5-0-pro, "
                "qwen/qwen-image-3"
            ),
            description=(
                "Comma or newline separated OpenRouter model IDs. "
                "Each ID will be registered as a separate model in Open WebUI. "
                "Copy-paste IDs directly from https://openrouter.ai/models."
            ),
        )
        name_prefix: str = Field(
            default="OR-Image/",
            description="Prefix added to model names shown in Open WebUI.",
        )
        enable_legacy_chat_fallback: bool = Field(
            default=False,
            description=(
                "Fallback to legacy /chat/completions path if the model is not served "
                "by the dedicated Image API. Never triggered for auth, billing, rate-limit "
                "or timeout errors, to avoid paying twice."
            ),
        )
        enable_capability_filtering: bool = Field(
            default=True,
            description=(
                "Fetch /api/v1/images/models and adapt each request to the model: "
                "drop unsupported parameters and clamp aspect ratio, resolution, n and "
                "number of reference images to what the model accepts."
            ),
        )
        capabilities_cache_ttl: int = Field(
            default=3600,
            description="Seconds to cache the /images/models capability list.",
        )
        legacy_modalities: str = Field(
            default="image,text",
            description="Comma-separated output modalities for legacy fallback.",
        )
        enable_size_detection: bool = Field(
            default=True,
            description="Auto-detect aspect ratio and resolution from prompt keywords.",
        )
        default_aspect_ratio: str = Field(
            default="1:1",
            description=(
                "Default aspect ratio for text-to-image when auto-detection finds nothing. "
                "Use 'auto' to let the provider choose."
            ),
        )
        edit_keep_source_ratio: bool = Field(
            default=True,
            description=(
                "When editing an image and the prompt names no aspect ratio, send 'auto' "
                "(or nothing) instead of the default ratio so the source image is not "
                "forced into a different shape."
            ),
        )
        default_resolution: str = Field(
            default="1K",
            description="Default image resolution: 512, 768, 1K, 1.5K, 2K, or 4K.",
        )
        size: Optional[str] = Field(
            default=None,
            description="Convenience size shorthand, e.g. '1024x1024' or '2K'.",
        )
        quality: Optional[str] = Field(
            default=None,
            description="Image quality: auto, low, medium, or high.",
        )
        output_format: Optional[str] = Field(
            default=None,
            description="Output image format: png, jpeg, webp, svg, or gif (model dependent).",
        )
        output_compression: Optional[int] = Field(
            default=None,
            description="Compression level 0-100 for jpeg/webp. Ignored for png.",
        )
        background: Optional[str] = Field(
            default=None,
            description="Background: auto, transparent, or opaque.",
        )
        seed: Optional[int] = Field(
            default=None,
            description="Seed for deterministic generation where supported.",
        )
        n: int = Field(
            default=1,
            description="Number of images to generate (1-10).",
        )
        provider_only: Optional[str] = Field(
            default=None,
            description="Provider slugs to pin (e.g. 'google-ai-studio').",
        )
        provider_ignore: Optional[str] = Field(
            default=None,
            description="Provider slugs to exclude (e.g. 'openrouter-auto').",
        )
        provider_order: Optional[str] = Field(
            default=None,
            description="Comma-separated provider order.",
        )
        provider_sort: Optional[str] = Field(
            default=None,
            description="Sort providers by 'price', 'throughput', or 'latency'.",
        )
        allow_fallbacks: bool = Field(
            default=True,
            description="Allow OpenRouter to fall back to another provider on failure.",
        )
        timeout: int = Field(
            default=300,
            description="Request timeout in seconds.",
        )
        user: str = Field(
            default="",
            description="Optional stable end-user identifier for usage tracking.",
        )
        enable_streaming: bool = Field(
            default=False,
            description=(
                "Request SSE streaming when the model supports it (currently OpenAI "
                "models only). Only the final image is shown; other models ignore this."
            ),
        )
        enable_debug: bool = Field(
            default=False,
            description="Print verbose debug logs.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._capabilities: Dict[str, Dict[str, Any]] = {}
        self._capabilities_next_fetch: float = 0.0
        self._capabilities_base: str = ""
        self.tooltips = [
            "Try a different aspect ratio — e.g. '16:9' for widescreen or '9:16' for vertical.",
            "Add artistic direction: 'in the style of Studio Ghibli' or 'cinematic lighting'.",
            "For higher resolution, mention '4K' in your prompt.",
            "Upload an image and describe changes to use image-to-image editing.",
            "Try changing the mood: 'make it darker and more mysterious'.",
            "Change the time of day: 'at golden hour', 'at midnight', 'at dawn'.",
            "Add atmosphere: 'foggy', 'rainy', 'neon-lit'.",
            "For portraits, try '3:4' or '4:5' ratio for a natural framing.",
        ]

    # ------------------------------------------------------------------
    # Model list parsing & manifold registration
    # ------------------------------------------------------------------

    def _parse_models(self) -> List[str]:
        raw = self.valves.models or ""
        ids = []
        for part in re.split(r"[,\n]+", raw):
            model_id = part.strip().strip(",; ")
            if model_id:
                ids.append(model_id)
        return ids

    @staticmethod
    def _clean_model_id(raw_model: str) -> str:
        """Open WebUI passes model as 'function_id.model_id' — strip the prefix."""
        if "." in raw_model:
            return raw_model.split(".", 1)[1]
        return raw_model

    def pipes(self) -> List[Dict[str, str]]:
        """Return models registered by this pipe. Called by Open WebUI."""
        if not self.valves.api_key:
            return [
                {
                    "id": "error_no_api_key",
                    "name": f"{self.valves.name_prefix}❌ Set OpenRouter API Key",
                }
            ]

        models = self._parse_models()
        if not models:
            return [
                {
                    "id": "error_no_models",
                    "name": f"{self.valves.name_prefix}❌ No models configured",
                }
            ]

        return [
            {"id": model_id, "name": f"{self.valves.name_prefix}{model_id}"}
            for model_id in models
        ]

    # ------------------------------------------------------------------
    # Image extraction helpers
    # ------------------------------------------------------------------

    def _extract_images_from_message(self, message: Dict[str, Any]) -> List[str]:
        images = []

        for img in message.get("images", []):
            if isinstance(img, str):
                if img.startswith("data:image"):
                    images.append(img)
                else:
                    images.append(self._normalize_base64(img))

        for file_item in message.get("files", []):
            if isinstance(file_item, dict):
                data = file_item.get("data", "")
                if data.startswith("data:image"):
                    images.append(data)

        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        images.append(url)

        if isinstance(content, str):
            for data_url in re.findall(r"\((data:image/[^;]+;base64,[^)]+)\)", content):
                images.append(data_url)

        return list(dict.fromkeys(images))

    def _find_images_in_conversation(self, messages: List[Dict]) -> List[str]:
        for message in reversed(messages):
            imgs = self._extract_images_from_message(message)
            if imgs:
                return imgs
        return []

    @staticmethod
    def _normalize_base64(b64_string: str) -> str:
        s = b64_string.strip()
        sig = s[:8]
        if sig.startswith("iVBORw0K"):
            return f"data:image/png;base64,{s}"
        if sig.startswith("/9j/"):
            return f"data:image/jpeg;base64,{s}"
        if sig.startswith("UklGR"):
            return f"data:image/webp;base64,{s}"
        if sig.startswith("R0lGOD"):
            return f"data:image/gif;base64,{s}"
        if s.startswith("PHN2Zy"):
            return f"data:image/svg+xml;base64,{s}"
        return f"data:image/png;base64,{s}"

    # ------------------------------------------------------------------
    # Size / ratio detection
    # ------------------------------------------------------------------

    ASPECT_RATIOS = [
        "9:19.5",
        "19.5:9",
        "9:20",
        "20:9",
        "9:21",
        "21:9",
        "2.35:1",
        "16:9",
        "9:16",
        "5:4",
        "4:3",
        "3:2",
        "1:1",
        "2:3",
        "3:4",
        "4:5",
        "5:7",
        "7:5",
        "5:2",
        "1:2",
        "2:1",
        "1:3",
        "3:1",
        "1:4",
        "4:1",
        "1:8",
        "8:1",
    ]

    # "3/4" is usually a fraction ("three-quarter view"), so only the common
    # video/screen ratios are accepted in slash form.
    SLASH_RATIOS = ["16/9", "9/16", "21/9", "9/21"]

    # Deliberately excludes ambiguous words such as "landscape", "portrait",
    # "cinematic", "wide", "tall", "story" or "phone" that commonly appear in
    # prompts without implying a canvas shape.
    RATIO_ALIASES = {
        "widescreen": "16:9",
        "ultrawide": "21:9",
        "ultra-wide": "21:9",
        "youtube thumbnail": "16:9",
        "desktop wallpaper": "16:9",
        "banner": "16:9",
        "landscape orientation": "16:9",
        "landscape format": "16:9",
        "square image": "1:1",
        "square format": "1:1",
        "square crop": "1:1",
        "square canvas": "1:1",
        "instagram post": "1:1",
        "profile picture": "1:1",
        "avatar": "1:1",
        "portrait orientation": "3:4",
        "portrait format": "3:4",
        "vertical": "9:16",
        "phone wallpaper": "9:16",
        "mobile wallpaper": "9:16",
        "instagram story": "9:16",
        "tiktok": "9:16",
        "reels": "9:16",
        "youtube shorts": "9:16",
    }

    RESOLUTIONS = ["512", "768", "1K", "1.5K", "2K", "4K"]
    TIER_VALUE = {
        "512": 0.5,
        "768": 0.75,
        "1K": 1.0,
        "1.5K": 1.5,
        "2K": 2.0,
        "4K": 4.0,
    }

    def _detect_aspect_ratio(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        best: Optional[Tuple[int, str]] = None

        def consider(pos: int, ratio: str) -> None:
            nonlocal best
            if best is None or pos < best[0]:
                best = (pos, ratio)

        for ratio in self.ASPECT_RATIOS:
            m = re.search(rf"(?<![\d:.]){re.escape(ratio)}(?![\d:]|\.\d)", p)
            if m:
                consider(m.start(), ratio)
        for slash in self.SLASH_RATIOS:
            m = re.search(rf"(?<![\d:/.]){re.escape(slash)}(?![\d:/]|\.\d)", p)
            if m:
                consider(m.start(), slash.replace("/", ":"))
        if best:
            return best[1]

        for alias in sorted(self.RATIO_ALIASES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", p):
                return self.RATIO_ALIASES[alias]
        return None

    def _detect_resolution(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        for res in sorted(self.RESOLUTIONS, key=self.TIER_VALUE.get, reverse=True):
            if res.endswith("K"):
                num = re.escape(res[:-1].lower())
                pattern = rf"(?<![\w.]){num}\s?k(?!\w)"
            else:
                pattern = rf"(?<![\w.]){res}\s?px\b"
            if re.search(pattern, p):
                return res
        return None

    def _resolve_size_and_ratio(
        self, prompt: str, is_editing: bool = False
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        detected_ratio = detected_res = None
        if self.valves.enable_size_detection:
            detected_ratio = self._detect_aspect_ratio(prompt)
            detected_res = self._detect_resolution(prompt)

        keep_source = is_editing and self.valves.edit_keep_source_ratio
        if detected_ratio:
            aspect_ratio: Optional[str] = detected_ratio
        elif keep_source:
            aspect_ratio = "auto"
        else:
            aspect_ratio = self.valves.default_aspect_ratio

        if detected_res:
            resolution: Optional[str] = detected_res
        elif keep_source:
            resolution = None
        else:
            resolution = self.valves.default_resolution

        return aspect_ratio, resolution, self.valves.size

    # ------------------------------------------------------------------
    # Model capabilities (/api/v1/images/models)
    # ------------------------------------------------------------------

    def _debug(self, message: str) -> None:
        if self.valves.enable_debug:
            print(f"DEBUG {message}")

    async def _get_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Return {model_id: model entry}; empty if unavailable. Cached with a TTL."""
        base = self.valves.base_url.rstrip("/")
        if self._capabilities_base != base:
            self._capabilities = {}
            self._capabilities_next_fetch = 0.0
            self._capabilities_base = base

        now = time.time()
        if now < self._capabilities_next_fetch:
            return self._capabilities

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{base}/images/models", headers=self._headers())
            resp.raise_for_status()
            entries = resp.json().get("data", [])
            self._capabilities = {
                m["id"]: m for m in entries if isinstance(m, dict) and m.get("id")
            }
            self._capabilities_next_fetch = now + max(
                self.valves.capabilities_cache_ttl, 60
            )
            self._debug(f"[Capabilities] loaded {len(self._capabilities)} image models")
        except Exception as e:
            self._capabilities_next_fetch = now + 60
            self._debug(f"[Capabilities] fetch failed ({type(e).__name__}): {e}")
        return self._capabilities

    async def _get_model_entry(self, model_id: str) -> Tuple[Optional[Dict], bool]:
        """Return (entry or None, capabilities_loaded)."""
        if not self.valves.enable_capability_filtering:
            return None, False
        caps = await self._get_capabilities()
        return caps.get(model_id), bool(caps)

    @staticmethod
    def _ratio_value(ratio: str) -> Optional[float]:
        try:
            a, b = ratio.split(":")
            a, b = float(a), float(b)
            return a / b if a > 0 and b > 0 else None
        except (ValueError, AttributeError):
            return None

    def _nearest_ratio(self, ratio: str, supported: List[str]) -> Optional[str]:
        target = self._ratio_value(ratio)
        if target is None:
            return None
        best, best_dist = None, None
        for cand in supported:
            val = self._ratio_value(cand)
            if val is None:
                continue
            dist = abs(math.log(val) - math.log(target))
            if best_dist is None or dist < best_dist:
                best, best_dist = cand, dist
        return best

    def _shape_ratio(
        self, ratio: Optional[str], sp: Optional[Dict], notes: List[str]
    ) -> Optional[str]:
        if not ratio:
            return None
        if sp is None:
            return ratio
        desc = sp.get("aspect_ratio")
        if not desc:
            return None
        values = desc.get("values") if desc.get("type") == "enum" else None
        if not values or ratio in values:
            return ratio
        if ratio == "auto":
            return None
        nearest = self._nearest_ratio(ratio, values)
        if nearest and nearest != ratio:
            notes.append(f"aspect ratio {ratio} → {nearest}")
        return nearest

    def _shape_resolution(
        self, resolution: Optional[str], sp: Optional[Dict], notes: List[str]
    ) -> Optional[str]:
        if not resolution:
            return None
        if sp is None:
            return resolution
        desc = sp.get("resolution")
        if not desc:
            return None
        values = desc.get("values") if desc.get("type") == "enum" else None
        if not values or resolution in values:
            return resolution
        target = self.TIER_VALUE.get(resolution)
        cands = [(v, self.TIER_VALUE[v]) for v in values if v in self.TIER_VALUE]
        if target is None or not cands:
            return None
        nearest = min(cands, key=lambda c: abs(math.log(c[1]) - math.log(target)))[0]
        notes.append(f"resolution {resolution} → {nearest}")
        return nearest

    @staticmethod
    def _apply_param(
        payload: Dict[str, Any],
        key: str,
        value: Any,
        sp: Optional[Dict],
        notes: List[str],
    ) -> None:
        """Set payload[key] if the model supports it, adapting to its descriptor."""
        if value is None or value == "":
            return
        if sp is None:
            payload[key] = value
            return
        desc = sp.get(key)
        if not desc:
            notes.append(f"{key} not supported by this model (ignored)")
            return
        kind = desc.get("type")
        if kind == "enum" and value not in desc.get("values", [value]):
            notes.append(f"{key}={value} not supported by this model (ignored)")
            return
        if kind == "range" and isinstance(value, (int, float)):
            clamped = max(desc.get("min", value), min(desc.get("max", value), value))
            if clamped != value:
                notes.append(f"{key} {value} → {clamped}")
            value = clamped
        payload[key] = value

    # ------------------------------------------------------------------
    # OpenRouter API helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.api_key.strip()}",
            "HTTP-Referer": "https://openwebui.com",
            "X-OpenRouter-Title": "Open WebUI Image Pipe",
        }

    @staticmethod
    def _split_csv(value: Optional[str]) -> List[str]:
        return [s.strip() for s in (value or "").split(",") if s.strip()]

    def _build_provider_object(self) -> Optional[Dict[str, Any]]:
        provider: Dict[str, Any] = {}
        for key, raw in (
            ("only", self.valves.provider_only),
            ("ignore", self.valves.provider_ignore),
            ("order", self.valves.provider_order),
        ):
            items = self._split_csv(raw)
            if items:
                provider[key] = items
        if self.valves.provider_sort:
            provider["sort"] = self.valves.provider_sort.strip()
        if not self.valves.allow_fallbacks:
            provider["allow_fallbacks"] = False
        return provider or None

    def _build_image_payload(self, req: "ImageRequest") -> Dict[str, Any]:
        entry = req.entry
        sp: Optional[Dict] = entry.get("supported_parameters", {}) if entry else None
        notes = req.notes
        payload: Dict[str, Any] = {"model": req.model_id, "prompt": req.prompt}

        # n
        n = max(1, min(10, self.valves.n or 1))
        if n > 1 and sp is not None:
            n_desc = sp.get("n") or {}
            n_max = n_desc.get("max", 1) if n_desc else 1
            if n > n_max:
                notes.append(f"n {n} → {n_max}")
                n = n_max
        if n > 1:
            payload["n"] = n

        # size / aspect ratio / resolution
        aspect_ratio, resolution, size = req.aspect_ratio, req.resolution, req.size
        if size:
            size = size.strip()
            if re.fullmatch(r"(?i)512|768|\d(?:\.5)?k", size):
                resolution = size.upper()
                size = None
            elif sp is None or "size" in sp:
                payload["size"] = size
            else:
                m = re.fullmatch(r"(\d+)\s*[xX×]\s*(\d+)", size)
                if m:
                    w, h = int(m.group(1)), int(m.group(2))
                    g = math.gcd(w, h) or 1
                    aspect_ratio = f"{w // g}:{h // g}"
                notes.append(f"size {size} not supported by this model (ignored)")
        if "size" not in payload:
            req.sent_ratio = self._shape_ratio(aspect_ratio, sp, notes)
            if req.sent_ratio:
                payload["aspect_ratio"] = req.sent_ratio
            req.sent_resolution = self._shape_resolution(resolution, sp, notes)
            if req.sent_resolution:
                payload["resolution"] = req.sent_resolution
        else:
            req.sent_ratio = req.sent_resolution = payload["size"]

        # generic optional parameters
        output_format = (self.valves.output_format or "").strip().lower() or None
        self._apply_param(payload, "quality", self.valves.quality, sp, notes)
        self._apply_param(payload, "output_format", output_format, sp, notes)
        if output_format in ("jpeg", "webp"):
            self._apply_param(
                payload, "output_compression", self.valves.output_compression, sp, notes
            )
        self._apply_param(payload, "background", self.valves.background, sp, notes)
        self._apply_param(payload, "seed", self.valves.seed, sp, notes)

        if self.valves.user:
            payload["user"] = self.valves.user[:256]
        if req.session_id:
            payload["session_id"] = req.session_id[:256]

        # streaming only when the model supports it
        if self.valves.enable_streaming and (
            entry is None or entry.get("supports_streaming")
        ):
            payload["stream"] = True

        # reference images
        if req.input_images:
            refs_desc = sp.get("input_references") if sp is not None else None
            max_refs = refs_desc.get("max", 16) if refs_desc else 16
            images = req.input_images[:max_refs]
            if len(images) < len(req.input_images):
                notes.append(
                    f"reference images {len(req.input_images)} → {len(images)} "
                    f"(model limit)"
                )
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": url}} for url in images
            ]

        provider = self._build_provider_object()
        if provider:
            payload["provider"] = provider
        return payload

    @staticmethod
    def _error_message(body: str) -> str:
        try:
            data = json.loads(body)
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str):
                return err
        except (ValueError, AttributeError):
            pass
        return body.strip()[:500] or "(empty response)"

    async def _generate_image_with_image_api(self, req: "ImageRequest") -> List[str]:
        url = f"{self.valves.base_url.rstrip('/')}/images"
        payload = self._build_image_payload(req)
        self._debug(
            f"[Image API] POST {url} model={req.model_id} "
            f"ratio={payload.get('aspect_ratio')} resolution={payload.get('resolution')} "
            f"size={payload.get('size')} refs={len(payload.get('input_references', []))} "
            f"n={payload.get('n', 1)} stream={payload.get('stream', False)} "
            f"notes={req.notes}"
        )

        async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
            async with client.stream(
                "POST", url, headers=self._headers(), json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    raise OpenRouterHTTPError(
                        resp.status_code, self._error_message(body)
                    )
                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    image_urls = await self._parse_sse(resp)
                else:
                    raw = await resp.aread()
                    try:
                        data = json.loads(raw)
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid JSON from Image API: {raw[:200]!r}"
                        ) from e
                    self._debug(f"[Image API] response keys: {list(data.keys())}")
                    image_urls = self._extract_images_from_response(data)

        if not image_urls:
            raise ValueError(
                "No images found in response. Check that the model supports /api/v1/images."
            )
        return image_urls

    @classmethod
    def _extract_images_from_response(cls, data: Dict[str, Any]) -> List[str]:
        """Extract data-URLs / URLs from a buffered Image API response."""
        image_urls = []
        for entry in data.get("data", []) or []:
            b64 = entry.get("b64_json") or entry.get("b64Json") or ""
            media_type = entry.get("media_type") or "image/png"
            if b64:
                image_urls.append(f"data:{media_type};base64,{b64}")
            elif entry.get("url"):
                image_urls.append(entry["url"])
        return list(dict.fromkeys(image_urls))

    @staticmethod
    async def _parse_sse(resp: httpx.Response) -> List[str]:
        """Collect final images from the SSE stream (partial images are ignored)."""
        image_urls: List[str] = []
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:") :].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type", "")
            if event_type == "image_generation.completed":
                b64 = event.get("b64_json") or event.get("b64Json") or ""
                if b64:
                    media_type = event.get("media_type") or "image/png"
                    image_urls.append(f"data:{media_type};base64,{b64}")
            elif event_type == "error":
                err = event.get("error")
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
                raise Exception(f"Stream error: {msg}")
        return image_urls

    def _build_legacy_payload(self, req: "ImageRequest") -> Dict[str, Any]:
        if req.input_images:
            content: Any = [{"type": "text", "text": req.prompt}]
            for img_data_url in req.input_images:
                content.append(
                    {"type": "image_url", "image_url": {"url": img_data_url}}
                )
        else:
            content = req.prompt
        modalities = self._split_csv(self.valves.legacy_modalities)

        image_config: Dict[str, Any] = {}
        if req.aspect_ratio and req.aspect_ratio != "auto":
            image_config["aspect_ratio"] = req.aspect_ratio
        if req.size:
            image_config["size"] = req.size
        elif req.resolution:
            image_config["image_size"] = req.resolution
        if self.valves.quality:
            image_config["quality"] = self.valves.quality
        req.sent_ratio = image_config.get("aspect_ratio")
        req.sent_resolution = image_config.get("image_size") or image_config.get("size")

        payload: Dict[str, Any] = {
            "model": req.model_id,
            "messages": [{"role": "user", "content": content}],
            "modalities": modalities,
            "stream": False,
        }
        if image_config:
            payload["image_config"] = image_config
        provider = self._build_provider_object()
        if provider:
            payload["provider"] = provider
        return payload

    async def _generate_image_with_legacy_api(self, req: "ImageRequest") -> List[str]:
        url = f"{self.valves.base_url.rstrip('/')}/chat/completions"
        payload = self._build_legacy_payload(req)
        self._debug(
            f"[Legacy API] POST {url} model={req.model_id} "
            f"image_config={payload.get('image_config')} refs={len(req.input_images)}"
        )
        async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
        if resp.status_code != 200:
            raise OpenRouterHTTPError(resp.status_code, self._error_message(resp.text))
        data = resp.json()
        self._debug(f"[Legacy API] response keys: {list(data.keys())}")

        image_urls: List[str] = []
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message", {}) or {}
            for img in message.get("images", []) or []:
                url_val = (img.get("image_url") or {}).get("url", "")
                if url_val:
                    image_urls.append(url_val)
            if not image_urls and isinstance(message.get("content"), list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url_val = (item.get("image_url") or {}).get("url", "")
                        if url_val:
                            image_urls.append(url_val)
        if not image_urls:
            self._debug(f"[Legacy API] response without images: {str(data)[:500]}")
            raise ValueError(
                "No images found in legacy response. Check legacy_modalities and model support."
            )
        return list(dict.fromkeys(image_urls))

    # HTTP statuses that mean "this request shape/model is not handled by the
    # Image API". Everything else (401/402/403/429/5xx/timeouts) is never retried
    # on another endpoint, to avoid paying twice or hiding real problems.
    LEGACY_FALLBACK_STATUSES = (400, 404, 405, 422)

    async def _generate_image(self, req: "ImageRequest") -> List[str]:
        if req.skip_image_api and self.valves.enable_legacy_chat_fallback:
            self._debug(f"{req.model_id} is not an Image API model; using legacy path")
            return await self._generate_image_with_legacy_api(req)

        try:
            return await self._generate_image_with_image_api(req)
        except OpenRouterHTTPError as e:
            if (
                not self.valves.enable_legacy_chat_fallback
                or e.status_code not in self.LEGACY_FALLBACK_STATUSES
            ):
                raise
            image_api_error = str(e)

        try:
            return await self._generate_image_with_legacy_api(req)
        except Exception as e:
            raise Exception(
                "Failed to generate image with both Image API and legacy API.\n"
                f"Image API error: {image_api_error}\n"
                f"Legacy API error: {e}"
            )

    # ------------------------------------------------------------------
    # Capability hints
    # ------------------------------------------------------------------

    # Only used when /images/models is unavailable or the model is not listed.
    IMAGE_INPUT_SUPPORTED = [
        "gemini",
        "gpt-image",
        "seedream",
        "recraft",
        "riverflow",
        "qwen-image",
        "flux",
        "krea",
        "grok-imagine",
        "gpt-4o",
        "claude-3",
        "pixtral",
    ]
    IMAGE_INPUT_BLOCKED = [
        "dall-e",
        "ideogram",
        "stable-diffusion",
        "sdxl",
        "playground",
        "aura",
        "muse-image",
    ]

    def _model_supports_image_input(self, model_id: str):
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in self.IMAGE_INPUT_BLOCKED):
            return False
        if any(kw in model_lower for kw in self.IMAGE_INPUT_SUPPORTED):
            return True
        return None

    @staticmethod
    def _entry_refs(entry: Dict[str, Any]) -> Dict[str, Any]:
        return (entry.get("supported_parameters") or {}).get("input_references") or {}

    def _supports_image_input(
        self, model_id: str, entry: Optional[Dict[str, Any]]
    ) -> Optional[bool]:
        if entry is not None:
            return self._entry_refs(entry).get("max", 0) > 0
        return self._model_supports_image_input(model_id)

    def _image_edit_alternatives(self, caps: Dict[str, Dict[str, Any]]) -> List[str]:
        configured = self._parse_models()
        if caps:
            alts = [
                m for m in configured
                if m in caps and self._entry_refs(caps[m]).get("max", 0) > 0
            ]
            if alts:
                return alts
        return [m for m in configured if self._model_supports_image_input(m)]

    # ------------------------------------------------------------------
    # Error presentation
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_error(err: Exception) -> str:
        if isinstance(err, OpenRouterHTTPError):
            hints = {
                401: "Check the API key in **Valves → api_key**.",
                402: "The Image API requires a balance above $1 before each request, "
                "regardless of the image price. Top up at https://openrouter.ai/credits.",
                429: "Rate limited. Wait a moment and try again.",
            }
            hint = hints.get(err.status_code)
            return f"{err}\n\n💡 {hint}" if hint else f"{err}"
        if isinstance(err, httpx.TimeoutException):
            return (
                f"Request timed out ({err.__class__.__name__}). "
                "Increase **Valves → timeout** for large/high-resolution images."
            )
        if isinstance(err, httpx.RequestError):
            return f"Network error talking to OpenRouter ({err.__class__.__name__}): {err}"
        return str(err) or err.__class__.__name__

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prompt(message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return (content or "").strip() if isinstance(content, str) else ""

    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        metadata = __metadata__ or {}

        # Open WebUI background tasks (title, tags, follow-ups, ...) call the
        # selected model too. Never spend money generating images for them; an
        # empty reply makes Open WebUI fall back to its defaults.
        if __task__ or metadata.get("task"):
            return

        model_id = self._clean_model_id(body.get("model", ""))

        if not self.valves.api_key:
            yield (
                "❌ **Error:** OpenRouter API key is missing. "
                "Please set it in **Valves → api_key**."
            )
            return

        if not model_id:
            yield "❌ **Error:** Could not determine model ID from request."
            return

        try:
            messages = body.get("messages") or []
            prompt = self._extract_prompt(messages[-1]) if messages else ""
            if not prompt:
                yield "❌ **Error:** No text prompt found in the last message."
                return

            input_images = self._find_images_in_conversation(messages)
            is_editing = len(input_images) > 0

            entry, caps_loaded = await self._get_model_entry(model_id)
            caps = self._capabilities if caps_loaded else {}

            if is_editing:
                support = self._supports_image_input(model_id, entry)
                if support is False:
                    alts = self._image_edit_alternatives(caps)
                    alt_text = "\n".join(f"- `{m}`" for m in alts)
                    yield (
                        f"⚠️ **Model `{model_id}` does not support image input.**\n\n"
                        "Image-to-image editing requires a model that accepts image uploads."
                        + (f" Please switch to one of these:\n\n{alt_text}\n\n" if alts else " ")
                        + "Or remove the uploaded image to use text-to-image generation."
                    )
                    return
                if support is None:
                    self._debug(
                        f"Model '{model_id}' not in compatibility list; "
                        "proceeding with image input."
                    )
            elif entry is not None and self._entry_refs(entry).get("min", 0) > 0:
                yield (
                    f"⚠️ **Model `{model_id}` is an edit-only model** and needs at least "
                    "one reference image. Upload an image and describe the change."
                )
                return

            aspect_ratio, resolution, size = self._resolve_size_and_ratio(
                prompt, is_editing
            )
            req = ImageRequest(
                model_id=model_id,
                prompt=prompt,
                input_images=input_images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                size=size,
                entry=entry,
                skip_image_api=caps_loaded and entry is None,
                session_id=metadata.get("chat_id"),
            )

            verb = "✏️ Editing image" if is_editing else "🎨 Generating image"
            shape = f"ratio: **{aspect_ratio or 'auto'}**"
            if resolution:
                shape += f" | resolution: **{resolution}**"
            status_msg = f"{verb} with `{model_id}` — {shape}…"
            self._debug(
                f"model={model_id} prompt={prompt[:80]!r} ratio={aspect_ratio} "
                f"resolution={resolution} size={size} images={len(input_images)}"
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": False},
                    }
                )

            image_urls = await self._generate_image(req)

            ratio_txt = req.sent_ratio or "auto"
            res_txt = req.sent_resolution or "model default"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"✅ Done! ({ratio_txt} · {res_txt})",
                            "done": True,
                        },
                    }
                )

            op_text = "edited" if is_editing else "generated"
            if len(image_urls) == 1:
                images_md = f"![Result]({image_urls[0]})"
            else:
                images_md = "\n\n".join(
                    f"![Result {i + 1}]({url})" for i, url in enumerate(image_urls)
                )

            response = (
                f"**🖼️ Here is your {op_text} image:**\n\n"
                f"{images_md}\n\n"
                f"**Ratio:** `{ratio_txt}` | **Resolution:** `{res_txt}` | "
                f"**Model:** `{model_id}`\n\n"
            )
            if req.notes:
                response += f"⚙️ _Adjusted for this model: {'; '.join(req.notes)}_\n\n"
            response += f"💡 _Tip: {random.choice(self.tooltips)}_"
            yield response

        except Exception as err:
            print(f"ERROR: {type(err).__name__}: {err}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "❌ Error occurred", "done": True},
                    }
                )
            yield f"❌ **Error:** {self._describe_error(err)}"
