"""
title: LiteLLM Unified Image Generator & Editor (Manifold)
author: adapted from dfirmans' OpenRouter image pipe
version: 1.0.0
description: >
  Unified Pipe for image generation and editing in Open WebUI via LiteLLM /
  a LiteLLM Proxy gateway (https://docs.litellm.ai).
  Text-to-image calls hit the proxy's OpenAI-compatible
  `POST /v1/images/generations`; editing (image-to-image) hits
  `POST /v1/images/edits` (multipart, single or multiple images).
  Optional fallback through `/v1/chat/completions` for proxies where the
  image routes are disabled and for chat-native image models
  (gemini image models, requires LiteLLM proxy >= 1.76.1).
  This is a manifold pipe: set `models` to a comma-separated list of model
  names as configured in your LiteLLM proxy (config.yaml `model_list`) and
  each one will appear as a selectable model in Open WebUI.
required_open_webui_version: 0.11.4
requirements: httpx
"""

import re
import json
import base64
import time
import uuid
import random
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

import httpx
from pydantic import BaseModel, Field
from fastapi import Request


class Pipe:
    """
    Unified LiteLLM image generation manifold pipe.

    Set `models` to a comma/newline-separated list of LiteLLM model names —
    the exact `model_name` strings your LiteLLM proxy serves (or qualified
    SDK ids like `gemini/gemini-2.5-flash-image` when the proxy simply
    forwards them). Each entry appears as its own model in Open WebUI.
    """

    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description=(
                "LiteLLM proxy virtual API key (sk-...), created in the "
                "LiteLLM Admin UI or set as the proxy master key."
            ),
        )
        base_url: str = Field(
            default="http://localhost:4000",
            description=(
                "LiteLLM proxy base URL, e.g. http://localhost:4000 or "
                "https://gateway.example.com. May be given with or without "
                "a trailing /v1 — both are handled."
            ),
        )
        models: str = Field(
            default=(
                "gpt-image-1, "
                "gemini/gemini-2.5-flash-image, "
                "gemini/gemini-3-pro-image-preview, "
                "vertex_ai/imagen-4.0-generate-001, "
                "black_forest_labs/flux-pro-1.1, "
                "stability/sd3.5-large"
            ),
            description=(
                "Comma or newline separated LiteLLM model names. Use the "
                "name defined in your proxy config.yaml `model_list` "
                "(e.g. gpt-image-1) or a qualified provider id routed via "
                "the proxy (e.g. gemini/gemini-2.5-flash-image). Each entry "
                "registers as a separate model in Open WebUI."
            ),
        )
        name_prefix: str = Field(
            default="LiteLLM-Image/",
            description="Prefix added to model names shown in Open WebUI.",
        )
        # ---------- size / ratio ----------
        enable_size_detection: bool = Field(
            default=True,
            description="Auto-detect aspect ratio and resolution from prompt keywords.",
        )
        default_aspect_ratio: str = Field(
            default="1:1",
            description="Default aspect ratio when auto-detection finds nothing.",
        )
        default_resolution: str = Field(
            default="1K",
            description="Default image resolution: 512, 1K, 2K, or 4K.",
        )
        size: Optional[str] = Field(
            default=None,
            description=(
                "Explicit size shorthand forwarded to LiteLLM, e.g. "
                "'1024x1024'. Overrides auto-detection when set."
            ),
        )
        strict_openai_sizes: bool = Field(
            default=True,
            description=(
                "For gpt-image-1 / dall-e model names, snap auto-detected "
                "dimensions to the canonical sizes those models accept "
                "(1024x1024, 1536x1024, 1024x1536, 1792x1024, 1024x1792). "
                "Models that are not OpenAI-family (gemini, imagen, flux, "
                "stability...) always get the exact auto-detected size, "
                "since LiteLLM maps size -> aspectRatio per provider."
            ),
        )
        # ---------- OpenAI Images params ----------
        quality: Optional[str] = Field(
            default=None,
            description=(
                "Image quality: auto/high/medium/low for gpt-image-1, "
                "hd/standard for dall-e-3. Ignored by some providers."
            ),
        )
        response_format: Optional[str] = Field(
            default=None,
            description=(
                "Return format asked from LiteLLM: url or b64_json. Leave "
                "empty to not send the field (gpt-image-1 always returns "
                "b64_json; Gemini returns base64 only)."
            ),
        )
        n: int = Field(
            default=1,
            description="Number of images to generate (1-10; dall-e-3 supports n=1).",
        )
        user: Optional[str] = Field(
            default=None,
            description=(
                "Optional stable end-user identifier forwarded to LiteLLM "
                "for end-user cost tracking."
            ),
        )
        background: Optional[str] = Field(
            default=None,
            description=(
                "Background handling for gpt-image-1 family: auto, "
                "transparent, or opaque."
            ),
        )
        # ---------- editing ----------
        max_input_images: int = Field(
            default=16,
            description="Max uploaded images sent to /v1/images/edits per request.",
        )
        # ---------- advanced ----------
        extra_params_json: str = Field(
            default="",
            description=(
                "Optional JSON object merged into the request payload for "
                "provider-specific params, e.g. "
                '{"output_format": "png", "seed": 42}.'
            ),
        )
        enable_chat_completions_fallback: bool = Field(
            default=False,
            description=(
                "Fall back to POST /v1/chat/completions (LiteLLM-standardized "
                "`message.images`, proxy >= 1.76.1, gemini image models) when "
                "the images routes fail."
            ),
        )
        timeout: int = Field(
            default=300,
            description="Request timeout in seconds.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.tooltips = [
            "Try a different aspect ratio — e.g. '16:9' for widescreen, '3:2' for photo, '9:16' for vertical.",
            "For higher resolution, mention '2K' or '4K' in your prompt.",
            "Add to the end of the prompt: --size 1024x1536 to pin an explicit size.",
            "Upload an image and describe changes to use image-to-image editing.",
            "Try changing the mood: 'make it darker and more mysterious'.",
            "Add atmosphere: 'foggy', 'rainy', 'neon-lit'.",
            "Keep naming consistent in edit requests: 'same shirt, new color'.",
            "Set the LiteLLM valve `response_format` to b64_json to always render inline images.",
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
        """Open WebUI passes model as 'function_id.model_id' — strip the prefix.

        Careful: many LiteLLM model names contain dots themselves
        (gemini-2.5-flash-image, sd3.5-large, flux.2-pro). Only strip when
        the segment before the first dot is slash-free (function ids never
        contain '/'), so 'fLitellm-pipe.google/gemini-2.5-flash-image' works
        while 'google/gemini-2.5-flash-image' passes through untouched.
        """
        dot_idx = raw_model.find(".")
        if dot_idx > -1 and "/" not in raw_model[:dot_idx]:
            return raw_model[dot_idx + 1:]
        return raw_model

    def pipes(self) -> List[Dict[str, str]]:
        """Return models registered by this pipe. Called by Open WebUI."""
        if not self.valves.api_key:
            return [
                {
                    "id": "error_no_api_key",
                    "name": f"{self.valves.name_prefix}❌ Set LiteLLM Proxy API Key",
                }
            ]

        if not (self.valves.base_url or "").strip().rstrip("/"):
            return [
                {
                    "id": "error_no_base_url",
                    "name": f"{self.valves.name_prefix}❌ Set LiteLLM base_url",
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
    # URL / header helpers
    # ------------------------------------------------------------------

    @property
    def _proxy_base(self) -> str:
        return (self.valves.base_url or "").strip().rstrip("/")

    def _api_url(self, endpoint: str) -> str:
        """endpoint examples: 'images/generations', 'images/edits',
        'chat/completions'. Handles base URLs with or without /v1."""
        base = self._proxy_base
        if base.endswith("/v1"):
            return f"{base}/{endpoint}"
        return f"{base}/v1/{endpoint}"

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.valves.api_key}",
        }

    def _json_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            **self._auth_headers(),
        }

    def _parse_extra_params(self) -> Dict[str, Any]:
        if not self.valves.extra_params_json:
            return {}
        try:
            parsed = json.loads(self.valves.extra_params_json)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError) as e:
            print(f"WARNING: extra_params_json is not valid JSON — ignored: {e}")
        return {}

    # ------------------------------------------------------------------
    # Image extraction helpers (from the conversation)
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
                url = file_item.get("url", "")
                if isinstance(data, str) and data.startswith("data:image"):
                    images.append(data)
                elif isinstance(url, str) and url.startswith(("http://", "https://")):
                    images.append(url)

        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image") or url.startswith(
                        ("http://", "https://")
                    ):
                        images.append(url)

        if isinstance(content, str):
            for data_url in re.findall(r"\((data:image/[^;]+;base64,[^)]+)\)", content):
                images.append(data_url)
            for http_url in re.findall(r"\((https?://\S+\.(?:png|jpe?g|webp|gif))\)",
                                       content, re.IGNORECASE):
                images.append(http_url)

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

    @classmethod
    def _decode_data_url(cls, data_url: str) -> Tuple[str, bytes]:
        """'data:image/png;base64,xxx' -> ('image/png', decoded_bytes).

        Raises ValueError on malformed input."""
        match = re.match(r"^data:([^;]+);base64,(.+)$", data_url, re.DOTALL)
        if not match:
            raise ValueError("Not a valid base64 data URL")
        mime = match.group(1)
        raw_b64 = match.group(2).strip()
        # tolerate whitespace / data-URL line breaks
        raw_b64 = re.sub(r"\s+", "", raw_b64)
        pad = "=" * (-len(raw_b64) % 4)
        payload = base64.b64decode(raw_b64 + pad, validate=False)
        return mime, payload

    @classmethod
    def _mime_from_bytes(cls, payload: bytes, fallback: str = "image/png") -> str:
        sig = payload[:12]
        if sig.startswith(b"\x89PNG"):
            return "image/png"
        if sig.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if sig.startswith(b"RIFF") and payload[8:12] == b"WEBP":
            return "image/webp"
        if sig.startswith(b"GIF8"):
            return "image/gif"
        return fallback

    EXT_BY_MIME = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }

    async def _image_to_part(
        self, item: str, index: int
    ) -> Optional[Tuple[str, bytes, str]]:
        """Convert a conversation image (data URL, bare base64, or remote URL)
        into a multipart file part (filename, bytes, content_type)."""
        try:
            if item.startswith("data:"):
                mime, payload = self._decode_data_url(item)
            elif item.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=60, follow_redirects=True) as dl:
                    resp = await dl.get(item)
                    resp.raise_for_status()
                    payload = resp.content
                    mime = (resp.headers.get("content-type") or "").split(";")[0]
                if not mime.startswith("image/"):
                    mime = self._mime_from_bytes(payload)
            else:
                payload = base64.b64decode(
                    re.sub(r"\s+", "", item) + "=" * (-len(re.sub(r"\s+", "", item)) % 4),
                    validate=False,
                )
                mime = self._mime_from_bytes(payload)
        except Exception as e:
            print(f"WARNING: could not convert input image #{index}: {e}")
            return None

        ext = self.EXT_BY_MIME.get(mime, "png")
        return f"input_image_{index}.{ext}", payload, mime

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
        "16:9",
        "5:4",
        "4:3",
        "3:2",
        "1:1",
        "2:3",
        "3:4",
        "4:5",
        "9:16",
        "1:2",
        "2:1",
        "1:4",
        "4:1",
        "1:8",
        "8:1",
    ]

    RATIO_ALIASES = {
        "widescreen": "16:9",
        "landscape": "3:2",
        "horizontal": "3:2",
        "wide": "16:9",
        "banner": "16:9",
        "header": "16:9",
        "desktop": "16:9",
        "monitor": "16:9",
        "youtube": "16:9",
        "cinematic": "21:9",
        "ultrawide": "21:9",
        "square": "1:1",
        "instagram post": "1:1",
        "profile picture": "1:1",
        "avatar": "1:1",
        "portrait": "3:4",
        "vertical": "9:16",
        "tall": "9:16",
        "mobile": "9:16",
        "phone": "9:16",
        "instagram story": "9:16",
        "story": "9:16",
        "tiktok": "9:16",
        "reels": "9:16",
        "shorts": "9:16",
    }

    RESOLUTIONS = ["512", "1K", "2K", "4K"]

    # base dimension per detected resolution keyword
    RES_BASE_PX = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}

    # sizes accepted by OpenAI-family image models, per orientation
    OPENAI_SIZES = {
        "default": {  # gpt-image-1 and friends
            "square": [(1024, 1024)],
            "landscape": [(1536, 1024)],
            "portrait": [(1024, 1536)],
        },
        "dall-e-3": {
            "square": [(1024, 1024)],
            "landscape": [(1792, 1024)],
            "portrait": [(1024, 1792)],
        },
        "dall-e-2": {
            "square": [(256, 256), (512, 512), (1024, 1024)],
            "landscape": [(1024, 1024)],
            "portrait": [(1024, 1024)],
        },
    }

    @classmethod
    def _openai_size_family(cls, model_id: str) -> str:
        model_lower = (model_id or "").lower()
        if "dall-e-3" in model_lower:
            return "dall-e-3"
        if "dall-e-2" in model_lower:
            return "dall-e-2"
        return "default"

    def _detect_aspect_ratio(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        for ratio in sorted(self.ASPECT_RATIOS, key=len, reverse=True):
            variants = [ratio, ratio.replace(":", "/")]
            if any(v in p for v in variants):
                return ratio
        for alias in sorted(self.RATIO_ALIASES, key=len, reverse=True):
            if alias in p:
                return self.RATIO_ALIASES[alias]
        return None

    def _detect_resolution(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        for res in sorted(
            self.RESOLUTIONS,
            key=lambda s: float(s[:-1]) if s.endswith("K") else int(s),
            reverse=True,
        ):
            tokens = [res.lower()]
            if res.endswith("K"):
                tokens.append(f"{res[:-1]} k")
            for token in tokens:
                if re.search(rf"\b{re.escape(token)}\b", p):
                    return res
        return None

    def _strip_inline_ar(self, prompt: str) -> str:
        """Remove '--ar 16:9' / '--ratio 16:9' style markers so they don't leak
        into the actual image caption."""
        cleaned = re.sub(r"--(?:ar|ratio)\s*[\d./:]+", " ", prompt, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def _ratio_to_size(self, ratio: str, resolution: str) -> str:
        """Map ratio (e.g. 16:9) + resolution (512/1K/2K/4K) to WxH, rounded
        to a multiple of 64 — friendly to Gemini/BFL/Bedrock targets."""
        base = self.RES_BASE_PX.get(resolution.upper(), 1024)
        try:
            w_ratio, h_ratio = (float(x) for x in ratio.split(":", 1))
        except ValueError:
            w_ratio, h_ratio = 1.0, 1.0
        if w_ratio <= 0 or h_ratio <= 0:
            w_ratio, h_ratio = 1.0, 1.0
        area = base * base
        side_a = (area * w_ratio / h_ratio) ** 0.5
        side_b = side_a * h_ratio / w_ratio
        long_side = int(round(max(side_a, side_b) / 64.0)) * 64
        short_side = int(round(min(side_a, side_b) / 64.0)) * 64
        w = long_side if w_ratio >= h_ratio else short_side
        h = short_side if w_ratio >= h_ratio else long_side
        return f"{max(64, w)}x{max(64, h)}"

    @classmethod
    def _is_openai_family(cls, model_id: str) -> bool:
        model_lower = (model_id or "").lower()
        return "gpt-image" in model_lower or "dall-e" in model_lower

    @classmethod
    def _snap_to_openai_size(cls, model_id: str, size_str: str) -> str:
        """Snap computed WxH to the canonical size OpenAI-family models accept,
        preserving orientation (gpt-image-1: 1536x1024 landscape; dall-e-3:
        1792x1024; dall-e-2: squares only)."""
        try:
            tw, th = (int(x) for x in size_str.lower().split("x", 1))
        except ValueError:
            return "1024x1024"
        family = cls._openai_size_family(model_id)
        table = cls.OPENAI_SIZES.get(family, cls.OPENAI_SIZES["default"])
        target_area = tw * th
        if 0.9 <= (tw / th) <= 1.1:
            orientation = "square"
        elif tw > th:
            orientation = "landscape"
        else:
            orientation = "portrait"
        candidates = table.get(orientation) or table["square"]
        # pick the closest-by-area candidate within the same orientation,
        # scaled to the detected resolution tier when possible
        best, best_diff = None, None
        for cw, ch in candidates:
            diff = abs((cw * ch) - target_area)
            if best_diff is None or diff < best_diff:
                best, best_diff = (cw, ch), diff
        return f"{best[0]}x{best[1]}"

    def _resolve_size(
        self, prompt: str, model_id: Optional[str] = None
    ) -> Tuple[str, str, Optional[str]]:
        """Returns (aspect_ratio для display, resolution для display, size
        string to send — or None when detection is disabled and no override)."""
        size = self.valves.size
        if not self.valves.enable_size_detection:
            return (
                self.valves.default_aspect_ratio,
                self.valves.default_resolution,
                size,
            )
        aspect_ratio = (
            self._detect_aspect_ratio(prompt) or self.valves.default_aspect_ratio
        )
        resolution = self._detect_resolution(prompt) or self.valves.default_resolution
        if size is None:
            size = self._ratio_to_size(aspect_ratio, resolution)
            if (
                size
                and self.valves.strict_openai_sizes
                and self._is_openai_family(model_id or "")
            ):
                size = self._snap_to_openai_size(model_id, size)
        return aspect_ratio, resolution, size

    def _prompt_override_size(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Support '--size 1024x1536' / 'size=1024x1536' inline directive."""
        match = re.search(r"--size[=\s]+(\d{3,5})x(\d{3,5})", prompt, re.IGNORECASE)
        if not match:
            return prompt, None
        size = f"{match.group(1)}x{match.group(2)}"
        cleaned = (prompt[: match.start()] + prompt[match.end():]).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned, size

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _common_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self.valves.n and self.valves.n > 1:
            params["n"] = self.valves.n
        if self.valves.quality:
            params["quality"] = self.valves.quality
        if self.valves.response_format:
            params["response_format"] = self.valves.response_format
        if self.valves.user:
            params["user"] = self.valves.user
        return params

    def _build_generation_payload(
        self,
        model_id: str,
        prompt: str,
        size: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
        }
        payload.update(self._common_params())
        if size:
            payload["size"] = size
        if self.valves.background:
            payload["background"] = self.valves.background
        extra = self._parse_extra_params()
        if extra:
            payload.update(extra)
        return payload

    def _edit_data_params(
        self, model_id: str, prompt: str, size: Optional[str]
    ) -> Dict[str, str]:
        """Textual (form) params for the /v1/images/edits multipart call."""
        data: Dict[str, str] = {"model": model_id, "prompt": prompt}
        common = self._common_params()
        for key in ("n", "quality", "response_format", "user"):
            if key in common:
                data[key] = str(common[key])
        if size:
            data["size"] = size
        extra = self._parse_extra_params()
        for key, value in extra.items():
            if isinstance(value, (dict, list)):
                data[key] = json.dumps(value)
            else:
                data[key] = str(value)
        return data

    # ------------------------------------------------------------------
    # LiteLLM proxy calls
    # ------------------------------------------------------------------

    async def _generate_with_images_route(
        self,
        model_id: str,
        prompt: str,
        size: Optional[str],
        input_images: List[str],
    ) -> List[str]:
        """Text-to-image -> /v1/images/generations (JSON);
        image-to-image -> /v1/images/edits (multipart)."""
        if input_images:
            url = self._api_url("images/edits")
            parts = []
            for idx, item in enumerate(input_images[: self.valves.max_input_images]):
                part = await self._image_to_part(item, idx)
                if part:
                    parts.append(part)
            if not parts:
                raise ValueError("Could not convert any uploaded image for editing.")
            if len(parts) >= 2:
                print(
                    f"DEBUG [Edits] Sending {len(parts)} input images as "
                    f"'image[]' field to {url}"
                )
            data = self._edit_data_params(model_id, prompt, size)
            if len(parts) == 1:
                # single image: plain 'image' field is universally accepted
                files = [("image", parts[0])]
            else:
                # multiple images: LiteLLM expects the 'image[]' alias
                # (-F "image[]=@img.png" per /images/edits docs)
                files = [("image[]", part) for part in parts]
            async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
                resp = await client.post(url, headers=self._auth_headers(), data=data,
                                         files=files)
        else:
            url = self._api_url("images/generations")
            payload = self._build_generation_payload(model_id, prompt, size)
            async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
                resp = await client.post(url, headers=self._json_headers(), json=payload)

        print(
            f"DEBUG [Images] POST {url}  model={model_id}  "
            f"mode={'edit' if input_images else 'generation'}  "
            f"size={size or 'provider-default'}  n={self.valves.n}  "
            f"input_images={len(input_images)}  -> HTTP {resp.status_code}"
        )
        if resp.status_code != 200:
            raise Exception(f"LiteLLM proxy HTTP {resp.status_code}: {resp.text}")
        data_json = resp.json()
        print(f"DEBUG [Images] Response keys: {list(data_json.keys())}")
        image_urls = self._extract_images_from_response(data_json)
        if not image_urls:
            print(f"DEBUG [Images] Full response: {data_json}")
            raise ValueError(
                "No images found in LiteLLM response. Check that the model "
                "supports the images routes (see "
                "https://docs.litellm.ai/docs/image_generation)."
            )
        return image_urls

    @classmethod
    def _extract_images_from_response(cls, data: Dict[str, Any]) -> List[str]:
        """Extract images from an OpenAI-format images response
        ({created, data:[{url|b64_json}], usage})."""
        image_urls = []
        for entry in data.get("data", []):
            if not isinstance(entry, dict):
                continue
            b64 = entry.get("b64_json") or ""
            if b64:
                image_urls.append(cls._normalize_base64(b64))
            elif entry.get("url"):
                image_urls.append(entry["url"])
        return list(dict.fromkeys(image_urls))

    def _build_chat_fallback_payload(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
    ) -> Dict[str, Any]:
        if input_images:
            content: Any = [{"type": "text", "text": prompt}]
            for img_data_url in input_images[: self.valves.max_input_images]:
                content.append({"type": "image_url", "image_url": {"url": img_data_url}})
        else:
            content = prompt
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }
        extra = self._parse_extra_params()
        if extra:
            payload.update(extra)
        return payload

    async def _generate_with_chat_fallback(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
    ) -> List[str]:
        """Fall through /v1/chat/completions. LiteLLM (proxy >= 1.76.1 for
        gemini image models) standardizes generated images as
        message.images = [{image_url: {url: data:image/...}, type: 'image_url'}]."""
        url = self._api_url("chat/completions")
        payload = self._build_chat_fallback_payload(model_id, prompt, input_images)
        print(
            f"DEBUG [Chat fallback] POST {url}  model={model_id}  "
            f"input_images={len(input_images)}"
        )
        async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
            resp = await client.post(url, headers=self._json_headers(), json=payload)
        if resp.status_code != 200:
            raise Exception(f"LiteLLM proxy HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        print(f"DEBUG [Chat fallback] Response keys: {list(data.keys())}")
        image_urls = self._extract_images_from_chat_response(data)
        if not image_urls:
            print(f"DEBUG [Chat fallback] Full response: {data}")
            raise ValueError(
                "No images found in chat-completions fallback response. "
                "Verify the model supports image output via LiteLLM "
                "(>= 1.76.1 for gemini image models)."
            )
        return image_urls

    @classmethod
    def _extract_images_from_chat_response(cls, data: Dict[str, Any]) -> List[str]:
        """Pull images from a LiteLLM chat completion: the standardized
        `message.images` (proxy >= 1.76.1) or inline markdown / image_url
        content returned by other OpenAI-compatible servers."""
        image_urls = []
        try:
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                # LiteLLM standardized `images` field on the assistant message
                for img in message.get("images", []) or []:
                    if isinstance(img, dict):
                        url_val = (img.get("image_url") or {}).get("url", "")
                        if url_val:
                            image_urls.append(url_val)
                # some proxies/models inline markdown images in content instead
                if not image_urls:
                    msg_content = message.get("content", "")
                    if isinstance(msg_content, str):
                        for data_url in re.findall(
                            r"\((data:image/[^;]+;base64,[^)]+)\)", msg_content
                        ):
                            image_urls.append(data_url)
                    elif isinstance(msg_content, list):
                        for item in msg_content:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                url_val = item.get("image_url", {}).get("url", "")
                                if url_val:
                                    image_urls.append(url_val)
        except Exception as e:
            print(f"DEBUG [Chat fallback] Parse error: {e}")
            raise ValueError(f"Cannot parse image from chat response: {data}") from e
        return list(dict.fromkeys(image_urls))

    async def _generate_image(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
        size: Optional[str],
    ) -> List[str]:
        errors = []
        try:
            return await self._generate_with_images_route(
                model_id, prompt, size, input_images
            )
        except Exception as e:
            errors.append(f"Images route error: {e}")
            print(f"DEBUG [Images] Falling back: {e}")
            if not self.valves.enable_chat_completions_fallback:
                raise
        try:
            return await self._generate_with_chat_fallback(
                model_id, prompt, input_images
            )
        except Exception as e:
            errors.append(f"Chat completions fallback error: {e}")
            raise Exception(
                "Failed to generate image through LiteLLM proxy (both "
                "images routes and chat fallback).\n" + "\n".join(errors)
            )

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    def _chunk(self, model_id: str, content: str) -> str:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    def _stop_chunk(self, model_id: str) -> str:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # ------------------------------------------------------------------
    # Capability hints
    # ------------------------------------------------------------------

    IMAGE_INPUT_SUPPORTED = [
        "gemini",
        "nano-banana",
        "imagen",
        "gpt-image",
        "seedream",
        "recraft",
        "qwen-image",
        "flux-kontext",
        "kontext",
        "black-forest",
        "black_forest",
        "stability",
        "sd3",
        "nscale",
        "xinference",
        "vllm-omni",
        "grok-imagine",
        "gpt-4o",
        "pixtral",
    ]
    IMAGE_INPUT_BLOCKED = [
        "dall-e-2",
        "dall-e-3",
        "titan",
        "ideogram",
        "imagegeneration@",
        "text-embedding",
    ]

    def _model_supports_image_input(self, model_id: str):
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in self.IMAGE_INPUT_BLOCKED):
            return False
        if any(kw in model_lower for kw in self.IMAGE_INPUT_SUPPORTED):
            return True
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: dict,
        __event_emitter__=None,
    ) -> AsyncGenerator[str, None]:

        raw_model = body.get("model", "")
        model_id = self._clean_model_id(raw_model)

        if not self.valves.api_key:
            yield self._chunk(
                model_id,
                "❌ **Error:** LiteLLM proxy API key is missing. "
                "Please set it in **Valves → api_key**.",
            )
            yield self._stop_chunk(model_id)
            return

        if not body.get("messages"):
            yield self._chunk(model_id, "❌ **Error:** No messages in request body.")
            yield self._stop_chunk(model_id)
            return

        try:
            last_msg = body["messages"][-1]
            content = last_msg.get("content", "")
            if isinstance(content, list):
                prompt = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                prompt = content
            prompt = prompt.strip()

            if not prompt:
                yield self._chunk(
                    model_id, "❌ **Error:** No text prompt found in the last message."
                )
                yield self._stop_chunk(model_id)
                return

            input_images = self._find_images_in_conversation(body["messages"])
            is_editing = len(input_images) > 0

            if is_editing:
                support = self._model_supports_image_input(model_id)
                if support is False:
                    yield self._chunk(
                        model_id,
                        f"⚠️ **Model `{model_id}` does not support image input.**\n\n"
                        f"Image-to-image editing requires a model whose LiteLLM "
                        f"provider supports `/v1/images/edits` (OpenAI gpt-image-1, "
                        f"Gemini `gemini-2.5-flash-image` family, FLUX Kontext, "
                        f"Stability, Vertex AI Gemini/Imagen, OpenRouter, "
                        f"vLLM-Omni). Please switch models or remove the "
                        f"uploaded image.",
                    )
                    yield self._stop_chunk(model_id)
                    return
                elif support is None:
                    print(
                        f"WARNING: Model '{model_id}' is not in the known "
                        f"compatibility list. Proceeding with image input — "
                        f"API may return an error."
                    )

            # inline directive overrides: --size 1024x1536
            prompt, inline_size = self._prompt_override_size(prompt)
            aspect_ratio, resolution, size = self._resolve_size(prompt, model_id)
            prompt = self._strip_inline_ar(prompt)
            if inline_size:
                size = inline_size

            if is_editing:
                status_msg = (
                    f"✏️ Editing image with `{model_id}` via LiteLLM — "
                    f"size: **{size or 'provider default'}**…"
                )
            else:
                status_msg = (
                    f"🎨 Generating image with `{model_id}` via LiteLLM — "
                    f"size: **{size or 'provider default'}**…"
                )

            print(
                f"DEBUG: model={model_id}  prompt={prompt[:80]}  "
                f"ratio={aspect_ratio}  resolution={resolution}  size={size}  "
                f"images={len(input_images)}"
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": False},
                    }
                )

            image_urls = await self._generate_image(
                model_id, prompt, input_images, size
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "✅ Done!",
                            "done": True,
                        },
                    }
                )

            op_text = "edited" if is_editing else "generated"
            tip = random.choice(self.tooltips)

            if len(image_urls) == 1:
                images_md = f"![Result]({image_urls[0]})"
            else:
                images_md = "\n\n".join(
                    f"![Result {i + 1}]({url})" for i, url in enumerate(image_urls)
                )

            response = (
                f"**🖼️ Here is your {op_text} image:**\n\n"
                f"{images_md}\n\n"
                f"**Size:** `{size or 'provider default'}` | **Model:** `{model_id}`\n\n"
                f"💡 _Tip: {tip}_"
            )
            yield self._chunk(model_id, response)

        except Exception as err:
            print(f"ERROR: {err}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "❌ Error occurred", "done": True},
                    }
                )
            yield self._chunk(model_id, f"❌ **Error:** {str(err)}")

        finally:
            yield self._stop_chunk(model_id)
