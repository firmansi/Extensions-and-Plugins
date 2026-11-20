"""
title: LiteLLM Unified Image Generator & Editor (Manifold)
author: adapted from dfirmans' OpenRouter image pipe
version: 1.3.0
description: >
  Unified Pipe for image generation and editing in Open WebUI via LiteLLM /
  a LiteLLM Proxy gateway (https://docs.litellm.ai).
  Text-to-image calls hit the proxy's OpenAI-compatible
  `POST /v1/images/generations`; editing (image-to-image) hits
  `POST /v1/images/edits` (multipart, single or multiple images).
  Optional fallback through `/v1/chat/completions` for proxies where the
  image routes are disabled and for chat-native image models
  (gemini image models, requires LiteLLM proxy >= 1.76.1).
  Generated images are persisted as Open WebUI files and earlier turns'
  images (uploads or previous results) are resolved back to bytes, so
  follow-up prompts like "add a hat" actually edit the previous image
  instead of generating a new one.
  This is a manifold pipe: set `models` to a comma-separated list of model
  names as configured in your LiteLLM proxy (config.yaml `model_list`) and
  each one will appear as a selectable model in Open WebUI.
required_open_webui_version: 0.11.4
requirements: httpx
"""

import re
import json
import base64
import hashlib
import time
import uuid
import random
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

import httpx
from pydantic import BaseModel, Field
from fastapi import Request


class ImageRouteError(Exception):
    """Failure of a /v1/images/* call, carrying the HTTP status code when
    available, so the caller can decide whether a chat-completions fallback
    is safe (i.e. the first request clearly did not generate anything)."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class Pipe:
    """
    Unified LiteLLM image generation manifold pipe.

    Set `models` to a comma/newline-separated list of LiteLLM model names —
    the exact `model_name` strings your LiteLLM proxy serves (or qualified
    SDK ids like `gemini/gemini-2.5-flash-image` when the proxy simply
    forwards them). Each entry appears as its own model in Open WebUI.

    v1.3.0:
    - Fix image-to-image editing based on an image generated on an earlier
      turn. Open WebUI (>= 0.11.4) now stores model-returned images as real
      files and rewrites inline `data:image/...` Markdown to relative
      `/api/v1/files/<id>/content` URLs. The previous extraction only
      accepted `data:` URLs and absolute http(s) URLs with an image
      extension, so the generated image became invisible to the pipe and the
      follow-up turn did a FRESH generation instead of editing it. History
      extraction now understands Open WebUI file references (relative or
      absolute), `files[]`/`images[]` entries in every shape, and Markdown
      image syntax with any URL. Referenced files are resolved back to bytes
      (via the Open WebUI file-content endpoint, forwarding the caller's own
      auth) before being sent to `/v1/images/edits`.
    - New valve `persist_generated_images` (default True): generated images
      are uploaded to Open WebUI storage and returned as stable
      `/api/v1/files/<id>/content` references, so later turns can always edit
      them. Falls back to inline data URLs when the upload is unavailable.

    v1.2.0:
    - Answer Open WebUI background task calls (chat title, tags, follow-up
      suggestions, ...) locally instead of generating an image. By default
      Open WebUI routes these task calls to the same model that handled the
      chat, so an image pipe would otherwise fire extra LiteLLM requests
      (and 429s) on every turn. New valve `handle_task_calls_locally`.

    v1.1.0:
    - Deduplicate returned images by content, so models/proxies that return
      the same image more than once (reported with Gemini 3 Pro Image /
      "nano banana" models, cf. open-webui issue #19518) no longer render
      duplicates in the chat.
    - The chat-completions fallback is now only attempted when the
      images-route request clearly did not generate anything (route missing,
      connection error, explicit "unsupported" errors). This prevents double
      generation (and double billing) for expensive models when the first
      call timed out or returned an unparseable success response.
    - New valve `history_images_as_input` to disable reusing previous outputs
      as edit inputs.
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
                "vertex_ai/gemini-3-pro-image-preview, "
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
            description=(
                "Number of images to generate (1-10; dall-e-3 supports n=1). "
                "For Gemini image models n maps to candidateCount — every "
                "candidate is a separate generation with its own cost."
            ),
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
        history_images_as_input: bool = Field(
            default=True,
            description=(
                "Use images from earlier turns (including images this pipe "
                "generated before) as edit inputs for follow-up prompts. Set "
                "to False to only use images attached to the newest message, "
                "so follow-ups do a fresh generation instead of editing the "
                "previous result."
            ),
        )
        persist_generated_images: bool = Field(
            default=True,
            description=(
                "Upload generated images to Open WebUI storage and return "
                "them as /api/v1/files/<id>/content references instead of "
                "inline base64. This keeps the images editable across turns "
                "(Open WebUI rewrites inline base64 in chat history, which "
                "breaks image-to-image editing) and keeps chats small. Set "
                "to False to return inline base64 data URLs as before."
            ),
        )
        handle_task_calls_locally: bool = Field(
            default=True,
            description=(
                "Open WebUI sends background task calls (chat title, tags, "
                "follow-up suggestions, ...) to the same model by default. "
                "When True, this pipe answers them locally with a stub text "
                "and does NOT call LiteLLM — otherwise every chat turn would "
                "also trigger image generations for those tasks."
            ),
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

    _FILE_ID_IN_URL_RE = re.compile(
        r"/api/v1/files/([^/?#\s]+)(?:/content)?", re.IGNORECASE
    )
    _BARE_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
    _B64_SIGNATURES = ("iVBORw0K", "/9j/", "UklGR", "R0lGOD", "PHN2Zy")

    @classmethod
    def _owui_file_id(cls, ref: str) -> Optional[str]:
        """Return the Open WebUI file id from a reference, if it is one."""
        if not isinstance(ref, str) or not ref:
            return None
        if ref.startswith(cls._B64_SIGNATURES):
            return None  # bare base64 image, not a file id
        match = cls._FILE_ID_IN_URL_RE.search(ref)
        if match:
            return match.group(1)
        if ref.startswith("/") or ref.startswith(("http://", "https://")):
            return None
        if cls._BARE_FILE_ID_RE.match(ref):
            return ref
        return None

    @staticmethod
    def _looks_like_image_ref(value: str) -> bool:
        if not isinstance(value, str):
            return False
        v = value.strip()
        if not v:
            return False
        if v.startswith("data:image/"):
            return True
        if v.startswith(("http://", "https://")):
            return True
        if v.startswith("/"):
            # Open WebUI file references and image-looking paths only. A
            # plain relative link is not assumed to be an image.
            return bool(
                re.search(r"\.(?:png|jpe?g|webp|gif|bmp|avif|heic)(?:[?#]|$)", v, re.I)
                or "/api/v1/files/" in v
                or "/files/" in v
            )
        return False

    @staticmethod
    def _is_image_file_entry(file_item: Dict[str, Any]) -> bool:
        if str(file_item.get("type") or "").lower() == "image":
            return True
        content_type = file_item.get("content_type")
        if not content_type and isinstance(file_item.get("meta"), dict):
            content_type = file_item["meta"].get("content_type")
        return isinstance(content_type, str) and content_type.lower().startswith(
            "image/"
        )

    @classmethod
    def _normalize_refs(cls, refs: List[str]) -> List[str]:
        """Normalize collected references to data URLs or resolvable URLs.

        Bare (data-URL-less) base64 is wrapped; a bare Open WebUI file id is
        turned into its /api/v1/files/<id>/content path; everything else is
        kept as-is so `_image_to_part` can resolve it later."""
        normalized: List[str] = []
        for ref in refs:
            if not isinstance(ref, str):
                continue
            v = ref.strip().strip("<>").strip()
            if not v:
                continue
            if v.startswith(("data:", "http://", "https://")):
                normalized.append(v)
            elif v.startswith("/"):
                normalized.append(v)
            elif v.startswith(cls._B64_SIGNATURES):
                normalized.append(cls._normalize_base64(v))
            elif cls._BARE_FILE_ID_RE.match(v):
                normalized.append(f"/api/v1/files/{v}/content")
            else:
                normalized.append(cls._normalize_base64(v))
        return normalized

    def _extract_images_from_message(self, message: Dict[str, Any]) -> List[str]:
        images = []

        # message["images"] is a first-class list of image handles in Open
        # WebUI: plain strings (data URL / base64 / file URL) or dicts such as
        # {"type": "image", "url": "..."} / {"image_url": {"url": "..."}}.
        for img in message.get("images", []) or []:
            if isinstance(img, str):
                images.append(img)
            elif isinstance(img, dict):
                for key in ("url", "image_url", "content_url", "src"):
                    value = img.get(key)
                    if isinstance(value, dict):
                        value = value.get("url")
                    if isinstance(value, str) and value:
                        images.append(value)
                        break

        # message["files"] holds Open WebUI uploads AND generated images. A
        # generated/uploaded image is referenced with a relative URL such as
        # /api/v1/files/<id>/content — which earlier versions of this pipe
        # ignored, making follow-up edits impossible.
        for file_item in message.get("files", []) or []:
            if isinstance(file_item, str):
                if self._looks_like_image_ref(file_item):
                    images.append(file_item)
                continue
            if not isinstance(file_item, dict):
                continue
            is_image = self._is_image_file_entry(file_item)
            content_type = file_item.get("content_type")
            if not content_type and isinstance(file_item.get("meta"), dict):
                content_type = file_item["meta"].get("content_type")
            has_media_meta = bool(
                file_item.get("type") or content_type
            )
            data = file_item.get("data")
            url = (
                file_item.get("url")
                or file_item.get("content_url")
                or file_item.get("path")
            )
            if not url and is_image:
                file_id = file_item.get("id")
                if isinstance(file_id, str) and file_id:
                    url = f"/api/v1/files/{file_id}/content"
            if isinstance(data, str) and data and (
                is_image or data.startswith("data:image")
            ):
                images.append(data)
                continue
            # Unlabelled entries (no type/content_type) are only trusted when
            # the reference itself looks like an image; otherwise a text/PDF
            # attachment would be mistaken for an image to edit.
            if isinstance(url, str) and url and (
                is_image or (not has_media_meta and self._looks_like_image_ref(url))
            ):
                images.append(url)

        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("image_url", "input_image", "image"):
                    url = item.get(
                        "image_url", item.get("url", item.get("image", ""))
                    )
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    if isinstance(url, str) and url:
                        images.append(url)

        if isinstance(content, str):
            # Markdown images: ![alt](url) — url may be a data URL, an
            # absolute http(s) URL, a relative Open WebUI file reference
            # (/api/v1/files/<id>/content) or a bare file id.
            for url in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", content):
                images.append(url)

        return self._dedupe_images(self._normalize_refs(images))

    @classmethod
    def _dedupe_images(cls, image_urls: List[str]) -> List[str]:
        """Drop duplicate images, including the same image returned twice in
        different encodings (e.g. two b64_json entries that only differ in
        their data-URL prefix). Some proxies/models return the identical
        image more than once (seen with Gemini 3 Pro Image / "nano banana"
        models), which would otherwise render as duplicated images."""
        seen = set()
        deduped = []
        for url in image_urls:
            key = url
            if isinstance(url, str) and url.startswith("data:"):
                try:
                    _, payload = cls._decode_data_url(url)
                    key = f"sha256:{hashlib.sha256(payload).hexdigest()}"
                except Exception:
                    pass
            if key in seen:
                continue
            seen.add(key)
            deduped.append(url)
        if len(deduped) != len(image_urls):
            print(
                f"DEBUG [Dedup] Dropped {len(image_urls) - len(deduped)} "
                f"duplicate image(s) from {len(image_urls)} returned image(s)."
            )
        return deduped

    def _find_images_in_conversation(self, messages: List[Dict]) -> List[str]:
        candidates = (
            messages if self.valves.history_images_as_input else messages[-1:]
        )
        for message in reversed(candidates):
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
        self, item: str, index: int, request=None, user: Optional[dict] = None
    ) -> Optional[Tuple[str, bytes, str]]:
        """Convert a conversation image (data URL, bare base64, remote URL, or
        Open WebUI file reference) into a multipart file part
        (filename, bytes, content_type)."""
        try:
            data_url = await self._to_data_url(item, request=request, user=user)
            if not data_url:
                return None
            mime, payload = self._decode_data_url(data_url)
        except Exception as e:
            print(f"WARNING: could not convert input image #{index}: {e}")
            return None

        ext = self.EXT_BY_MIME.get(mime, "png")
        return f"input_image_{index}.{ext}", payload, mime

    async def _to_data_url(
        self, item: str, request=None, user: Optional[dict] = None
    ) -> Optional[str]:
        """Resolve any accepted image reference to a `data:image/...` URL."""
        item = (item or "").strip()
        if item.startswith("data:"):
            return item

        if self._owui_file_id(item):
            data_url = await self._fetch_owui_file(item, request=request, user=user)
            if data_url:
                return data_url
            return None

        if item.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as dl:
                resp = await dl.get(item)
                resp.raise_for_status()
                payload = resp.content
                mime = (resp.headers.get("content-type") or "").split(";")[0]
            if not mime.startswith("image/"):
                mime = self._mime_from_bytes(payload)
            return f"data:{mime};base64,{base64.b64encode(payload).decode('utf-8')}"

        payload = base64.b64decode(
            re.sub(r"\s+", "", item) + "=" * (-len(re.sub(r"\s+", "", item)) % 4),
            validate=False,
        )
        return f"data:{self._mime_from_bytes(payload)};base64,{base64.b64encode(payload).decode('utf-8')}"

    # ------------------------------------------------------------------
    # Open WebUI file access (generated images & uploads)
    # ------------------------------------------------------------------

    @staticmethod
    def _owui_base(request) -> str:
        if request is None:
            return ""
        try:
            return str(request.base_url).rstrip("/")
        except Exception:
            return ""

    @staticmethod
    def _owui_headers(request) -> Dict[str, str]:
        """Forward the caller's own auth so Open WebUI serves the file to us."""
        headers: Dict[str, str] = {}
        if request is None:
            return headers
        for header, key in (("authorization", "Authorization"), ("cookie", "Cookie")):
            value = request.headers.get(header)
            if value:
                headers[key] = value
        return headers

    async def _fetch_owui_file(
        self, ref: str, request=None, user: Optional[dict] = None
    ) -> Optional[str]:
        """Read an Open WebUI image file reference back as a data URL.

        Accepts `/api/v1/files/<id>/content`, an absolute URL of the same
        shape, or a bare file id. Reads are authorized with the caller's own
        credentials so Open WebUI's ownership checks still apply."""
        file_id = self._owui_file_id(ref)
        if not file_id:
            return None

        base = self._owui_base(request)
        if base:
            if ref.startswith(("http://", "https://")):
                target = ref
            elif ref.startswith("/"):
                target = f"{base}{ref}"
            else:
                target = f"{base}/api/v1/files/{file_id}/content"
            try:
                async with httpx.AsyncClient(timeout=60, follow_redirects=True) as dl:
                    resp = await dl.get(target, headers=self._owui_headers(request))
                    resp.raise_for_status()
                    payload = resp.content
                    mime = (resp.headers.get("content-type") or "").split(";")[0]
                if not mime.startswith("image/"):
                    mime = self._mime_from_bytes(payload)
                return f"data:{mime};base64,{base64.b64encode(payload).decode('utf-8')}"
            except Exception as e:
                print(f"WARNING: could not read Open WebUI file {file_id}: {e}")

        # Fallback: read straight from Open WebUI's storage layer.
        try:
            from open_webui.utils.files import (  # type: ignore
                get_image_base64_from_file_id,
            )
        except Exception:
            return None
        user_obj = await self._user_model(user)
        if user_obj is None:
            return None
        try:
            data_url = await get_image_base64_from_file_id(file_id, user_obj)
            return data_url or None
        except Exception as e:
            print(f"WARNING: internal read of Open WebUI file {file_id} failed: {e}")
            return None

    @staticmethod
    async def _user_model(user: Optional[dict]):
        if not user:
            return None
        try:
            from open_webui.models.users import Users  # type: ignore
        except Exception:
            return None
        user_id = user.get("id")
        if not user_id:
            return None
        try:
            result = Users.get_user_by_id(user_id)
            if hasattr(result, "__await__"):
                result = await result
            return result
        except Exception:
            return None

    async def _persist_image_urls(
        self, image_urls: List[str], request=None, body: Optional[dict] = None
    ) -> List[str]:
        """Store generated images as Open WebUI files.

        Returns the references to render in the chat: relative
        `/api/v1/files/<id>/content` URLs when the upload succeeded, otherwise
        the original (inline) reference. Persisting matters because Open
        WebUI rewrites inline base64 out of the saved chat, which would leave
        a follow-up "add a hat" turn with no image to edit."""
        if not self.valves.persist_generated_images or request is None:
            return image_urls

        metadata_source = (body or {}).get("metadata") or {}
        metadata = {
            "source": "litellm_image_pipe",
            "chat_id": metadata_source.get("chat_id"),
            "message_id": metadata_source.get("message_id"),
        }

        persisted: List[str] = []
        for url in image_urls:
            if not (isinstance(url, str) and url.startswith("data:")):
                persisted.append(url)
                continue
            try:
                mime, payload = self._decode_data_url(url)
            except Exception as e:
                print(f"WARNING: could not decode generated image for upload: {e}")
                persisted.append(url)
                continue
            file_url = await self._upload_generated_image(
                request, payload, mime, metadata
            )
            persisted.append(file_url or url)
        return persisted

    async def _upload_generated_image(
        self,
        request,
        payload: bytes,
        mime: str,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Upload image bytes to Open WebUI; returns a relative content URL."""
        base = self._owui_base(request)
        if not base:
            return None
        ext = self.EXT_BY_MIME.get(mime, "png")
        filename = f"litellm-image-{uuid.uuid4().hex}.{ext}"
        meta = dict(metadata or {})
        meta.setdefault("source", "litellm_image_pipe")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{base}/api/v1/files/",
                    params={"process": "false"},
                    files={"file": (filename, payload, mime)},
                    data={"metadata": json.dumps(meta)},
                    headers=self._owui_headers(request),
                )
            if resp.status_code != 200:
                print(
                    f"WARNING: Open WebUI image upload failed "
                    f"(HTTP {resp.status_code}): {resp.text[:200]}"
                )
                return None
            file_id = (resp.json() or {}).get("id")
            if not file_id:
                print("WARNING: Open WebUI image upload returned no file id.")
                return None
            return f"/api/v1/files/{file_id}/content"
        except Exception as e:
            print(f"WARNING: Open WebUI image upload failed: {e}")
            return None

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
        request=None,
        user: Optional[dict] = None,
    ) -> List[str]:
        """Text-to-image -> /v1/images/generations (JSON);
        image-to-image -> /v1/images/edits (multipart)."""
        if input_images:
            url = self._api_url("images/edits")
            parts = []
            for idx, item in enumerate(input_images[: self.valves.max_input_images]):
                part = await self._image_to_part(item, idx, request=request, user=user)
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
            raise ImageRouteError(
                f"LiteLLM proxy HTTP {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )
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
        return cls._dedupe_images(image_urls)

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
        return cls._dedupe_images(image_urls)

    def _fallback_is_safe(self, exc: Exception) -> bool:
        """Only fall back to /v1/chat/completions when the images-route call
        clearly did NOT generate anything. Retrying after a timeout, a 5xx,
        or an unparseable 200 response can make expensive models (Gemini 3
        Pro Image / nano banana, gpt-image-1, ...) generate — and bill — the
        image a second time."""
        if isinstance(exc, httpx.TimeoutException):
            return False
        if isinstance(exc, httpx.HTTPError):
            return True
        status = getattr(exc, "status_code", None)
        if status in (404, 405, 501):
            return True
        if isinstance(status, int) and 400 <= status < 500:
            message = str(exc).lower()
            return "not supported" in message or "unsupported" in message
        return False

    async def _generate_image(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
        size: Optional[str],
        request=None,
        user: Optional[dict] = None,
    ) -> List[str]:
        errors = []
        try:
            return await self._generate_with_images_route(
                model_id, prompt, size, input_images, request=request, user=user
            )
        except Exception as e:
            errors.append(f"Images route error: {e}")
            if not self.valves.enable_chat_completions_fallback:
                raise
            if not self._fallback_is_safe(e):
                print(
                    f"DEBUG [Images] Skipping chat fallback (retrying could "
                    f"generate the image a second time): {e}"
                )
                raise
            print(f"DEBUG [Images] Images route unavailable, falling back: {e}")
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
    # Open WebUI background task calls (title / tags / follow-ups / ...)
    # ------------------------------------------------------------------

    TASK_MARKERS = (
        "title",
        "follow_up",
        "tags",
        "emoji",
        "query",
        "autocomplete",
        "image_prompt",
        "function_calling",
        "moa",
    )

    def _detect_background_task(self, body: dict) -> Optional[str]:
        """Return the Open WebUI task type when this request is a background
        task call (chat title, tags, follow-up suggestions, ...). Open WebUI
        marks them with metadata.task and, by default, routes them to the
        same model that handled the chat — for an image pipe that means
        extra image generations (and 429s) on every chat turn."""
        metadata = body.get("metadata") or {}
        task = str(metadata.get("task") or "").lower()
        if not task:
            return None
        for marker in self.TASK_MARKERS:
            if marker in task:
                return task
        return None

    def _task_stub_chunks(self, task: str, model_id: str) -> List[str]:
        """Minimal text answer for a task call, shaped so Open WebUI's own
        fallbacks kick in: title generation falls back to the first user
        message when the reply is not JSON, and tags/follow-ups disappear
        when the reply is an empty list. No LiteLLM call is made."""
        if "title" in task:
            content = "title"  # no JSON braces -> OWUI falls back to the user's message
        elif "tags" in task or "follow_up" in task:
            content = "[]"  # parsed as an empty list -> feature stays quiet
        else:
            content = ""
        return [self._chunk(model_id, content), self._stop_chunk(model_id)]

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

        if self.valves.handle_task_calls_locally:
            task = self._detect_background_task(body)
            if task:
                print(
                    f"DEBUG: answering Open WebUI task call '{task}' locally "
                    f"— no image generation."
                )
                for chunk in self._task_stub_chunks(task, model_id):
                    yield chunk
                return

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

            # Open WebUI may carry the current turn's uploads at the top level
            # instead of inside the last message; merge them in so an upload
            # is never missed.
            messages = body["messages"]
            extra_files = body.get("files") or []
            if extra_files and messages:
                messages = list(messages)
                last = dict(messages[-1])
                existing = list(last.get("files") or [])
                last["files"] = existing + [f for f in extra_files if f not in existing]
                messages[-1] = last

            input_images = self._find_images_in_conversation(messages)
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
                model_id, prompt, input_images, size, request=__request__, user=__user__
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

            # Persist generated images as Open WebUI files so the next turn
            # can edit them (see _persist_image_urls).
            image_urls = await self._persist_image_urls(
                image_urls, request=__request__, body=body
            )

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
