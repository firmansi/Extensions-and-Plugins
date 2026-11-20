"""
title: Qwen Image Generation & Editing (Manifold)
author: dfirmans (upgraded by AI)
version: 5.1.0
description: >
  Manifold Pipe for Alibaba Cloud Model Studio (DashScope) Qwen-Image models.
  Every model listed in the `models` valve is registered as its own selectable
  model in Open WebUI, so one function can serve the whole family:
  qwen-image-3.0 / 3.0-pro, qwen-image-2.0 / 2.0-pro, qwen-image-max,
  qwen-image-plus and the qwen-image-edit / edit-plus / edit-max series.
  Text-to-image and image-to-image (editing) are routed automatically from the
  attached image, with per-family parameter and size handling.
  v5.1.0: Open WebUI file references (uploads and previously generated
  images) are resolved back to bytes, generated results are persisted as
  Open WebUI files so later turns can edit them, background tasks are
  ignored, and image preparation runs off the event loop with aggressive
  downscaling — noticeably faster edits.
  Compatible with Open WebUI 0.11.4+.
required_open_webui_version: 0.11.4
requirements: pillow, httpx, anyio
"""

import asyncio
import base64
import io
import json
import random
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import anyio
import httpx
from fastapi import Request
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

DEFAULT_MODELS = (
    "qwen-image-3.0-pro, "
    "qwen-image-3.0, "
    "qwen-image-2.0-pro, "
    "qwen-image-2.0, "
    "qwen-image-max, "
    "qwen-image-plus, "
    "qwen-image-edit-max, "
    "qwen-image-edit-plus"
)

# Sizes accepted for Qwen image models. Validation is per-model-family:
#   - qwen-image-3.0 / 2.0 : pixel area 512*512 .. 2048*2048, ratio 1:8..8:1
#   - qwen-image-edit-plus/-max : each side 512..2048
#   - qwen-image-max / plus : fixed presets only
IMAGE_SIZES = Literal[
    "auto",
    "2048*2048",  # 1:1
    "1536*1536",  # 1:1
    "1328*1328",  # 1:1 (max/plus preset)
    "1024*1024",  # 1:1
    "2688*1536",  # 16:9 (2.0/3.0)
    "1920*1080",  # 16:9
    "1664*928",  # 16:9 (max/plus default)
    "1280*720",  # 16:9
    "1536*2688",  # 9:16 (2.0/3.0)
    "1080*1920",  # 9:16
    "928*1664",  # 9:16 (max/plus preset)
    "720*1280",  # 9:16
    "2368*1728",  # 4:3 (2.0/3.0)
    "1440*1080",  # 4:3
    "1472*1104",  # 4:3 (max/plus preset)
    "1280*960",  # 4:3
    "1728*2368",  # 3:4 (2.0/3.0)
    "1080*1440",  # 3:4
    "1104*1472",  # 3:4 (max/plus preset)
    "960*1280",  # 3:4
    "1536*1024",  # 3:2
    "1152*768",  # 3:2
    "1024*1536",  # 2:3
    "768*1152",  # 2:3
]

# Fixed size presets for the legacy qwen-image-max / qwen-image-plus families.
PRESET_SIZES = {"1664*928", "1472*1104", "1328*1328", "1104*1472", "928*1664"}

GENERATION_FAMILIES = {
    "qwen-image-3.0",
    "qwen-image-2.0",
    "qwen-image-max",
    "qwen-image-plus",
    "unknown",
}

EDITING_FAMILIES = {
    "qwen-image-3.0",
    "qwen-image-2.0",
    "qwen-image-edit",
    "qwen-image-edit-plus",
    "qwen-image-edit-max",
    "unknown",
}

MULTI_OUTPUT_FAMILIES = {
    "qwen-image-3.0",
    "qwen-image-2.0",
    "qwen-image-edit-plus",
    "qwen-image-edit-max",
    "unknown",
}


def _model_family(model: str) -> str:
    """Classify a free-form model name into a Qwen-Image family."""
    m = (model or "").strip().lower()
    if "qwen-image-3.0" in m or m.startswith("qwen-image-3"):
        return "qwen-image-3.0"
    if "qwen-image-2.0" in m or m.startswith("qwen-image-2"):
        return "qwen-image-2.0"
    if "qwen-image-edit-max" in m:
        return "qwen-image-edit-max"
    if "qwen-image-edit-plus" in m:
        return "qwen-image-edit-plus"
    if "qwen-image-edit" in m:
        return "qwen-image-edit"
    if "qwen-image-max" in m:
        return "qwen-image-max"
    if "qwen-image-plus" in m or m == "qwen-image":
        return "qwen-image-plus"
    return "unknown"


def _parse_size(size: str) -> Optional[Tuple[int, int]]:
    try:
        w, h = size.lower().split("x") if "x" in size else size.split("*")
        return int(w), int(h)
    except Exception:
        return None


def _size_is_valid(family: str, size: str) -> bool:
    """Validate a size string against the constraints of a model family."""
    if size == "auto":
        return family == "qwen-image-3.0"
    if family == "qwen-image-edit":
        return True  # size is not customizable; value is ignored later
    parsed = _parse_size(size)
    if not parsed:
        return False
    w, h = parsed
    if family in ("qwen-image-3.0", "qwen-image-2.0", "unknown"):
        if not (512 * 512 <= w * h <= 2048 * 2048):
            return False
        return max(w, h) / max(1, min(w, h)) <= 8.0
    if family in ("qwen-image-edit-plus", "qwen-image-edit-max"):
        return 512 <= w <= 2048 and 512 <= h <= 2048
    if family in ("qwen-image-max", "qwen-image-plus"):
        return size in PRESET_SIZES
    return True


def _supports_generation(model: str) -> bool:
    return _model_family(model) in GENERATION_FAMILIES


def _supports_editing(model: str) -> bool:
    return _model_family(model) in EDITING_FAMILIES


class Pipe:
    """Open WebUI manifold Pipe for Qwen-Image generation and editing."""

    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description="Your Alibaba Cloud Model Studio (DashScope) API key",
        )
        base_url: str = Field(
            default="https://dashscope-intl.aliyuncs.com/api/v1",
            description=(
                "DashScope API base URL. Default is Singapore (dashscope-intl). "
                "For China (Beijing) use https://dashscope.aliyuncs.com/api/v1. "
                "Alibaba now recommends workspace domains, e.g. "
                "https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api/v1"
            ),
        )
        models: str = Field(
            default=DEFAULT_MODELS,
            description=(
                "Comma or newline separated Qwen-Image model IDs. Each one is "
                "registered as a separate selectable model in Open WebUI. "
                "Example: qwen-image-3.0-pro, qwen-image-max, qwen-image-edit-max"
            ),
        )
        name_prefix: str = Field(
            default="Qwen/",
            description="Prefix added to each model name shown in Open WebUI.",
        )
        default_size: IMAGE_SIZES = Field(
            default="1024*1024",
            description=(
                "Default output size for generation when no size keyword is "
                "detected. Unsupported values fall back to the family default."
            ),
        )
        editing_size: IMAGE_SIZES = Field(
            default="1024*1024",
            description=(
                "Default output size for editing when no size keyword is "
                "detected. Ignored by qwen-image-edit (size is fixed)."
            ),
        )
        enable_size_detection: bool = Field(
            default=True,
            description="Detect size/aspect ratio from prompt keywords.",
        )
        prompt_extend: bool = Field(
            default=True,
            description=(
                "Enable prompt rewriting. For Qwen-Image 3.0 this also controls "
                "prompt_extend_mode and enable_thinking. Prompt rewriting runs "
                "an extra LLM pass before rendering, so it adds several seconds "
                "of latency — disable for faster responses when your prompts "
                "are already detailed."
            ),
        )
        prompt_extend_mode: Literal["direct", "agent"] = Field(
            default="direct",
            description=(
                "Qwen-Image 3.0 only. 'direct' works for T2I and I2I; 'agent' "
                "(Agent Prompt Enhancement) is T2I only and is forced back to "
                "'direct' when editing."
            ),
        )
        enable_thinking: bool = Field(
            default=False,
            description=(
                "Qwen-Image 3.0 only. Extends reasoning for better quality at "
                "the cost of latency. Requires prompt_extend enabled. The API "
                "default is true; set false for faster responses."
            ),
        )
        negative_prompt: str = Field(
            default="",
            description="Negative prompt (max 500 chars). Leave empty to omit.",
        )
        watermark: bool = Field(
            default=False,
            description="Add a Qwen-Image watermark to the output.",
        )
        seed: int = Field(
            default=-1,
            ge=-1,
            le=2147483647,
            description="Random seed (0-2147483647). Use -1 for random.",
        )
        generation_output_count: int = Field(
            default=1,
            ge=1,
            le=6,
            description=(
                "Number of images for generation (1-6). Supported by the 3.0 "
                "and 2.0 families; older max/plus/image models are fixed at 1."
            ),
        )
        edit_output_count: int = Field(
            default=1,
            ge=1,
            le=6,
            description=(
                "Number of output images when editing (1-6). qwen-image-edit "
                "(base) always returns 1."
            ),
        )
        max_reference_images: int = Field(
            default=3,
            ge=1,
            le=3,
            description="Max reference images sent for editing (Qwen supports 1-3).",
        )
        input_max_side: int = Field(
            default=2048,
            ge=512,
            le=8192,
            description=(
                "Downscale input images so their longest side is at most this "
                "many pixels before JPEG-encoding them for the API. Smaller "
                "values are much faster to prepare and upload; 2048 is "
                "plenty for Qwen editing."
            ),
        )
        persist_generated_images: bool = Field(
            default=True,
            description=(
                "Upload generated images to Open WebUI storage and return them "
                "as /api/v1/files/<id>/content references instead of inline "
                "base64 or DashScope's 24h URLs. Keeps results editable in "
                "later turns (Open WebUI rewrites inline base64 out of chat "
                "history, and DashScope URLs expire) and keeps chats small."
            ),
        )
        files_api_token: str = Field(
            default="",
            description=(
                "Optional Open WebUI API token used to fetch uploaded images "
                "when the request carries no Authorization header (e.g. "
                "direct API calls)."
            ),
        )
        prefer_async_legacy: bool = Field(
            default=False,
            description=(
                "Use the asynchronous text2image endpoint for qwen-image and "
                "qwen-image-plus instead of the synchronous endpoint. Sync is "
                "recommended and is used automatically for every other model."
            ),
        )
        timeout: int = Field(
            default=180,
            ge=30,
            le=900,
            description="HTTP timeout in seconds for generation/editing calls.",
        )
        min_request_interval: float = Field(
            default=1.5,
            ge=0.0,
            le=60.0,
            description="Minimum seconds between consecutive Qwen API calls.",
        )
        max_retries: int = Field(
            default=4,
            ge=0,
            le=10,
            description="Max retries for 429 / 5xx / network errors.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._last_qwen_call = 0.0
        self._rate_limit_lock = asyncio.Lock()
        self.tooltips = [
            "Try adjusting the style of the image, e.g., 'in the style of Van Gogh'.",
            "How about generating the same image from a different perspective?",
            "What about changing the time of day? Try 'at sunset' or 'at night'.",
            "Add a new element, for example, 'add a small fox in the foreground'.",
            "You can also remove elements that you don't like.",
            "Change the color palette, e.g., 'make the colors warmer and more saturated'.",
            "Let's change the mood of the image, try 'make it more mysterious'.",
            "Try different aspect ratios like 16:9 for widescreen or 9:16 for vertical.",
            "Upload an image to edit it! Just attach a file and describe the changes.",
        ]

    # -------------------------------------------------------------------------
    # Manifold registration
    # -------------------------------------------------------------------------

    def _parse_models(self) -> List[str]:
        ids: List[str] = []
        for part in re.split(r"[,\n]+", self.valves.models or ""):
            model_id = part.strip().strip(",; ")
            if model_id and model_id not in ids:
                ids.append(model_id)
        return ids

    @staticmethod
    def _clean_model_id(raw_model: str) -> str:
        """Open WebUI passes 'function_id.sub_id'; strip only the first segment."""
        if "." in raw_model:
            return raw_model.split(".", 1)[1]
        return raw_model

    def _resolve_model(self, raw_model: str) -> str:
        """Map the request model back to a configured Qwen model id."""
        ids = self._parse_models()
        if raw_model in ids:
            return raw_model
        cleaned = self._clean_model_id(raw_model)
        if cleaned in ids:
            return cleaned
        for model_id in ids:
            if raw_model.endswith(model_id):
                return model_id
        return cleaned

    def pipes(self) -> List[Dict[str, str]]:
        """Expose every configured model as its own Open WebUI model entry."""
        if not self.valves.api_key:
            return [
                {
                    "id": "error_no_api_key",
                    "name": f"{self.valves.name_prefix}❌ Set DashScope API Key",
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

    # -------------------------------------------------------------------------
    # Image encoding / extraction
    # -------------------------------------------------------------------------

    _MAX_B64_BYTES = 10 * 1024 * 1024

    def _downscale(self, img: Image.Image) -> Image.Image:
        """Shrink an input image to `input_max_side` before encoding.

        Encoding a 4K upload at high quality is the slowest part of an edit
        request; Qwen gains nothing from anything above ~2K.
        """
        max_side = max(512, int(self.valves.input_max_side or 2048))
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        ratio = max_side / max(w, h)
        return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert a PIL Image to a base64 data URL under Qwen's 10 MB limit."""
        img = self._downscale(img)
        if img.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(
                img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
            )
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        def _encode_jpeg(image: Image.Image, quality: int) -> Tuple[str, int]:
            buf = io.BytesIO()
            # `optimize=True` measurably slows big-image encoding for a tiny
            # size gain; the quality ladder already handles the 10 MB limit.
            image.save(buf, format="JPEG", quality=quality)
            raw = buf.getvalue()
            b64 = base64.b64encode(raw).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}", len(raw)

        def _resize_to_max(image: Image.Image, max_side: int) -> Image.Image:
            w, h = image.size
            if max(w, h) <= max_side:
                return image
            ratio = max_side / max(w, h)
            return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        for quality in (90, 80, 70, 60, 45, 30):
            data_url, raw_size = _encode_jpeg(img, quality)
            if raw_size <= self._MAX_B64_BYTES:
                return data_url

        img_resized = _resize_to_max(img, 1536)
        for quality in (80, 65, 50, 40):
            data_url, raw_size = _encode_jpeg(img_resized, quality)
            if raw_size <= self._MAX_B64_BYTES:
                return data_url

        img_small = _resize_to_max(img, 1024)
        data_url, raw_size = _encode_jpeg(img_small, 50)
        if raw_size <= self._MAX_B64_BYTES:
            return data_url

        raise ValueError(
            f"Image cannot be compressed below the 10 MB API limit "
            f"(final size: {raw_size / 1024 / 1024:.1f} MB). "
            "Please use a smaller source image."
        )

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
    def _collect_image_refs(cls, message: Dict[str, Any]) -> List[str]:
        """Collect every image reference from one message (raw strings)."""
        refs: List[str] = []

        # message["files"] holds Open WebUI uploads AND generated images —
        # usually as /api/v1/files/<id>/content, which the old data-URL-only
        # extraction never saw, so follow-up edits silently turned into
        # fresh generations.
        for file_item in message.get("files", []) or []:
            if isinstance(file_item, str):
                if cls._looks_like_image_ref(file_item):
                    refs.append(file_item)
                continue
            if not isinstance(file_item, dict):
                continue
            is_image = cls._is_image_file_entry(file_item)
            content_type = file_item.get("content_type")
            if not content_type and isinstance(file_item.get("meta"), dict):
                content_type = file_item["meta"].get("content_type")
            has_media_meta = bool(file_item.get("type") or content_type)
            data = file_item.get("data")
            url = (
                file_item.get("url")
                or file_item.get("content_url")
                or file_item.get("path")
            )
            if not url and is_image and isinstance(file_item.get("id"), str) and file_item["id"]:
                url = f"/api/v1/files/{file_item['id']}/content"
            if isinstance(data, str) and data and (
                is_image or data.startswith("data:image")
            ):
                refs.append(data)
                continue
            if isinstance(url, str) and url and (
                is_image or (not has_media_meta and cls._looks_like_image_ref(url))
            ):
                refs.append(url)

        # message["images"] in every Open WebUI shape
        for img in message.get("images", []) or []:
            if isinstance(img, str):
                refs.append(img)
            elif isinstance(img, dict):
                for key in ("url", "image_url", "content_url", "src"):
                    value = img.get(key)
                    if isinstance(value, dict):
                        value = value.get("url")
                    if isinstance(value, str) and value:
                        refs.append(value)
                        break

        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("image_url", "input_image", "image"):
                    url = item.get("image_url", item.get("url", item.get("image", "")))
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    if isinstance(url, str) and url:
                        refs.append(url)
        elif isinstance(content, str):
            # Markdown images: ![alt](url) — data URL, http(s), relative Open
            # WebUI file reference or bare file id.
            for url in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", content):
                refs.append(url)

        # normalize + dedupe while preserving order
        normalized: List[str] = []
        seen: set = set()
        for ref in refs:
            if not isinstance(ref, str):
                continue
            v = ref.strip().strip("<>").strip()
            if not v or v in seen:
                continue
            if v.startswith(("data:", "http://", "https://", "/")):
                pass
            elif v.startswith(cls._B64_SIGNATURES):
                v = f"data:image/png;base64,{v}"
            elif cls._BARE_FILE_ID_RE.match(v):
                v = f"/api/v1/files/{v}/content"
            else:
                continue
            seen.add(v)
            normalized.append(v)
        return normalized

    def _owui_headers(self, request=None) -> Dict[str, str]:
        """Caller's own credentials so Open WebUI serves the file to us."""
        headers: Dict[str, str] = {}
        if request is not None:
            auth = request.headers.get("authorization")
            cookie = request.headers.get("cookie")
            if auth:
                headers["Authorization"] = auth
            if cookie:
                headers["Cookie"] = cookie
        if "Authorization" not in headers and self.valves.files_api_token:
            headers["Authorization"] = f"Bearer {self.valves.files_api_token}"
        return headers

    async def _resolve_ref_bytes(
        self, ref: str, request=None
    ) -> Optional[bytes]:
        """Download/decode any supported image reference to raw bytes."""
        ref = (ref or "").strip()
        if not ref:
            return None

        if ref.lower().startswith("data:"):
            try:
                return base64.b64decode(ref.split(",", 1)[1])
            except Exception as e:
                self._dbg("data URI decode failed:", e)
                return None

        base = str(request.base_url).rstrip("/") if request is not None else ""

        if ref.startswith((("http://", "https://"))):
            url = ref
            src_host = urlparse(ref).netloc.lower()
            base_host = urlparse(base).netloc.lower() if base else ""
            # Only forward Open WebUI credentials to Open WebUI itself —
            # never to DashScope/CDN hosts.
            send_auth = bool(base_host) and src_host == base_host
        elif ref.startswith("/"):
            if not base:
                self._dbg("cannot resolve server file without request base url:", ref)
                return None
            url = f"{base}{ref}"
            send_auth = True
        else:
            file_id = self._owui_file_id(ref)
            if not file_id or not base:
                return None
            url = f"{base}/api/v1/files/{file_id}/content"
            send_auth = True

        headers = self._owui_headers(request) if send_auth else {}
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code == 200 and resp.content:
                return resp.content
            self._dbg(f"image fetch failed HTTP {resp.status_code}: {url[:120]}")
        except Exception as e:
            self._dbg(f"image fetch error ({url[:120]}): {e}")
        return None

    @staticmethod
    def _decode_pil(data: bytes) -> Optional[Image.Image]:
        try:
            pil_img = Image.open(io.BytesIO(data))
            pil_img.load()
            return pil_img
        except Exception as e:
            print(f"DEBUG: Error processing image: {e}")
            return None

    async def _extract_images_from_message(
        self, message: Dict[str, Any], request=None
    ) -> List[Dict[str, Any]]:
        """Extract unique uploaded images from an Open WebUI message."""
        refs = self._collect_image_refs(message)
        if not refs:
            return []

        # Resolve references concurrently; identical refs are already deduped.
        results = await asyncio.gather(
            *[self._resolve_ref_bytes(ref, request) for ref in refs]
        )

        raw_pil_images: List[Image.Image] = []
        for data in results:
            if not data:
                continue
            # PIL decode runs in a worker thread so a big upload doesn't
            # block the event loop (other chats keep responding).
            pil_img = await anyio.to_thread.run_sync(self._decode_pil, data)
            if pil_img is not None:
                raw_pil_images.append(pil_img)

        if not raw_pil_images:
            return []

        seen: set = set()
        unique_pils: List[Image.Image] = []
        for img in raw_pil_images:
            key = img.tobytes()
            if key not in seen:
                seen.add(key)
                unique_pils.append(img)

        encoded = await asyncio.gather(
            *[
                anyio.to_thread.run_sync(self._image_to_base64, pil_img)
                for pil_img in unique_pils
            ]
        )
        return [
            {"pil": pil_img, "base64_url": b64}
            for pil_img, b64 in zip(unique_pils, encoded)
        ]

    # -------------------------------------------------------------------------
    # Persist generated images as Open WebUI files
    # -------------------------------------------------------------------------

    def _dbg(self, *args) -> None:
        print("DEBUG[qwen]:", *args)

    @staticmethod
    def _mime_from_bytes(payload: bytes, fallback: str = "image/png") -> str:
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

    def _owui_upload_headers(self, request=None) -> Dict[str, str]:
        return self._owui_headers(request)

    async def _upload_to_owui(
        self,
        request,
        payload: bytes,
        mime: str,
        filename: str,
    ) -> Optional[str]:
        """Upload image bytes to Open WebUI; returns a relative content URL."""
        base = str(request.base_url).rstrip("/") if request is not None else ""
        if not base:
            self._dbg("cannot upload generated image without request base url")
            return None
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{base}/api/v1/files/",
                    params={"process": "false"},
                    files={"file": (filename, payload, mime)},
                    data={
                        "metadata": json.dumps({"source": "qwen_image_pipe"})
                    },
                    headers=self._owui_upload_headers(request),
                )
            if resp.status_code != 200:
                self._dbg(
                    f"Open WebUI image upload failed (HTTP {resp.status_code}): "
                    f"{resp.text[:200]}"
                )
                return None
            file_id = (resp.json() or {}).get("id")
            if not file_id:
                self._dbg("Open WebUI image upload returned no file id")
                return None
            return f"/api/v1/files/{file_id}/content"
        except Exception as e:
            self._dbg("Open WebUI image upload error:", e)
            return None

    async def _persist_image_urls(
        self, image_urls: List[str], request=None
    ) -> List[str]:
        """Store generated images as Open WebUI files (in parallel).

        DashScope image URLs expire after 24 hours and Open WebUI rewrites
        inline base64 out of saved chat history — both leave a later "add a
        hat" turn with nothing to edit. Stable file references fix that."""
        if not self.valves.persist_generated_images or request is None:
            return image_urls

        async def _persist_one(url: str) -> str:
            if not isinstance(url, str):
                return url
            if url.startswith("/api/v1/files/"):
                return url
            if url.startswith("data:"):
                try:
                    payload = base64.b64decode(url.split(",", 1)[1])
                except Exception as e:
                    self._dbg("could not decode generated image for upload:", e)
                    return url
            elif url.startswith(("http://", "https://")):
                payload = await self._resolve_ref_bytes(url, request)
                if not payload:
                    return url
            else:
                return url
            mime = self._mime_from_bytes(payload)
            ext = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/webp": "webp",
                "image/gif": "gif",
            }.get(mime, "png")
            file_url = await self._upload_to_owui(
                request, payload, mime, f"qwen-image-{uuid.uuid4().hex}.{ext}"
            )
            return file_url or url

        persisted = await asyncio.gather(*[_persist_one(url) for url in image_urls])
        return list(persisted)

    # -------------------------------------------------------------------------
    # Size detection / resolution
    # -------------------------------------------------------------------------

    def _detect_size_from_prompt(self, prompt: str, editing: bool = False) -> str:
        """Detect desired image size from prompt keywords."""
        prompt_lower = prompt.lower()
        default = self.valves.editing_size if editing else self.valves.default_size

        for size, patterns in {
            "2048*2048": ["2048x2048", "2048*2048"],
            "1536*1536": ["1536x1536", "1536*1536"],
            "1328*1328": ["1328x1328", "1328*1328"],
            "1024*1024": ["1024x1024", "1024*1024"],
            "2688*1536": ["2688x1536", "2688*1536"],
            "1920*1080": ["1920x1080", "1920*1080"],
            "1664*928": ["1664x928", "1664*928"],
            "1280*720": ["1280x720", "1280*720"],
            "1536*2688": ["1536x2688", "1536*2688"],
            "1080*1920": ["1080x1920", "1080*1920"],
            "928*1664": ["928x1664", "928*1664"],
            "720*1280": ["720x1280", "720*1280"],
            "2368*1728": ["2368x1728", "2368*1728"],
            "1440*1080": ["1440x1080", "1440*1080"],
            "1472*1104": ["1472x1104", "1472*1104"],
            "1280*960": ["1280x960", "1280*960"],
            "1728*2368": ["1728x2368", "1728*2368"],
            "1080*1440": ["1080x1440", "1080*1440"],
            "1104*1472": ["1104x1472", "1104*1472"],
            "960*1280": ["960x1280", "960*1280"],
            "1536*1024": ["1536x1024", "1536*1024"],
            "1152*768": ["1152x768", "1152*768"],
            "1024*1536": ["1024x1536", "1024*1536"],
            "768*1152": ["768x1152", "768*1152"],
        }.items():
            if any(p in prompt_lower for p in patterns):
                return size

        vertical_words = ("vertical", "portrait", "tall")
        ar_map = {
            (
                "16:9",
                "16/9",
                "widescreen",
                "landscape",
                "wide",
                "banner",
                "header",
                "desktop",
                "monitor",
            ): ("1920*1080", "1080*1920"),
            ("9:16", "9/16", "phone", "mobile", "story", "tiktok", "reels"): (
                "1080*1920",
                "1920*1080",
            ),
            ("4:3", "4/3", "four by three"): ("1440*1080", "1080*1440"),
            ("3:4", "3/4", "three by four"): ("1080*1440", "1440*1080"),
            ("3:2", "3/2", "three by two"): ("1536*1024", "1024*1536"),
            ("2:3", "2/3", "two by three"): ("1024*1536", "1536*1024"),
            ("1:1", "square", "instagram post", "profile picture", "avatar"): (
                "1024*1024",
                "1024*1024",
            ),
        }
        for keywords, (landscape_size, portrait_size) in ar_map.items():
            for kw in keywords:
                # Use word boundary regex to avoid false positives like "1:1" matching "11:11"
                pattern = r'(?<!\d)' + re.escape(kw) + r'(?!\d)'
                if re.search(pattern, prompt_lower):
                    if any(v in prompt_lower for v in vertical_words):
                        return portrait_size
                    return landscape_size

        return default

    def _resolve_size(self, model: str, requested: str, editing: bool) -> Optional[str]:
        """Return a size valid for the model family, or None to omit `size`."""
        family = _model_family(model)
        if family == "qwen-image-edit":
            return None  # not customizable
        if requested and _size_is_valid(family, requested):
            return requested
        if family in ("qwen-image-3.0", "qwen-image-2.0", "unknown"):
            return "1024*1024"
        if family in ("qwen-image-edit-plus", "qwen-image-edit-max"):
            return "1024*1024"
        if family in ("qwen-image-max", "qwen-image-plus"):
            return "1664*928"
        return requested or "1024*1024"

    def _effective_output_count(self, model: str, requested: int, editing: bool) -> int:
        """Clamp the requested count to what the model family can actually return."""
        family = _model_family(model)
        if family in ("qwen-image-edit", "qwen-image-max", "qwen-image-plus"):
            return 1
        return max(1, min(6, requested))

    # -------------------------------------------------------------------------
    # Qwen API client
    # -------------------------------------------------------------------------

    @staticmethod
    def _async_supported(model: str) -> bool:
        return _model_family(model) == "qwen-image-plus"

    def _api_url(self, path: str) -> str:
        return f"{self.valves.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _headers(self, async_mode: bool = False) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if async_mode:
            headers["X-DashScope-Async"] = "enable"
        return headers

    async def _post_json(
        self, url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        async def _request() -> httpx.Response:
            timeout = httpx.Timeout(self.valves.timeout, connect=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(url, headers=headers, json=payload)

        resp = await self._retry_with_backoff(_request)
        if resp.status_code != 200:
            text = resp.text
            try:
                data = resp.json()
                msg = (
                    data.get("message")
                    or data.get("error", {}).get("message")
                    or text
                )
                code = data.get("code", "N/A")
            except Exception:
                msg = text
                code = "N/A"
            raise Exception(
                f"Qwen API error – HTTP {resp.status_code} (code: {code}): {msg}"
            )
        return resp.json()

    async def _call_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._api_url("/services/aigc/multimodal-generation/generation")
        await self._rate_limit(self.valves.min_request_interval)
        return await self._post_json(url, self._headers(), payload)

    async def _call_async(
        self,
        model: str,
        prompt: str,
        parameters: Dict[str, Any],
        emitter=None,
    ) -> Dict[str, Any]:
        url = self._api_url("/services/aigc/text2image/image-synthesis")
        async_payload = {
            "model": model,
            "input": {"prompt": prompt},
            "parameters": {k: v for k, v in parameters.items() if k != "n"},
        }
        await self._rate_limit(self.valves.min_request_interval)
        data = await self._post_json(url, self._headers(async_mode=True), async_payload)
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise ValueError(
                "No task_id returned from async submission: "
                f"{json.dumps(data, default=str)}"
            )
        return await self._poll_async_task(task_id, emitter)

    # -------------------------------------------------------------------------
    # Retry / rate-limit helpers
    # -------------------------------------------------------------------------

    async def _rate_limit(self, min_interval: float = 1.5) -> None:
        async with self._rate_limit_lock:
            now = time.monotonic()
            elapsed = now - self._last_qwen_call
            if elapsed < min_interval:
                await anyio.sleep(min_interval - elapsed)
            self._last_qwen_call = time.monotonic()

    async def _retry_with_backoff(
        self,
        fn,
        max_retries: Optional[int] = None,
        base_delay: float = 1.5,
        max_delay: float = 30.0,
    ) -> httpx.Response:
        """Retry an async HTTP request on 429/5xx with exponential backoff + jitter."""
        retries = max_retries if max_retries is not None else self.valves.max_retries
        last_exception: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = await fn()
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exception = Exception(
                        f"HTTP {resp.status_code} on attempt {attempt + 1}"
                    )
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(float(retry_after), base_delay)
                        except ValueError:
                            delay = base_delay * (2**attempt)
                    else:
                        delay = min(base_delay * (2**attempt), max_delay)
                        delay *= 1.0 + random.random() * 0.25
                    print(
                        f"DEBUG: Retrying after {delay:.1f}s due to HTTP {resp.status_code}"
                    )
                    await anyio.sleep(delay)
                    continue
                return resp
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                last_exception = exc
                delay = min(base_delay * (2**attempt), max_delay) * (
                    1.0 + random.random() * 0.25
                )
                print(f"DEBUG: Network error, retrying after {delay:.1f}s: {exc}")
                await anyio.sleep(delay)

        raise Exception(
            f"Qwen API request failed after {retries} attempts: {last_exception}"
        )

    async def _poll_async_task(self, task_id: str, emitter=None) -> Dict[str, Any]:
        """Poll an async task until it succeeds, fails, or times out."""
        url = self._api_url(f"/tasks/{task_id}")
        headers = {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Accept": "application/json",
        }
        max_attempts = 60
        poll_interval = 1.5

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0)
        ) as client:
            for attempt in range(max_attempts):
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    raise Exception(
                        f"Task polling error – HTTP {resp.status_code}: {resp.text}"
                    )

                data = resp.json()
                status = data.get("output", {}).get("task_status") or data.get(
                    "output", {}
                ).get("status", "UNKNOWN")

                if status in ("SUCCEEDED", "SUCCESS"):
                    return data
                if status in ("FAILED", "FAILURE", "CANCELLED"):
                    code = data.get("output", {}).get("code", "N/A")
                    msg = data.get("output", {}).get(
                        "message", json.dumps(data, default=str)
                    )
                    raise Exception(
                        f"Async task failed (status={status}, code={code}): {msg}"
                    )

                if emitter:
                    await emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": (
                                    f"⏳ Waiting for image... "
                                    f"({attempt + 1}/{max_attempts})"
                                ),
                                "done": False,
                            },
                        }
                    )

                await anyio.sleep(poll_interval)

        raise TimeoutError(
            f"Async task {task_id} did not complete within the polling window"
        )

    # -------------------------------------------------------------------------
    # Payload construction
    # -------------------------------------------------------------------------

    def _build_parameters(
        self, model: str, size: Optional[str], n: int, editing: bool
    ) -> Dict[str, Any]:
        """Build the parameters object respecting model-family constraints."""
        family = _model_family(model)
        params: Dict[str, Any] = {}

        if size and family != "qwen-image-edit":
            params["size"] = size

        neg = (self.valves.negative_prompt or "").strip()
        if neg:
            params["negative_prompt"] = neg[:500]

        if self.valves.watermark:
            params["watermark"] = True

        if self.valves.seed >= 0:
            params["seed"] = self.valves.seed

        # Base qwen-image-edit: no n, no prompt_extend, no size.
        if family == "qwen-image-edit":
            return params

        # Qwen-Image 3.0: prompt_extend_mode + enable_thinking.
        if family == "qwen-image-3.0":
            if n and n > 1:
                params["n"] = max(1, min(6, n))
            params["prompt_extend"] = bool(self.valves.prompt_extend)
            if self.valves.prompt_extend:
                mode = self.valves.prompt_extend_mode
                if editing:
                    mode = "direct"
                params["prompt_extend_mode"] = mode
                params["enable_thinking"] = bool(self.valves.enable_thinking)
            return params

        # 2.0 / edit-plus / edit-max / unknown: n + prompt_extend.
        if family in (
            "qwen-image-2.0",
            "qwen-image-edit-plus",
            "qwen-image-edit-max",
            "unknown",
        ):
            if n and n > 1:
                params["n"] = max(1, min(6, n))
            params["prompt_extend"] = bool(self.valves.prompt_extend)
            return params

        # Legacy max / plus / image: n fixed at 1 (omit).
        params["prompt_extend"] = bool(self.valves.prompt_extend)
        return params

    async def _generate_image(
        self, model: str, prompt: str, size: Optional[str], emitter=None
    ) -> List[str]:
        """Generate images from a text prompt. Returns image URLs."""
        n = self._effective_output_count(
            model, self.valves.generation_output_count, editing=False
        )
        parameters = self._build_parameters(model, size, n, editing=False)
        payload = {
            "model": model,
            "input": {
                "messages": [{"role": "user", "content": [{"text": prompt}]}]
            },
            "parameters": parameters,
        }

        print(f"DEBUG: [GENERATE] model={model}, size={size}, n={parameters.get('n', 1)}")

        if self._async_supported(model) and self.valves.prefer_async_legacy:
            result = await self._call_async(model, prompt, parameters, emitter)
            return self._extract_all_image_urls(result)

        try:
            result = await self._call_sync(payload)
        except Exception as exc:
            # Some legacy plus deployments only accept the async endpoint.
            if self._async_supported(model):
                print(f"DEBUG: sync failed ({exc}); retrying via async endpoint")
                result = await self._call_async(model, prompt, parameters, emitter)
            else:
                raise
        return self._extract_all_image_urls(result)

    async def _edit_image(
        self,
        model: str,
        prompt: str,
        base64_images: List[str],
        size: Optional[str],
    ) -> List[str]:
        """Edit one or more reference images. Returns image URLs."""
        n = self._effective_output_count(
            model, self.valves.edit_output_count, editing=True
        )
        max_refs = max(1, min(3, self.valves.max_reference_images))

        content: List[Dict[str, str]] = [
            {"image": url} for url in base64_images[:max_refs]
        ]
        content.append({"text": prompt})

        parameters = self._build_parameters(model, size, n, editing=True)
        payload = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": parameters,
        }

        print(
            f"DEBUG: [EDIT] model={model}, size={size}, "
            f"images={len(base64_images[:max_refs])}, n={parameters.get('n', 1)}"
        )
        result = await self._call_sync(payload)
        return self._extract_all_image_urls(result)

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    def _extract_all_image_urls(self, result: dict) -> List[str]:
        """Extract image URLs from any known Qwen response shape."""
        urls: List[str] = []

        # DashScope sync: output.choices[].message.content[].image
        for choice in result.get("output", {}).get("choices", []):
            for item in choice.get("message", {}).get("content", []):
                if isinstance(item, dict) and item.get("image"):
                    urls.append(item["image"])

        # DashScope async: output.results[].url
        if not urls:
            for item in result.get("output", {}).get("results", []):
                if isinstance(item, dict) and item.get("url"):
                    urls.append(item["url"])

        # OpenAI-compatible image API: data[].url / data[].b64_json
        if not urls:
            for item in result.get("data", []) or []:
                if not isinstance(item, dict):
                    continue
                if item.get("url"):
                    urls.append(item["url"])
                elif item.get("b64_json"):
                    media_type = item.get("media_type", "image/png")
                    urls.append(f"data:{media_type};base64,{item['b64_json']}")

        if not urls:
            raise ValueError(
                "No image URLs found in Qwen response: "
                f"{json.dumps(result, indent=2, default=str)}"
            )
        return urls

    # -------------------------------------------------------------------------
    # Rendering / errors
    # -------------------------------------------------------------------------

    def _render(self, model: str, urls: List[str], size: Optional[str], editing: bool) -> str:
        label = "edited" if editing else "generated"
        header = f"**✏️ Here is your {label} image:**\n" if editing else (
            "**🖼️ Here is your generated image:**\n"
        )

        blocks: List[str] = []
        for idx, url in enumerate(urls, start=1):
            if len(urls) > 1:
                caption = f"Variation {idx}"
            else:
                caption = "Edited Image" if editing else "Generated Image"
            blocks.append(f"![{caption}]({url})")

        shown_size = size or "auto (input-based)"
        meta = (
            f"**Size:** `{shown_size}` | **Model:** `{model}` | "
            f"**Outputs:** `{len(urls)}`"
        )
        tip = random.choice(self.tooltips)
        return f"{header}\n" + "\n\n".join(blocks) + f"\n\n{meta}\n\n💡 _Tip: {tip}_"

    def _error(self, message: str) -> str:
        return f"❌ **Error:**\n\n{message}"

    @staticmethod
    def _model_list_help(models: List[str], family_filter) -> str:
        return ", ".join(f"`{m}`" for m in models if family_filter(m))

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: Dict[str, Any],
        __event_emitter__=None,
        __task__=None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Route to generation or editing based on whether an image is attached."""

        # Open WebUI background tasks (chat title, tags, follow-up questions)
        # also call the selected model. Generating images for them wastes a
        # paid API call, makes the next real request wait behind it and is a
        # common reason the pipe "feels slow". Reply with nothing instead.
        if __task__ or (__metadata__ or {}).get("task"):
            return

        if not self.valves.api_key:
            yield self._error(
                "DashScope API key is missing. Set it in the function's Valves."
            )
            return

        model = self._resolve_model(body.get("model", "") or "")
        if not model:
            yield self._error("Could not determine the selected Qwen model.")
            return

        try:
            messages = body.get("messages", []) or []
            last_message = messages[-1] if messages else {}
            raw_content = last_message.get("content", "")

            if isinstance(raw_content, list):
                user_prompt = "".join(
                    item.get("text", "")
                    for item in raw_content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                user_prompt = str(raw_content)
            user_prompt = user_prompt.strip()

            if not user_prompt:
                yield self._error("No text prompt found in the last message.")
                return

            # Look for images in the current message, then earlier messages.
            # Open WebUI may carry the current turn's uploads at the top level
            # instead of inside the last message; merge them in first.
            messages = list(messages)
            last_message = dict(messages[-1]) if messages else {}
            extra_files = body.get("files") or []
            if extra_files:
                existing = list(last_message.get("files") or [])
                last_message["files"] = existing + [
                    f for f in extra_files if f not in existing
                ]
                messages[-1] = last_message

            images = await self._extract_images_from_message(last_message, __request__)
            if not images:
                for msg in reversed(messages[:-1]):
                    found = await self._extract_images_from_message(msg, __request__)
                    if found:
                        images = found
                        break

            is_editing = bool(images)
            configured = self._parse_models()

            if is_editing and not _supports_editing(model):
                editable = self._model_list_help(configured, _supports_editing)
                yield self._error(
                    f"Model `{model}` does not support image editing.\n\n"
                    f"Switch to an edit-capable model: {editable or 'none configured'}."
                )
                return

            if not is_editing and not _supports_generation(model):
                generatable = self._model_list_help(configured, _supports_generation)
                yield self._error(
                    f"Model `{model}` is an editing-only model and needs an image.\n\n"
                    f"Attach an image, or switch to a generation model: "
                    f"{generatable or 'none configured'}."
                )
                return

            if self.valves.enable_size_detection:
                detected = self._detect_size_from_prompt(
                    user_prompt, editing=is_editing
                )
            else:
                detected = (
                    self.valves.editing_size if is_editing else self.valves.default_size
                )
            size = self._resolve_size(model, detected, is_editing)

            if __event_emitter__:
                shown_size = size or "auto"
                if is_editing:
                    n = self._effective_output_count(
                        model, self.valves.edit_output_count, editing=True
                    )
                    status = (
                        f"✏️ Editing image → {n} result(s) at {shown_size} "
                        f"using {model}…"
                    )
                else:
                    n = self._effective_output_count(
                        model, self.valves.generation_output_count, editing=False
                    )
                    status = (
                        f"🎨 Generating {n} image(s) at {shown_size} using {model}…"
                    )
                await __event_emitter__(
                    {"type": "status", "data": {"description": status, "done": False}}
                )

            if is_editing:
                base64_urls = [img["base64_url"] for img in images]
                urls = await self._edit_image(model, user_prompt, base64_urls, size)
            else:
                urls = await self._generate_image(
                    model, user_prompt, size, __event_emitter__
                )

            # Persist results as Open WebUI files so later turns can edit them
            # (DashScope URLs expire and Open WebUI rewrites inline base64).
            urls = await self._persist_image_urls(urls, __request__)

            yield self._render(model, urls, size, is_editing)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "✅ Image processed successfully",
                            "done": True,
                        },
                    }
                )

        except Exception as err:
            print(f"ERROR: {err}")
            yield self._error(str(err))
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "❌ An error occurred", "done": True},
                    }
                )
