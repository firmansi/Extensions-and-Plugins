"""
title: Qwen Image Generation & Editing (Manifold)
author: dfirmans (upgraded by AI)
version: 5.0.0
description: >
  Manifold Pipe for Alibaba Cloud Model Studio (DashScope) Qwen-Image models.
  Every model listed in the `models` valve is registered as its own selectable
  model in Open WebUI, so one function can serve the whole family:
  qwen-image-3.0 / 3.0-pro, qwen-image-2.0 / 2.0-pro, qwen-image-max,
  qwen-image-plus and the qwen-image-edit / edit-plus / edit-max series.
  Text-to-image and image-to-image (editing) are routed automatically from the
  attached image, with per-family parameter and size handling.
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
            default="2048*2048",
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
                "prompt_extend_mode and enable_thinking."
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

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert a PIL Image to a base64 data URL under Qwen's 10 MB limit."""
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
            image.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
            b64 = base64.b64encode(raw).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}", len(raw)

        def _resize_to_max(image: Image.Image, max_side: int) -> Image.Image:
            w, h = image.size
            if max(w, h) <= max_side:
                return image
            ratio = max_side / max(w, h)
            return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        for quality in (95, 85, 75, 60, 45, 30):
            data_url, raw_size = _encode_jpeg(img, quality)
            if raw_size <= self._MAX_B64_BYTES:
                return data_url

        img_resized = _resize_to_max(img, 2048)
        for quality in (85, 70, 55, 40):
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

    def _extract_images_from_message(
        self, message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract unique uploaded images from an Open WebUI message."""
        raw_pil_images: List[Image.Image] = []

        def _add_pil_from_b64_url(data_url: str):
            try:
                b64_data = data_url.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_data, validate=True)
                pil_img = Image.open(io.BytesIO(image_bytes))
                pil_img.load()
                raw_pil_images.append(pil_img)
            except Exception as e:
                print(f"DEBUG: Error processing image: {e}")

        def _add_from_string(value: str):
            if not isinstance(value, str):
                return
            if value.startswith("data:image"):
                _add_pil_from_b64_url(value)

        # 1. Open WebUI file uploads and image payloads
        for file_item in message.get("files", []) or []:
            if isinstance(file_item, dict):
                _add_from_string(file_item.get("data", ""))

        for image_item in message.get("images", []) or []:
            _add_from_string(image_item)

        # 2. content list (image_url type or direct image field)
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    _add_from_string(item.get("image_url", {}).get("url", ""))
                elif "image" in item:
                    _add_from_string(item["image"])

        # 3. Markdown images embedded in string content
        if isinstance(content, str):
            for data_url in re.findall(
                r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)", content
            ):
                _add_from_string(data_url)

        if not raw_pil_images:
            return []

        seen: set = set()
        unique_pils: List[Image.Image] = []
        for img in raw_pil_images:
            key = img.tobytes()
            if key not in seen:
                seen.add(key)
                unique_pils.append(img)

        return [
            {"pil": pil_img, "base64_url": self._image_to_base64(pil_img)}
            for pil_img in unique_pils
        ]

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
                "2048*2048",
                "2048*2048",
            ),
        }
        for keywords, (landscape_size, portrait_size) in ar_map.items():
            if any(kw in prompt_lower for kw in keywords):
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
            return "1024*1024" if editing else "2048*2048"
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
        poll_interval = 2.0

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
    ) -> AsyncGenerator[str, None]:
        """Route to generation or editing based on whether an image is attached."""

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
            images = self._extract_images_from_message(last_message)
            if not images:
                for msg in reversed(messages[:-1]):
                    found = self._extract_images_from_message(msg)
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
