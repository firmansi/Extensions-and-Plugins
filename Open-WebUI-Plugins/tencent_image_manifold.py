"""
title: Tencent TokenHub Image Generation & Editing (Manifold)
author: dfirmans (upgraded by AI)
version: 1.0.0
description: >
  Manifold Pipe for Tencent Cloud LLM Service TokenHub image models
  (https://tokenhub-intl.tencentcloudmaas.com/v1). Every model listed in the
  `models` valve is registered as its own selectable model in Open WebUI, so
  one function can serve the whole family: Tencent Hunyuan Hy-Image
  (hy-image-v3 / hy-image-v3.5-preview), WAND-Vega-Image (lite / flash / pro),
  plus third-party models hosted through the TokenHub AI gateway such as
  Seedream-Image v5.0 (pro / lite), Vidu-Image-q2 and Kling-Image.
  Text-to-image and image-to-image (editing) are routed automatically from the
  attached image, with per-family endpoint, size and sampling handling built on
  the official docs for docs.product/1300 (Hy Image, WAND, Vidu, Seedream).
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
    "hy-image-v3, "
    "hy-image-v3.5-preview, "
    "seedream-image-v5.0-pro, "
    "seedream-image-v5.0-lite, "
    "vidu-image-q2, "
    "kling-image-v3, "
    "kling-image-o1, "
    "kling-image-v3-omni, "
    "wand-vega-image-lite, "
    "wand-vega-image-flash, "
    "wand-vega-image-pro"
)

# Common output sizes. Not every value is valid for every family; each family
# validates/normalizes the requested size. Seedream v5 also accepts the
# resolution tiers 1K / 1.5K / 2K / 3K / 4K natively.
IMAGE_SIZES = Literal[
    "1024x1024",  # 1:1
    "2048x2048",  # 1:1 (Seedream/WAND up to 2K/4K)
    "1280x720",  # 16:9 (Hunyuan preset)
    "720x1280",  # 9:16 (Hunyuan preset)
    "1024x768",  # 4:3 (Hunyuan preset)
    "768x1024",  # 3:4 (Hunyuan preset)
    "1152x896",  # 9:7 (Hunyuan preset)
    "896x1152",  # 7:9 (Hunyuan preset)
    "1344x704",  # ~19:10 (Hunyuan preset)
    "704x1344",  # ~10:19 (Hunyuan preset)
    "1536x512",  # 3:1 (Hunyuan banner preset)
    "512x1536",  # 1:3 (Hunyuan story preset)
    "2048x512",  # 4:1 (Hunyuan ultra-wide preset)
    "512x2048",  # 1:4 (Hunyuan tall preset)
    "1920x1080",  # 16:9 HD
    "1080x1920",  # 9:16
    "1536x1024",  # 3:2
    "1024x1536",  # 2:3
    "1440x2160",  # 2:3 2K
    "2160x1440",  # 3:2 2K
    "1440x1080",  # 4:3
    "1080x1440",  # 3:4
    "2560x1440",  # 16:9 2.5K (Seedream-lite pixel floor)
    "1440x2560",  # 9:16 2.5K
    "4096x4096",  # 1:1 4K (Seedream-lite / WAND)
    "1K",
    "1.5K",
    "2K",
    "3K",
    "4K",
]

# Size presets locked for the WAND-Vega family (resolution tier known).
WAND_PRESET_SIZES = {
    "1024x1024",
    "2048x2048",
    "4096x4096",
    "1024x1536",
    "1440x2160",
    "3328x4992",
    "1536x1024",
    "2160x1440",
    "4992x3328",
    "1080x1440",
    "1440x1920",
    "2160x2880",
    "3584x4800",
    "1440x1080",
    "1920x1440",
    "2880x2160",
    "4800x3584",
    "1080x1920",
    "1920x1080",
    "1024x2187",
    "2187x1024",
}

# Vidu-Image-q2 aspect ratio options.
VIDU_ASPECTS = Literal["16:9", "9:16", "1:1", "3:4", "4:3", "21:9", "2:3", "3:2", "auto"]
VIDU_RESOLUTIONS = Literal["1080p", "2K", "4K"]

# Seedream v5 resolution tiers accepted by the `size` field.
SEEDREAM_PRO_TIERS = {"1K", "1.5K", "2K"}
SEEDREAM_LITE_TIERS = {"2K", "3K", "4K"}

# Pixmap constraints per Seedream tier (used only when a pixel size is given).
SEEDREAM_PIXEL_RANGE = {
    "seedream-image-v5.0-pro": (921600, 4624220),
    "seedream-image-v5.0-lite": (3686400, 16777216),
}

GENERATION_AND_EDITING_FAMILIES = {
    "hunyuan",
    "hunyuan-v3.5",
    "seedream",
    "vidu",
    "kling",
    "wand",
    "unknown",
}


def _model_family(model: str) -> str:
    """Classify a free-form model name into a TokenHub image family."""
    m = (model or "").strip().lower()
    if m.startswith("hy-image-v3.5") or "3.5-preview" in m:
        return "hunyuan-v3.5"
    if "hy-image" in m or "hunyuan" in m:
        return "hunyuan"
    if "seedream" in m:
        return "seedream"
    if "vidu" in m:
        return "vidu"
    if "kling-image" in m:
        return "kling"
    if "wand-vega" in m or "vega" in m:
        return "wand"
    return "unknown"


def _family_endpoint(family: str) -> str:
    """Submit endpoint on the TokenHub gateway for a model family."""
    if family == "hunyuan":
        return "/v1/wand/hunyuan-image/v3-generation"
    if family == "hunyuan-v3.5":
        return "/v1/wand/hunyuan-image/v35-generation"
    if family == "seedream":
        return "/v1/wand/si-image/generation"
    if family == "vidu":
        return "/v1/wand/vidu-image/generation"
    if family == "kling":
        return "/v1/wand/kling-image/generation"
    if family == "wand":
        return "/v1/wand/vega-images/generations"
    raise ValueError(f"No known TokenHub endpoint for model family '{family}'")


def _family_is_async(family: str) -> bool:
    return family in ("vidu", "wand")


def _family_task_endpoint(family: str, task_id: str) -> str:
    if family == "vidu":
        return f"/v1/wand/vidu-image/tasks/{task_id}"
    if family == "kling":
        return f"/v1/wand/kling-image/tasks/{task_id}"
    if family == "wand":
        return f"/v1/wand/vega-images/tasks/{task_id}"
    raise ValueError(f"Family '{family}' has no task endpoint")


def _max_reference_images(model: str) -> int:
    """Upper bound of reference images each model accepts."""
    family = _model_family(model)
    if family == "hunyuan":
        return 3
    if family == "hunyuan-v3.5":
        return 5
    if family == "seedream":
        return 10 if "lite" not in model else 14
    if family == "vidu":
        return 7
    if family == "kling":
        return 10
    if family == "wand":
        return 3 if "lite" in model else 6
    return 3


def _max_prompt_chars(model: str) -> int:
    family = _model_family(model)
    if family in ("hunyuan", "hunyuan-v3.5"):
        return 8192
    if family == "seedream":
        return 600
    if family in ("vidu", "wand"):
        return 2000
    if family == "kling":
        return 2500
    return 4096


def _parse_size(size: str) -> Optional[Tuple[int, int]]:
    if not size:
        return None
    s = str(size).strip().lower().replace("*", "x")
    if "x" not in s:
        return None
    try:
        w, h = s.split("x")
        return int(w.strip()), int(h.strip())
    except Exception:
        return None


def _pixel_ratio_ok(w: int, h: int) -> bool:
    return max(w, h) / max(1, min(w, h)) <= 16.0 and max(w, h) / max(1, min(w, h)) >= 1 / 16.0


def _size_is_valid(model: str, size: str) -> bool:
    """Validate a size string against the constraints of a model."""
    if not size:
        return False
    family = _model_family(model)
    if family in ("vidu", "kling"):
        return True  # these families use aspect ratio instead of pixels
    if family in ("hunyuan", "hunyuan-v3.5"):
        parsed = _parse_size(size)
        if not parsed:
            return False
        w, h = parsed
        if family == "hunyuan-v3.5":
            # v35 (Hy-Image 3.5): sides 256..8192, area <= 4096x4096
            return (
                256 <= w <= 8192
                and 256 <= h <= 8192
                and w * h <= 4096 * 4096
            )
        return 512 <= w <= 2048 and 512 <= h <= 2048 and w * h <= 1024 * 1024
    if family == "seedream":
        s = str(size).strip().upper().replace("*", "X")
        if "X" not in s:
            tiers = SEEDREAM_PRO_TIERS | SEEDREAM_LITE_TIERS
            return s in tiers
        parsed = _parse_size(s)
        if not parsed:
            return False
        low, high = SEEDREAM_PIXEL_RANGE.get(model, (921600, 16777216))
        area = parsed[0] * parsed[1]
        return low <= area <= high and _pixel_ratio_ok(*parsed)
    if family == "wand":
        return str(size).strip().lower().replace("*", "x") in WAND_PRESET_SIZES
    # unknown family: accept anything that parses
    return _parse_size(size) is not None


def _supports_generation(model: str) -> bool:
    return _model_family(model) in GENERATION_AND_EDITING_FAMILIES


def _supports_editing(model: str) -> bool:
    return _model_family(model) in GENERATION_AND_EDITING_FAMILIES


class Pipe:
    """Open WebUI manifold Pipe for Tencent TokenHub image models."""

    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description="Your Tencent Cloud TokenHub API key",
        )
        base_url: str = Field(
            default="https://tokenhub-intl.tencentcloudmaas.com/v1",
            description=(
                "TokenHub API base URL. Default is the international endpoint. "
                "For China (Beijing) use https://tokenhub.tencentcloudmaas.com/v1. "
                "Both forms work: with or without the trailing /v1 (the plugin "
                "adds exactly one /v1 automatically)."
            ),
        )
        models: str = Field(
            default=DEFAULT_MODELS,
            description=(
                "Comma or newline separated TokenHub image model IDs. Each one "
                "is registered as a separate selectable model in Open WebUI.\n"
                "Hunyuan: hy-image-v3, hy-image-v3.5-preview\n"
                "WAND-Vega: wand-vega-image-lite, wand-vega-image-flash, wand-vega-image-pro\n"
                "Seedream (hosted): seedream-image-v5.0-pro, seedream-image-v5.0-lite\n"
                "Vidu (hosted): vidu-image-q2\n"
                "Kling (hosted): kling-image-v3, kling-image-o1, kling-image-v3-omni"
            ),
        )
        name_prefix: str = Field(
            default="Tencent/",
            description="Prefix added to each model name shown in Open WebUI.",
        )
        default_size: IMAGE_SIZES = Field(
            default="1024x1024",
            description=(
                "Default output size for generation when no size keyword is "
                "detected. Seedream v5 also accepts 1K / 1.5K / 2K / 3K / 4K "
                "tiers. Unsupported values fall back to the family default."
            ),
        )
        editing_size: IMAGE_SIZES = Field(
            default="1024x1024",
            description=(
                "Default output size for editing when no size keyword is "
                "detected. Vidu ignores this and uses aspect_ratio instead."
            ),
        )
        vidu_resolution: VIDU_RESOLUTIONS = Field(
            default="1080p",
            description="Vidu-Image-q2 output resolution tier (1080p / 2K / 4K).",
        )
        kling_resolution: Literal["1K", "2K"] = Field(
            default="2K",
            description="Kling-Image output resolution tier (1K / 2K).",
        )
        enable_size_detection: bool = Field(
            default=True,
            description="Detect size/aspect ratio from prompt keywords.",
        )
        revise: bool = Field(
            default=True,
            description=(
                "Hunyuan only. Rewrite/optimize the prompt (revise). "
                "Recommended to keep enabled."
            ),
        )
        footnote: str = Field(
            default="",
            description=(
                "Hunyuan only. Custom watermark text (max 16 chars) drawn in "
                "the bottom-right corner. Leave empty to omit."
            ),
        )
        optimize_prompt_mode: Literal["standard", "fast"] = Field(
            default="standard",
            description=(
                "Seedream v5 only. Prompt optimization mode. 'fast' is not "
                "supported by seedream-image-v5.0-lite."
            ),
        )
        seedream_output_format: Literal["png", "jpeg"] = Field(
            default="jpeg",
            description="Seedream v5 only. File format of the generated image.",
        )
        watermark: bool = Field(
            default=False,
            description=(
                "Seedream v5 only. Add an 'AI-generated' watermark to the "
                "bottom-right corner of the output."
            ),
        )
        enable_sequential_generation: bool = Field(
            default=False,
            description=(
                "Seedream v5.0-lite only. Enable group image mode so a "
                "content-related set of images can be returned."
            ),
        )
        sequential_max_images: int = Field(
            default=4,
            ge=1,
            le=15,
            description="Seedream v5.0-lite group mode: maximum images (1-15).",
        )
        seed: int = Field(
            default=0,
            ge=0,
            le=4294967295,
            description=(
                "Random seed (1-4294967295). Hunyuan/Vidu only take effect "
                "with a single output. 0 = random."
            ),
        )
        timeout: int = Field(
            default=180,
            ge=30,
            le=900,
            description="HTTP timeout in seconds for generation/editing calls.",
        )
        min_request_interval: float = Field(
            default=1.0,
            ge=0.0,
            le=60.0,
            description="Minimum seconds between consecutive TokenHub API calls.",
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
            "Hy-Image and Seedream both support reference images: attach a few and describe the scene.",
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
        """Map the request model back to a configured TokenHub model id."""
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
                    "name": f"{self.valves.name_prefix}❌ Set TokenHub API Key",
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
        """Convert a PIL Image to a base64 data URL under 10 MB."""
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

        for file_item in message.get("files", []) or []:
            if isinstance(file_item, dict):
                _add_from_string(file_item.get("data", ""))

        for image_item in message.get("images", []) or []:
            _add_from_string(image_item)

        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    _add_from_string(item.get("image_url", {}).get("url", ""))
                elif "image" in item:
                    _add_from_string(item["image"])

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
    # Size / aspect detection & resolution
    # -------------------------------------------------------------------------

    def _detect_size_from_prompt(self, prompt: str, editing: bool = False) -> str:
        """Detect a desired pixel size from prompt keywords."""
        prompt_lower = prompt.lower()
        default = self.valves.editing_size if editing else self.valves.default_size

        for size, patterns in {
            "1024x1024": ["1024x1024", "1024*1024"],
            "2048x2048": ["2048x2048", "2048*2048"],
            "1280x720": ["1280x720", "1280*720"],
            "720x1280": ["720x1280", "720*1280"],
            "1024x768": ["1024x768", "1024*768"],
            "768x1024": ["768x1024", "768*1024"],
            "1152x896": ["1152x896", "1152*896"],
            "896x1152": ["896x1152", "896*1152"],
            "1344x704": ["1344x704", "1344*704"],
            "704x1344": ["704x1344", "704*1344"],
            "1536x512": ["1536x512", "1536*512"],
            "512x1536": ["512x1536", "512*1536"],
            "2048x512": ["2048x512", "2048*512"],
            "512x2048": ["512x2048", "512*2048"],
            "1920x1080": ["1920x1080", "1920*1080"],
            "1080x1920": ["1080x1920", "1080*1920"],
            "1536x1024": ["1536x1024", "1536*1024"],
            "1024x1536": ["1024x1536", "1024*1536"],
            "1440x2160": ["1440x2160", "1440*2160"],
            "2160x1440": ["2160x1440", "2160*1440"],
            "2560x1440": ["2560x1440", "2560*1440"],
            "1440x2560": ["1440x2560", "1440*2560"],
            "4096x4096": ["4096x4096", "4096*4096"],
            "2K": ["2k resolution", "2k image", "2k", "resolution 2k"],
            "4K": ["4k resolution", "4k image", "4k", "resolution 4k"],
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
            ): ("1280x720", "720x1280"),
            ("9:16", "9/16", "phone", "mobile", "story", "tiktok", "reels"): (
                "720x1280",
                "1280x720",
            ),
            ("4:3", "4/3", "four by three"): ("1024x768", "768x1024"),
            ("3:4", "3/4", "three by four"): ("768x1024", "1024x768"),
            ("3:2", "3/2", "three by two"): ("1152x896", "896x1152"),
            ("2:3", "2/3", "two by three"): ("896x1152", "1152x896"),
            ("21:9", "21/9", "ultrawide", "ultra wide"): ("1536x512", "512x1536"),
            ("1:1", "square", "instagram post", "profile picture", "avatar"): (
                "1024x1024",
                "1024x1024",
            ),
        }
        for keywords, (landscape_size, portrait_size) in ar_map.items():
            if any(kw in prompt_lower for kw in keywords):
                if any(v in prompt_lower for v in vertical_words):
                    return portrait_size
                return landscape_size

        return default

    def _detect_vidu_aspect(self, prompt: str, editing: bool) -> str:
        """Detect a Vidu aspect_ratio from prompt keywords."""
        prompt_lower = prompt.lower()
        for aspect, keywords in {
            "1:1": ("1:1", "1/1", "square", "instagram post", "profile picture", "avatar"),
            "16:9": ("16:9", "16/9", "widescreen", "landscape", "wide", "banner"),
            "9:16": ("9:16", "9/16", "portrait", "vertical", "phone", "story", "reels"),
            "4:3": ("4:3", "4/3", "four by three"),
            "3:4": ("3:4", "3/4", "three by four"),
            "3:2": ("3:2", "3/2", "three by two"),
            "2:3": ("2:3", "2/3", "two by three"),
            "21:9": ("21:9", "21/9", "ultrawide", "ultra wide"),
        }.items():
            if any(kw in prompt_lower for kw in keywords):
                return aspect
        if editing:
            return "auto"
        return "16:9"

    def _normalize_size(self, size: str) -> str:
        """Normalize '*' separators to 'x' and canonicalize tier case for Tencent."""
        s = str(size or "").strip()
        if not s:
            return s
        if s.upper() in ("1K", "1.5K", "2K", "3K", "4K"):
            return s.upper()
        return s.lower().replace("*", "x")

    def _resolve_size(
        self, model: str, requested: str, editing: bool
    ) -> Optional[str]:
        """Return a size valid for the model family, or None to omit `size`."""
        family = _model_family(model)
        if family in ("vidu", "kling"):
            # These families are driven by aspect_ratio/resolution, not pixels.
            return None

        req = self._normalize_size(requested)
        if req and _size_is_valid(model, req):
            return req

        if family in ("hunyuan", "hunyuan-v3.5"):
            return "1024x1024"
        if family == "seedream":
            return "2K"
        if family == "wand":
            return "1024x1024"
        return req or "1024x1024"

    # -------------------------------------------------------------------------
    # TokenHub API client
    # -------------------------------------------------------------------------

    def _api_url(self, path: str) -> str:
        """Join base_url and path with exactly one 'v1' segment (avoids /v1/v1)."""
        base = (self.valves.base_url or "").strip().rstrip("/")
        p = (path or "").lstrip("/")
        if p.startswith("v1/"):
            p = p[3:]
        if base.endswith("/v1"):
            return f"{base}/{p}" if p else base
        return f"{base}/v1/{p}" if p else f"{base}/v1"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _post_json(
        self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        async def _request() -> httpx.Response:
            timeout = httpx.Timeout(self.valves.timeout, connect=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(
                    url, headers=headers or self._headers(), json=payload
                )

        resp = await self._retry_with_backoff(_request)
        if resp.status_code != 200:
            text = resp.text
            try:
                data = resp.json()
                msg = (
                    data.get("message")
                    or data.get("error", {}).get("message")
                    or data.get("ErrMsg")
                    or text
                )
                code = data.get("code", "N/A")
            except Exception:
                msg = text
                code = "N/A"
            raise Exception(
                f"TokenHub API error – HTTP {resp.status_code} (code: {code}): {msg}"
            )
        return resp.json()

    async def _get_json(self, url: str) -> Dict[str, Any]:
        async def _request() -> httpx.Response:
            timeout = httpx.Timeout(self.valves.timeout, connect=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.get(url, headers=self._headers())

        resp = await self._retry_with_backoff(_request)
        if resp.status_code != 200:
            raise Exception(
                f"TokenHub task polling error – HTTP {resp.status_code}: {resp.text}"
            )
        return resp.json()

    # -------------------------------------------------------------------------
    # Retry / rate-limit helpers
    # -------------------------------------------------------------------------

    async def _rate_limit(self, min_interval: float = 1.0) -> None:
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
            f"TokenHub API request failed after {retries} attempts: {last_exception}"
        )

    # -------------------------------------------------------------------------
    # Async task polling (Vidu / WAND)
    # -------------------------------------------------------------------------

    async def _poll_task(
        self, model: str, task_id: str, emitter=None, poll_interval: float = 3.0
    ) -> Dict[str, Any]:
        """Poll an async TokenHub task until it succeeds, fails, or times out."""
        family = _model_family(model)
        url = self._api_url(_family_task_endpoint(family, task_id))
        max_attempts = 100

        # Vidu: created/queueing/processing/success/failed  -> success
        # WAND:  queued/in_progress/completed/failed/cancelled/incomplete -> completed
        # Kling: top-level or wrapped data[].status  processing/succeeded/failed
        if family == "vidu":
            done_statuses = {"success"}
            fail_statuses = {"failed"}
        elif family == "kling":
            done_statuses = {"succeeded", "success", "succeed"}
            fail_statuses = {"failed", "cancelled", "canceled"}
        else:
            done_statuses = {"completed"}
            fail_statuses = {"failed", "cancelled", "incomplete"}

        for attempt in range(max_attempts):
            data = await self._get_json(url)
            status = str(
                data.get("state")
                or data.get("status")
                or data.get("task_status")
                or "UNKNOWN"
            ).lower()
            # Kling wraps tasks in a data[] array with per-item status
            if status == "unknown":
                items = data.get("data", [])
                if isinstance(items, list) and items:
                    status = str(items[0].get("status", "UNKNOWN")).lower()

            if status in done_statuses:
                return data
            if status in fail_statuses:
                msg = data.get("message") or data.get("ErrMsg") or json.dumps(data, default=str)
                raise Exception(f"TokenHub async task failed (status={status}): {msg}")

            if emitter:
                await emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": (
                                f"⏳ Waiting for image... "
                                f"({attempt + 1}/{max_attempts}, {status})"
                            ),
                            "done": False,
                        },
                    }
                )

            await anyio.sleep(poll_interval)

        raise TimeoutError(
            f"TokenHub async task {task_id} did not complete within the polling window"
        )

    async def _submit_task(self, payload: Dict[str, Any], endpoint: str) -> str:
        """Submit an async generation task and return its task_id."""
        await self._rate_limit(self.valves.min_request_interval)
        data = await self._post_json(self._api_url(endpoint), payload)
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            task_id = data.get("data", {}).get("id")
        if not task_id:
            task_id = data.get("data", {}).get("task_id")
        if not task_id:
            nested = (data.get("result") or data.get("task") or {})
            if isinstance(nested, dict):
                task_id = nested.get("id") or nested.get("task_id")
        if not task_id:
            raise ValueError(
                "No task_id returned from async submission: "
                f"{json.dumps(data, default=str)}"
            )
        return str(task_id)

    # -------------------------------------------------------------------------
    # Payload construction
    # -------------------------------------------------------------------------

    def _clamp_prompt(self, model: str, prompt: str) -> str:
        limit = _max_prompt_chars(model)
        if len(prompt) <= limit:
            return prompt
        print(f"DEBUG: prompt truncated to {limit} chars for {model}")
        return prompt[:limit]

    def _build_payload(
        self,
        model: str,
        prompt: str,
        size: Optional[str],
        base64_images: Optional[List[str]] = None,
        editing: bool = False,
    ) -> Dict[str, Any]:
        """Build a per-family payload for a TokenHub image model."""
        family = _model_family(model)
        prompt = self._clamp_prompt(model, prompt)
        images = base64_images or []

        if family == "vidu":
            aspect = self._detect_vidu_aspect(prompt, editing)
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect,
                "resolution": self.valves.vidu_resolution,
            }
            if images:
                payload["images"] = images[: _max_reference_images(model)]
            if self.valves.seed > 0:
                payload["seed"] = self.valves.seed
            return payload

        if family == "kling":
            aspect = self._detect_vidu_aspect(prompt, editing)
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect,
                "resolution": self.valves.kling_resolution,
            }
            if images:
                payload["images"] = images[: _max_reference_images(model)]
            if self.valves.seed > 0:
                payload["seed"] = self.valves.seed
            return payload

        if family == "hunyuan-v3.5":
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for url in images[: _max_reference_images(model)]:
                content.append({"type": "image_url", "image_url": {"url": url}})
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
            }
            if size:
                s = self._normalize_size(size)
                if s in ("1K", "1.5K", "2K"):
                    payload["generate_max_pixels"] = s
                else:
                    payload["size"] = s
            if self.valves.seed > 0:
                payload["seed"] = self.valves.seed
            footnote = (self.valves.footnote or "").strip()
            if footnote:
                payload["footnote"] = footnote[:16]
            return payload

        if family == "hunyuan":
            payload = {"model": model, "prompt": prompt}
            if images:
                payload["images"] = images[: _max_reference_images(model)]
            if size:
                payload["size"] = self._normalize_size(size)
            if self.valves.seed > 0:
                payload["seed"] = self.valves.seed
            payload["revise"] = bool(self.valves.revise)
            footnote = (self.valves.footnote or "").strip()
            if footnote:
                payload["footnote"] = footnote[:16]
            return payload

        if family == "seedream":
            payload = {"model": model, "prompt": prompt}
            if images:
                payload["images"] = images[: _max_reference_images(model)]
            if size:
                payload["size"] = self._normalize_size(size)
            payload["output_format"] = self.valves.seedream_output_format
            payload["response_format"] = "url"
            if self.valves.watermark:
                payload["watermark"] = True
            mode = self.valves.optimize_prompt_mode
            payload["optimize_prompt_options"] = {"mode": mode}
            lite = "lite" in model
            if lite and self.valves.enable_sequential_generation:
                payload["sequential_image_generation"] = "auto"
                payload["sequential_image_generation_options"] = {
                    "max_images": self.valves.sequential_max_images
                }
            return payload

        if family == "wand":
            payload = {"model": model, "prompt": prompt}
            if images:
                payload["input"] = [
                    {
                        "content": [
                            {"type": "input_image", "image_url": url}
                            for url in images[: _max_reference_images(model)]
                        ]
                    }
                ]
            if size:
                payload["size"] = self._normalize_size(size)
            return payload

        raise ValueError(f"Unsupported model family for {model}")

    async def _call_kling(
        self, payload: Dict[str, Any], model: str, emitter=None
    ) -> List[str]:
        """Kling-Image: return URLs directly if synchronous, else poll the task."""
        endpoint = _family_endpoint("kling")
        await self._rate_limit(self.valves.min_request_interval)
        result = await self._post_json(self._api_url(endpoint), payload)

        urls = self._extract_task_urls(model, result)
        if urls:
            return urls

        data = result.get("data")
        task_id = (
            result.get("task_id")
            or result.get("id")
            or (data.get("id") if isinstance(data, dict) else None)
            or (data.get("task_id") if isinstance(data, dict) else None)
        )
        if not task_id and isinstance(data, list) and data and isinstance(data[0], dict):
            task_id = data[0].get("id") or data[0].get("task_id")
        if not task_id and isinstance(result, dict) and result:
            for key in ("result", "task", "job_id"):
                nested = result.get(key)
                if isinstance(nested, dict):
                    task_id = nested.get("id") or nested.get("task_id")
                    if task_id:
                        break
        if not task_id:
            raise ValueError(
                "Kling returned no image URLs or task id: "
                f"{json.dumps(result, default=str)}"
            )

        polled = await self._poll_task(model, str(task_id), emitter)
        urls = self._extract_task_urls(model, polled)
        if not urls:
            raise ValueError(
                "No image URLs found in Kling task result: "
                f"{json.dumps(polled, default=str)}"
            )
        return urls

    async def _generate_image(
        self, model: str, prompt: str, size: Optional[str], emitter=None
    ) -> List[str]:
        """Generate images from a text prompt. Returns image URLs."""
        payload = self._build_payload(model, prompt, size, editing=False)
        family = _model_family(model)
        endpoint = _family_endpoint(family)
        print(
            f"DEBUG: [GENERATE] model={model}, size={size}, "
            f"endpoint={endpoint}"
        )

        if family == "kling":
            return await self._call_kling(payload, model, emitter)

        if _family_is_async(family):
            task_id = await self._submit_task(payload, endpoint)
            result = await self._poll_task(model, task_id, emitter)
            urls = self._extract_task_urls(model, result)
            if not urls:
                raise ValueError(
                    "No image URLs found in task result: "
                    f"{json.dumps(result, default=str)}"
                )
            return urls

        await self._rate_limit(self.valves.min_request_interval)
        result = await self._post_json(self._api_url(endpoint), payload)
        return self._extract_all_image_urls(result)

    async def _edit_image(
        self,
        model: str,
        prompt: str,
        base64_images: List[str],
        size: Optional[str],
        emitter=None,
    ) -> List[str]:
        """Edit one or more reference images. Returns image URLs."""
        payload = self._build_payload(
            model, prompt, size, base64_images=base64_images, editing=True
        )
        family = _model_family(model)
        endpoint = _family_endpoint(family)
        print(
            f"DEBUG: [EDIT] model={model}, size={size}, "
            f"images={len(base64_images)}, endpoint={endpoint}"
        )

        if family == "kling":
            return await self._call_kling(payload, model, emitter)

        if _family_is_async(family):
            task_id = await self._submit_task(payload, endpoint)
            result = await self._poll_task(model, task_id, emitter)
            urls = self._extract_task_urls(model, result)
            if not urls:
                raise ValueError(
                    "No image URLs found in task result: "
                    f"{json.dumps(result, default=str)}"
                )
            return urls

        await self._rate_limit(self.valves.min_request_interval)
        result = await self._post_json(self._api_url(endpoint), payload)
        return self._extract_all_image_urls(result)

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    def _collect_urls(self, node: Any, urls: List[str]) -> None:
        """Generic scan for image URLs/b64 in a response tree (practical fallback)."""
        if isinstance(node, dict):
            for key in ("url", "image", "image_url"):
                value = node.get(key)
                if isinstance(value, str) and (
                    value.startswith("http") or value.startswith("data:image")
                ):
                    if value not in urls:
                        urls.append(value)
            b64 = node.get("b64_json")
            if isinstance(b64, str) and b64:
                fmt = node.get("output_format", "png")
                data_url = f"data:image/{fmt};base64,{b64}"
                if data_url not in urls:
                    urls.append(data_url)
            for value in node.values():
                self._collect_urls(value, urls)
        elif isinstance(node, list):
            for item in node:
                self._collect_urls(item, urls)

    def _extract_all_image_urls(self, result: dict) -> List[str]:
        """Extract image URLs from sync TokenHub responses."""
        urls: List[str] = []
        items = result.get("data", []) or []
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if isinstance(item, dict):
                if item.get("url"):
                    urls.append(item["url"])
                elif item.get("b64_json"):
                    output_format = item.get("output_format", "png")
                    urls.append(f"data:image/{output_format};base64,{item['b64_json']}")

        # OpenAI-style shapes (Hy-Image 3.5 may wrap results in choices[]).
        for choice in result.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            delta_image = (choice.get("delta") or {}).get("image")
            if isinstance(delta_image, dict) and delta_image.get("url"):
                urls.append(delta_image["url"])
            elif isinstance(delta_image, str) and delta_image:
                urls.append(delta_image)
            for part in (choice.get("message") or {}).get("content", []) or []:
                if isinstance(part, dict) and part.get("image"):
                    urls.append(part["image"])

        if not urls:
            self._collect_urls(result, urls)

        if not urls:
            raise ValueError(
                "No image URLs found in TokenHub response: "
                f"{json.dumps(result, indent=2, default=str)}"
            )
        return urls

    def _extract_task_urls(self, model: str, result: dict) -> List[str]:
        """Extract image URLs from async task result payloads."""
        family = _model_family(model)
        urls: List[str] = []
        if family == "vidu":
            for item in result.get("creations", []) or []:
                if isinstance(item, dict) and item.get("url"):
                    urls.append(item["url"])
        else:
            items = result.get("data", []) or []
            if isinstance(items, dict):
                items = [items]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("url"):
                    urls.append(item["url"])
                elif item.get("b64_json"):
                    output_format = item.get("output_format", "png")
                    urls.append(
                        f"data:image/{output_format};base64,{item['b64_json']}"
                    )
        if not urls:
            self._collect_urls(result, urls)
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

        if _model_family(model) in ("vidu", "kling"):
            shown_size = f"{self._detect_vidu_aspect('', editing)}"
            meta = (
                f"**Aspect:** `{shown_size}` | **Model:** `{model}` | "
                f"**Outputs:** `{len(urls)}`"
            )
        else:
            shown_size = size or "auto"
            meta = (
                f"**Size:** `{shown_size}` | **Model:** `{model}` | "
                f"**Outputs:** `{len(urls)}`"
            )
        tip = random.choice(self.tooltips)
        return f"{header}\n" + "\n\n".join(blocks) + f"\n\n{meta}\n\n💡 _Tip: {tip}_"

    def _error(self, message: str) -> str:
        return f"❌ **Error:**\n\n{message}"

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
                "TokenHub API key is missing. Set it in the function's Valves."
            )
            return

        model = self._resolve_model(body.get("model", "") or "")
        if not model:
            yield self._error("Could not determine the selected TokenHub model.")
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

            images = self._extract_images_from_message(last_message)
            if not images:
                for msg in reversed(messages[:-1]):
                    found = self._extract_images_from_message(msg)
                    if found:
                        images = found
                        break

            is_editing = bool(images)
            family = _model_family(model)

            if family == "unknown":
                yield self._error(
                    f"Model `{model}` does not match any known TokenHub image "
                    "family (Hunyuan, Seedream, Vidu, Kling, WAND-Vega)."
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
                if family in ("vidu", "kling"):
                    aspect = self._detect_vidu_aspect(user_prompt, is_editing)
                    resolution = (
                        self.valves.vidu_resolution
                        if family == "vidu"
                        else self.valves.kling_resolution
                    )
                    status = (
                        f"✏️ Editing image ({aspect}, {resolution}) using {model}…"
                        if is_editing
                        else f"🎨 Generating image ({aspect}, {resolution}) "
                        f"using {model}…"
                    )
                else:
                    shown_size = size or "auto"
                    status = (
                        f"✏️ Editing image at {shown_size} using {model}…"
                        if is_editing
                        else f"🎨 Generating image at {shown_size} using {model}…"
                    )
                await __event_emitter__(
                    {"type": "status", "data": {"description": status, "done": False}}
                )

            if is_editing:
                base64_urls = [img["base64_url"] for img in images]
                urls = await self._edit_image(
                    model, user_prompt, base64_urls, size, __event_emitter__
                )
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