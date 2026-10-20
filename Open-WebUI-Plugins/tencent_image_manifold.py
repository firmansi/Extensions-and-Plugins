"""
title: Tencent TokenHub Image Generation & Editing (Manifold)
author: dfirmans (upgraded by AI)
version: 1.2.0
description: >
  Manifold Pipe for Tencent Cloud LLM Service TokenHub image models
  (https://tokenhub-intl.tencentcloudmaas.com/v1). Every model listed in the
  `models` valve is registered as its own selectable model in Open WebUI, so
  one function can serve the whole family: Tencent Hunyuan Hy-Image
  (hy-image-v3 / hy-image-v3.5-preview), WAND-Vega-Image (lite / flash / pro),
  plus third-party models hosted through the TokenHub AI gateway such as
  Seedream-Image v5.0 (pro / lite) and Vidu-Image-q2.
  Text-to-image and image-to-image (editing) are routed automatically, and a
  follow-up instruction with no image attached reuses the most recent picture
  from the conversation, so "change the colour to blue" edits the image that
  was just produced instead of starting a new one. Per-family endpoint, size
  and sampling handling is built on the official docs for docs.product/1300
  (Hy Image, WAND, Vidu, Seedream). The requested output geometry is parsed
  from the prompt once and then translated into whatever the target family
  accepts, so asking for 1:1 really does produce a 1:1 image instead of a
  silently substituted default.
  Compatible with Open WebUI 0.11.4+.
required_open_webui_version: 0.11.4
requirements: pillow, httpx, anyio
"""

import asyncio
import base64
import io
import json
import math
import random
import re
import time
import uuid
from dataclasses import dataclass
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

# ---------------------------------------------------------------------------
# Aspect-ratio engine
# ---------------------------------------------------------------------------
# Every request is first reduced to a canonical aspect ratio (the keys below),
# then translated into whatever the target family actually accepts: a pixel
# size (Hunyuan / Seedream / WAND), an aspect_ratio string (Vidu), or a
# bare resolution tier when the ratio was never specified and the model should
# be left to decide on its own.

# Canonical ratios, widest to tallest. The `w:h` spelling is exactly what the
# `aspect_ratio` field of the Vidu-Image family expects.
ASPECT_RATIOS: Dict[str, float] = {
    "21:9": 21 / 9,
    "2:1": 2 / 1,
    "16:9": 16 / 9,
    "3:2": 3 / 2,
    "4:3": 4 / 3,
    "5:4": 5 / 4,
    "1:1": 1 / 1,
    "4:5": 4 / 5,
    "3:4": 3 / 4,
    "2:3": 2 / 3,
    "9:16": 9 / 16,
    "1:2": 1 / 2,
    "9:21": 9 / 21,
}

# `aspect_ratio` enum of the Vidu-Image family. Ratios outside this list (4:5,
# 5:4, 1:2) are snapped into it rather than sent and rejected.
HOSTED_ASPECTS: Tuple[str, ...] = (
    "21:9",
    "16:9",
    "3:2",
    "4:3",
    "1:1",
    "3:4",
    "2:3",
    "9:16",
)

# Pixel budget per resolution tier, used to derive a concrete WxH for the
# families that take pixels instead of an aspect ratio.
TIER_PIXEL_BUDGET: Dict[str, int] = {
    "1K": 1048576,
    "1.5K": 2359296,
    "2K": 4194304,
    "3K": 9437184,
    "4K": 16777216,
}

# Hy-Image-3.0's 37 documented preset sizes (docs.product/1300/83708). Every one
# of them satisfies w,h in [512, 2048] with area <= 1024x1024, so any requested
# ratio can be snapped onto the closest real preset.
HUNYUAN_V3_PRESET_SIZES: Tuple[Tuple[int, int], ...] = (
    (2048, 512), (1984, 512), (1920, 512), (1856, 512), (1792, 512), (1728, 512),
    (1664, 512), (1600, 512), (1536, 512), (1472, 576), (1408, 640), (1344, 704),
    (1280, 768), (1216, 832), (1152, 896), (1088, 960), (1024, 1024), (960, 1088),
    (896, 1152), (832, 1216), (768, 1280), (704, 1344), (640, 1408), (576, 1472),
    (512, 1536), (512, 1600), (512, 1664), (512, 1728), (512, 1792), (512, 1856),
    (512, 1920), (512, 1984), (512, 2048), (768, 1024), (720, 1280), (1024, 768),
    (1280, 720),
)

# WAND-Vega-Image preset sizes, as (width, height) pairs. The API only accepts
# these exact strings, so any requested ratio has to snap onto one of them.
WAND_PRESET_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    sorted(
        (
            (1024, 1024), (2048, 2048), (4096, 4096), (1024, 1536), (1440, 2160),
            (3328, 4992), (1536, 1024), (2160, 1440), (4992, 3328), (1080, 1440),
            (1440, 1920), (2160, 2880), (3584, 4800), (1440, 1080), (1920, 1440),
            (2880, 2160), (4800, 3584), (1080, 1920), (1920, 1080), (1024, 2187),
            (2187, 1024),
        ),
        key=lambda wh: (wh[0] * wh[1], wh[0]),
    )
)

# `WxH` / `W:H` / `W*H` tokens. Both sides >= 64 means pixels, anything smaller
# means an aspect ratio, which is what lets "1024x1024" and "1:1" both work.
_DIMENSIONS_RE = re.compile(
    r"(?<![\w.])(\d{1,4})\s*([:x×*])\s*(\d{1,4})(?![\w.])", re.IGNORECASE
)
_DIMENSIONS_BY_RE = re.compile(
    r"(?<![\w.])(\d{1,2})\s+by\s+(\d{1,2})(?![\w.])", re.IGNORECASE
)

# Markdown image links, used to recover pictures from earlier turns. This has to
# match plain `http(s)` targets, not just inline base64: images this function
# generated are rendered as remote links, and those are exactly the references a
# follow-up like "change the colour" needs to find.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?\s*\)")

# Word-based aspect hints. Numeric ratios are handled by _DIMENSIONS_RE, so
# these entries only need the vocabulary. English + Indonesian, since the
# plugin is driven from Indonesian prompts as well.
_ASPECT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "21:9": (
        r"\bultra[\s-]?wide\b",
        r"\bultra\s+lebar\b",
        r"\bsinemascope\b",
    ),
    "16:9": (
        r"\bwide[\s-]?screen\b",
        r"\blayar\s+lebar\b",
        r"\bthumbnail\b",
        r"\byoutube\b",
        r"\bdesktop\s+wallpaper\b",
        r"\b16\s*:\s*9\b",
    ),
    "3:2": (r"\b3\s*:\s*2\b",),
    "4:3": (r"\b4\s*:\s*3\b",),
    "5:4": (r"\b5\s*:\s*4\b",),
    "1:1": (
        r"\b1\s*:\s*1\b",
        r"\bsquare\b",
        r"\bpersegi\b",
        r"\bavatar\b",
        r"\bprofile\s+picture\b",
        r"\bfoto\s+profil\b",
    ),
    # 4:5 is what a portrait social post usually means, so it is inferred.
    "4:5": (
        r"\b4\s*:\s*5\b",
        r"\binstagram\s+post\b",
        r"\bfeed\s+post\b",
        r"\bportrait\s+post\b",
    ),
    "3:4": (r"\b3\s*:\s*4\b",),
    "2:3": (r"\b2\s*:\s*3\b",),
    "9:16": (
        r"\b9\s*:\s*16\b",
        r"\bstory\b",
        r"\breels\b",
        r"\bshorts\b",
        r"\btiktok\b",
        r"\bvertikal\b",
        r"\binstagram\s+story\b",
    ),
    "1:2": (r"\b1\s*:\s*2\b",),
}

# Orientation words are weaker evidence than any ratio keyword: they are only
# consulted when nothing else matched, and a portrait cue beats a wide cue.
_ORIENTATION_PATTERNS: Tuple[Tuple[str, str, int], ...] = (
    ("9:16", r"\b(?:vertical|vertikal|portrait)\b", 0),
    ("16:9", r"\b(?:horizontal|landscape|lanskap)\b", 1),
)


def _nearest_aspect(
    value: float, candidates: Optional[Tuple[str, ...]] = None
) -> str:
    """Map any width/height float onto the closest canonical ratio."""
    if value <= 0:
        return "1:1"
    names = candidates if candidates is not None else tuple(ASPECT_RATIOS)
    best, best_score = names[0], float("inf")
    for name in names:
        score = abs(math.log(ASPECT_RATIOS[name] / value))
        if score < best_score:
            best, best_score = name, score
    return best


def _snap_to_multiple(value: float, multiple: int = 8, minimum: int = 8) -> int:
    """Round to the nearest multiple, half-up, so ratios stay accurate."""
    return max(minimum, int(math.floor(value / multiple + 0.5) * multiple))


def _pixels_for_ratio(ratio: float, target_area: int) -> Tuple[int, int]:
    """Build a WxH whose width/height is `ratio` and whose area is `target_area`.

    Orientation is preserved: a portrait ratio such as 9:16 yields a WxH that is
    taller than it is wide.
    """
    w = _snap_to_multiple(math.sqrt(target_area * ratio))
    h = _snap_to_multiple(math.sqrt(target_area / ratio))
    return w, h


def _fit_area(
    w: int,
    h: int,
    low: int,
    high: int,
    min_side: int = 8,
) -> Tuple[int, int]:
    """Nudge the longer side until the pixel count lands inside [low, high]."""
    for _ in range(64):
        area = w * h
        if low <= area <= high:
            break
        if area > high:
            step = -8
        else:
            step = 8
        if w >= h:
            w = max(min_side, w + step)
        else:
            h = max(min_side, h + step)
    return w, h


def _snap_to_preset(
    ratio: float, target_area: int, presets: Tuple[Tuple[int, int], ...]
) -> Tuple[int, int]:
    """Choose the preset whose ratio is closest, breaking ties on pixel count."""
    best, best_score = presets[0], None
    for w, h in presets:
        score = (abs(math.log((w / h) / ratio)), abs(w * h - target_area))
        if best_score is None or score < best_score:
            best, best_score = (w, h), score
    return best


@dataclass
class OutputSpec:
    """Resolved output geometry for a single request.

    `kind` is "pixels" (families taking WxH), "aspect" (families taking an
    aspect_ratio string) or "tier" (no ratio was requested, so the model is
    left to pick the shape itself).
    """

    kind: str
    value: str
    aspect: Optional[str] = None
    source: str = "default"

    @property
    def label(self) -> str:
        return self.value

    @property
    def status_value(self) -> str:
        if self.aspect:
            return f"{self.value} ({self.aspect})"
        return self.value


def _classify_dimensions(first: int, second: int) -> Tuple[str, Any]:
    """Classify a `a:b` token as a pixel size or as an aspect ratio."""
    if first <= 0 or second <= 0:
        return ("none", None)
    if first >= 64 and second >= 64:
        return ("pixels", (first, second))
    return ("aspect", _nearest_aspect(first / second))


def detect_dimensions(prompt: str) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Scan a prompt for an explicit size, in precedence order.

    Returns `(aspect, pixels)`. Exactly one of the two is normally set; both
    are None when the prompt states no geometry at all.

    The tiers, strongest first:
      1. any numeric `a:b` / `axb` / `a by b` token (earliest wins),
      2. ratio vocabulary such as "square" or "story" (earliest wins),
      3. bare orientation words such as "portrait".
    """
    text = (prompt or "").lower()

    # 1. Numeric tokens. This is what makes "1024x1024" and "1:1" agree with
    #    each other instead of feeding two different detectors.
    matches: List[Tuple[int, str, Any]] = []
    for match in _DIMENSIONS_RE.finditer(text):
        kind, value = _classify_dimensions(int(match.group(1)), int(match.group(3)))
        if kind != "none":
            matches.append((match.start(), kind, value))
    for match in _DIMENSIONS_BY_RE.finditer(text):
        kind, value = _classify_dimensions(int(match.group(1)), int(match.group(2)))
        if kind != "none":
            matches.append((match.start(), kind, value))
    if matches:
        matches.sort(key=lambda item: item[0])
        kind, value = matches[0][1], matches[0][2]
        return (value, None) if kind == "aspect" else (None, value)

    # 2. Ratio vocabulary.
    keyword_hits: List[Tuple[int, str]] = []
    for aspect, patterns in _ASPECT_KEYWORDS.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                keyword_hits.append((match.start(), aspect))
                break
    if keyword_hits:
        keyword_hits.sort(key=lambda item: item[0])
        return keyword_hits[0][1], None

    # 3. Orientation, with portrait winning a positional tie.
    orientation_hits: List[Tuple[int, int, str]] = []
    for aspect, pattern, priority in _ORIENTATION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            orientation_hits.append((match.start(), priority, aspect))
    if orientation_hits:
        orientation_hits.sort(key=lambda item: (item[0], item[1]))
        return orientation_hits[0][2], None

    return None, None


def detect_resolution_tier(prompt: str) -> Optional[str]:
    """Find a resolution tier such as "4K" or "2k resolution" in a prompt."""
    text = (prompt or "").lower()
    match = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*k\b(?![\w])", text)
    if not match:
        return None
    value = float(match.group(1))
    known = sorted(TIER_PIXEL_BUDGET)
    for tier in known:
        if abs(float(tier.rstrip("kK")) - value) < 0.01:
            return tier
    return None

GENERATION_AND_EDITING_FAMILIES = {
    "hunyuan",
    "hunyuan-v3.5",
    "seedream",
    "vidu",
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
    if family == "wand":
        return "/v1/wand/vega-images/generations"
    raise ValueError(f"No known TokenHub endpoint for model family '{family}'")


def _family_is_async(family: str) -> bool:
    return family in ("vidu", "wand")


def _family_task_endpoint(family: str, task_id: str) -> str:
    if family == "vidu":
        return f"/v1/wand/vidu-image/tasks/{task_id}"
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
    if family == "vidu":
        return True  # this family uses aspect ratio instead of pixels
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
                "Vidu (hosted): vidu-image-q2"
            ),
        )
        name_prefix: str = Field(
            default="Tencent/",
            description="Prefix added to each model name shown in Open WebUI.",
        )
        default_size: IMAGE_SIZES = Field(
            default="1024x1024",
            description=(
                "Fallback output geometry for generation when the prompt names "
                "no size. A WxH value sets the pixel budget used to derive the "
                "size for the requested ratio; a tier such as 2K / 4K sets the "
                "resolution. Values outside a family's documented window are "
                "clamped to it, and ignored entirely by Vidu, which takes an "
                "aspect ratio and the resolution valve below."
            ),
        )
        editing_size: IMAGE_SIZES = Field(
            default="1024x1024",
            description=(
                "Same as default_size but for editing. Used when the prompt "
                "names no size and no reference image is attached."
            ),
        )
        vidu_resolution: VIDU_RESOLUTIONS = Field(
            default="1080p",
            description="Vidu-Image-q2 output resolution tier (1080p / 2K / 4K).",
        )
        default_aspect_ratio: VIDU_ASPECTS = Field(
            default="16:9",
            description=(
                "Aspect ratio for Vidu when the prompt states no geometry. "
                "'auto' follows the aspect ratio of the first attached image."
            ),
        )
        enable_size_detection: bool = Field(
            default=True,
            description=(
                "Read the output geometry out of the prompt. Recognises WxH, "
                "W:H and 'W by H' tokens (English and Indonesian), ratio words "
                "such as 'square' / 'story' / 'ultrawide', resolution tiers, "
                "and bare 'portrait' / 'landscape' as a last resort. When off, "
                "only the fallback valves above are used."
            ),
        )
        reuse_previous_images: bool = Field(
            default=True,
            description=(
                "When the current message has no image attached, reuse the most "
                "recent image from earlier in the conversation as the reference. "
                "This is what makes a follow-up like 'change the colour to blue' "
                "edit the picture that was just produced instead of starting a "
                "brand new one. Turn it off to always generate from scratch."
            ),
        )
        history_scan_depth: int = Field(
            default=3,
            ge=1,
            le=20,
            description=(
                "How many previous messages to search backwards for a reference "
                "image when the current message has none. 1 looks only at the "
                "immediately preceding turn."
            ),
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
            "No need to re-upload: just say 'make the background darker' and the last image is reused.",
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

    def _collect_image_refs(self, message: Dict[str, Any]) -> List[str]:
        """Gather raw image references from a message, in priority order.

        Both inline `data:` URLs and remote `http(s)` URLs are collected, so a
        message that merely links to a picture still counts as a reference.
        """
        refs: List[str] = []

        def _add(value: Any) -> None:
            if not isinstance(value, str):
                return
            value = value.strip()
            if not value:
                return
            if value.startswith(("data:image", "http://", "https://")):
                if value not in refs:
                    refs.append(value)

        for file_item in message.get("files", []) or []:
            if isinstance(file_item, dict):
                _add(file_item.get("data", ""))
                _add(file_item.get("url", ""))
            else:
                _add(file_item)

        for image_item in message.get("images", []) or []:
            if isinstance(image_item, dict):
                _add(image_item.get("url", ""))
                _add(image_item.get("image_url", ""))
                if isinstance(image_item.get("image_url"), dict):
                    _add(image_item["image_url"].get("url", ""))
            else:
                _add(image_item)

        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    _add(image_url.get("url", "") if isinstance(image_url, dict) else image_url)
                elif "image" in item:
                    _add(item["image"])
        elif isinstance(content, str):
            for url in _MARKDOWN_IMAGE_RE.findall(content):
                _add(url.strip().strip("<>"))

        return refs

    def _decode_data_url(self, data_url: str) -> Optional[Image.Image]:
        try:
            b64_data = data_url.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data, validate=True)
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.load()
            return pil_img
        except Exception as e:
            print(f"DEBUG: could not decode inline image: {e}")
            return None

    async def _fetch_remote_image(self, url: str) -> Optional[Image.Image]:
        """Download a previously generated image so it can be re-encoded."""
        try:
            timeout = httpx.Timeout(30, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, follow_redirects=True)
            if resp.status_code != 200:
                print(f"DEBUG: reference image {url} -> HTTP {resp.status_code}")
                return None
            if len(resp.content) > self._MAX_B64_BYTES * 3:
                print(f"DEBUG: reference image {url} is too large to inline")
                return None
            pil_img = Image.open(io.BytesIO(resp.content))
            pil_img.load()
            return pil_img
        except Exception as e:
            print(f"DEBUG: could not download reference image: {e}")
            return None

    async def _load_reference(
        self, ref: str, source: str
    ) -> Optional[Dict[str, Any]]:
        """Normalise one reference into {ref, pil, source} for the payload."""
        if ref.startswith("data:image"):
            pil_img = self._decode_data_url(ref)
            if pil_img is None:
                return None
            return {
                "ref": self._image_to_base64(pil_img),
                "pil": pil_img,
                "source": source,
            }

        # A remote URL: normally a picture this function generated on an
        # earlier turn. Inline it when possible so the model definitely gets
        # the pixels, and otherwise hand the URL over untouched. A failed
        # download must never abort the request, so it degrades to the URL.
        try:
            pil_img = await self._fetch_remote_image(ref)
        except Exception as e:
            print(f"DEBUG: reference fetch failed for {ref}: {e}")
            pil_img = None

        if pil_img is not None:
            try:
                return {
                    "ref": self._image_to_base64(pil_img),
                    "pil": pil_img,
                    "source": source,
                }
            except Exception as e:
                print(f"DEBUG: could not re-encode reference image: {e}")
        return {"ref": ref, "pil": None, "source": source}

    async def _load_references(
        self, refs: List[str], source: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Resolve raw references into deduped, payload-ready images.

        Returns the usable images plus the refs that could not be read, so the
        caller can tell "nothing here" apart from "something is here but broken".
        """
        images: List[Dict[str, Any]] = []
        unreadable: List[str] = []
        seen: set = set()
        for ref in refs:
            image = await self._load_reference(ref, source)
            if image is None:
                unreadable.append(ref)
                continue
            key = image["ref"][:256]
            if key in seen:
                continue
            seen.add(key)
            images.append(image)
        return images, unreadable

    async def _images_from_message(
        self, message: Dict[str, Any], source: str = "attached"
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Resolve the images carried by a single message."""
        return await self._load_references(
            self._collect_image_refs(message), source
        )

    async def _resolve_reference_images(
        self, messages: List[Dict[str, Any]], last_message: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Pick the images this turn should treat as references.

        An image attached to the current message always wins. Only when the
        turn carries none do we look back through recent history, so a
        follow-up instruction continues the previous picture. A reference that
        is present but unreadable raises instead of silently falling back to
        something else, because quietly editing a different image is worse
        than saying so.
        """
        attached, attached_broken = await self._images_from_message(
            last_message, "attached"
        )
        if attached:
            return attached, "attached"
        if attached_broken:
            # The user did attach something. Editing an older image instead
            # would be silently wrong, so stop and say so.
            raise ValueError(
                "The image attached to this message could not be read "
                f"({len(attached_broken)} attachment(s) failed to decode). "
                "Please re-upload it."
            )

        if not self.valves.reuse_previous_images:
            return [], None

        depth = max(1, int(self.valves.history_scan_depth))
        unreadable: List[str] = []
        for msg in reversed(messages[:-1][-depth:]):
            refs = self._collect_image_refs(msg)
            if not refs:
                continue
            found, broken = await self._load_references(refs, "previous")
            if found:
                origin = (
                    "previous output"
                    if msg.get("role") == "assistant"
                    else "earlier upload"
                )
                return found, origin
            unreadable.extend(refs)

        if unreadable:
            raise ValueError(
                "Found an image in this conversation but could not read it "
                f"({len(unreadable)} reference(s) failed to load). TokenHub "
                "result links expire after about 12 hours, so please re-upload "
                "the image you want to edit."
            )
        return [], None

    # -------------------------------------------------------------------------
    # Size / aspect detection & resolution
    # -------------------------------------------------------------------------

    def _reference_aspect(
        self, images: Optional[List[Dict[str, Any]]]
    ) -> Optional[str]:
        """Canonical ratio of the first usable reference image."""
        for image in images or []:
            pil_image = image.get("pil")
            width, height = getattr(pil_image, "size", (0, 0))
            if width and height:
                return _nearest_aspect(width / height)
        return None

    def _default_size_valve(self, editing: bool) -> str:
        return (
            self.valves.editing_size if editing else self.valves.default_size
        )

    def _target_area(self, prompt: str, editing: bool) -> int:
        """Pixel budget for this request: prompt tier first, then the valve.

        This is a target, not a guarantee. It gets clamped into each family's
        documented area window before a WxH is derived from it.
        """
        tier = detect_resolution_tier(prompt)
        if tier is not None:
            return TIER_PIXEL_BUDGET[tier]
        valve = self._default_size_valve(editing)
        if valve.upper() in TIER_PIXEL_BUDGET:
            return TIER_PIXEL_BUDGET[valve.upper()]
        parsed = _parse_size(valve)
        return parsed[0] * parsed[1] if parsed else TIER_PIXEL_BUDGET["2K"]

    def _resolve_output(
        self,
        model: str,
        prompt: str,
        editing: bool = False,
        reference: Optional[str] = None,
    ) -> OutputSpec:
        """Turn a prompt into the geometry the target family actually accepts.

        `reference` is the canonical ratio of the first attached image, used as
        the fallback when editing and the prompt says nothing about geometry.
        """
        family = _model_family(model)
        aspect: Optional[str] = None
        pixels: Optional[Tuple[int, int]] = None

        if self.valves.enable_size_detection:
            aspect, pixels = detect_dimensions(prompt)
            if aspect:
                source = "prompt"
            elif pixels:
                # An explicit WxH still names a ratio, which matters for Vidu,
                # the one family that only accepts `aspect_ratio`.
                aspect = _nearest_aspect(pixels[0] / pixels[1])
                source = "explicit"
            elif reference:
                aspect, source = reference, "reference"
            else:
                source = "default"
        elif reference:
            aspect, source = reference, "reference"
        else:
            source = "disabled"

        area = self._target_area(prompt, editing)

        # Vidu takes a ratio string, never a pixel size.
        if family == "vidu":
            if aspect is None:
                resolved = self.valves.default_aspect_ratio
                return OutputSpec("aspect", resolved, resolved, "default")
            if aspect == "auto":
                return OutputSpec("aspect", "auto", None, source)
            if aspect not in HOSTED_ASPECTS:
                # 4:5 / 5:4 / 1:2 are outside the Vidu enum, so snap them in
                # rather than letting the API reject the request.
                snapped = _nearest_aspect(ASPECT_RATIOS[aspect], HOSTED_ASPECTS)
                return OutputSpec("aspect", snapped, aspect, "snapped")
            return OutputSpec("aspect", aspect, aspect, source)

        # A pixel size the user typed that the family accepts is used verbatim.
        if pixels is not None:
            explicit = f"{pixels[0]}x{pixels[1]}"
            if _size_is_valid(model, explicit):
                return OutputSpec(
                    "pixels",
                    explicit,
                    _nearest_aspect(pixels[0] / pixels[1]),
                    "explicit",
                )

        # No ratio was stated: hand over a resolution tier and let the model
        # choose the shape. This is the documented Seedream / Hy-3.5 behaviour
        # and avoids forcing a square onto a prompt that never asked for one.
        if aspect is None:
            tier = self._resolve_tier(model, prompt, editing)
            if family == "wand":
                # WAND only ever accepts concrete preset sizes, never a tier.
                w, h = _snap_to_preset(1.0, TIER_PIXEL_BUDGET[tier], WAND_PRESET_PAIRS)
                return OutputSpec("pixels", f"{w}x{h}", None, source)
            return OutputSpec("tier", tier, None, source)

        # Otherwise derive a valid WxH that honours the requested ratio.
        if family == "hunyuan":
            w, h = _snap_to_preset(ASPECT_RATIOS[aspect], area, HUNYUAN_V3_PRESET_SIZES)
            return OutputSpec("pixels", f"{w}x{h}", aspect, source)

        if family == "hunyuan-v3.5":
            # Sides 256..8192, area capped at 4096x4096.
            w, h = self._derive_pixels(
                ASPECT_RATIOS[aspect], area, 256 * 256, 4096 * 4096, 256
            )
            return OutputSpec("pixels", f"{w}x{h}", aspect, source)

        if family == "seedream":
            # Each Seedream variant has its own total-pixel window; the requested
            # area is clamped into it before a WxH is derived.
            low, high = SEEDREAM_PIXEL_RANGE.get(model, (921600, 16777216))
            w, h = self._derive_pixels(ASPECT_RATIOS[aspect], area, low, high, 64)
            return OutputSpec("pixels", f"{w}x{h}", aspect, source)

        if family == "wand":
            w, h = _snap_to_preset(ASPECT_RATIOS[aspect], area, WAND_PRESET_PAIRS)
            return OutputSpec("pixels", f"{w}x{h}", aspect, source)

        return OutputSpec("tier", self._default_tier(model, editing), aspect, source)

    @staticmethod
    def _derive_pixels(
        ratio: float, area: int, low: int, high: int, min_side: int
    ) -> Tuple[int, int]:
        """Derive a WxH for `ratio` that lands inside the family's area window.

        Clamping the target area *before* deriving matters: Seedream-lite has a
        3.6 MP floor, so a 1:1 request asked for at 1 MP has to be scaled up or
        the API rejects the request outright.
        """
        target = min(max(area, low), high)
        w, h = _pixels_for_ratio(ratio, target)
        return _fit_area(w, h, low, high, min_side)

    def _default_tier(self, model: str, editing: bool) -> str:
        """Resolution tier to fall back on when the prompt names no geometry."""
        family = _model_family(model)
        valve = self._default_size_valve(editing).upper()
        allowed = self._allowed_tiers(model)
        if valve in TIER_PIXEL_BUDGET:
            return self._clamp_tier(valve, allowed)
        if allowed:
            # 2K is the documented default for both Seedream variants.
            return "2K" if "2K" in allowed else sorted(allowed)[0]
        return "1.5K" if family == "hunyuan-v3.5" else "2K"

    def _resolve_tier(self, model: str, prompt: str, editing: bool) -> str:
        """Resolution tier for this request: prompt tier, then valve, then default."""
        requested = detect_resolution_tier(prompt)
        if requested is not None:
            return self._clamp_tier(requested, self._allowed_tiers(model))
        return self._default_tier(model, editing)

    @staticmethod
    def _allowed_tiers(model: str) -> Optional[set]:
        """Tiers a family actually accepts, or None when it takes any tier."""
        if _model_family(model) != "seedream":
            return None
        return SEEDREAM_LITE_TIERS if "lite" in model else SEEDREAM_PRO_TIERS

    @staticmethod
    def _clamp_tier(tier: str, allowed: Optional[set]) -> str:
        """Pull a tier into the supported set by pixel count, e.g. 4K -> 2K for pro."""
        if not allowed or tier in allowed:
            return tier
        target = TIER_PIXEL_BUDGET.get(tier)
        if target is None:
            return sorted(allowed)[0]
        return min(allowed, key=lambda t: abs(TIER_PIXEL_BUDGET[t] - target))

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
        url = self._api_url(
            _family_task_endpoint(_model_family(model), task_id)
        )
        max_attempts = 100

        # Vidu: created/queueing/processing/success/failed  -> success
        # WAND:  queued/in_progress/completed/failed/cancelled/incomplete -> completed
        # The success spelling drifts between the upstream vendors, so every
        # family accepts each of them rather than timing out on a live task.
        done_statuses = {"success", "succeeded", "succeed", "completed"}
        fail_statuses = {"failed", "cancelled", "canceled", "incomplete"}

        for attempt in range(max_attempts):
            data = await self._get_json(url)
            status = str(
                data.get("state")
                or data.get("status")
                or data.get("task_status")
                or "UNKNOWN"
            ).lower()
            # Some families wrap the task in a data[] array with a per-item status
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
        spec: OutputSpec,
        reference_images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a per-family payload for a TokenHub image model."""
        family = _model_family(model)
        prompt = self._clamp_prompt(model, prompt)
        # Each entry is a {"ref": data-or-http URL, "pil": ...} mapping.
        images = [image["ref"] for image in (reference_images or [])]

        if family == "vidu":
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": spec.value,
                "resolution": self.valves.vidu_resolution,
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
            if spec.kind == "pixels":
                payload["size"] = spec.value
            else:
                # generate_max_pixels is an integer pixel budget, not a tier
                # string, and it only applies while `size` is absent.
                payload["generate_max_pixels"] = TIER_PIXEL_BUDGET.get(
                    spec.value, 2359296
                )
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
            if spec.kind == "pixels":
                payload["size"] = spec.value
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
            if spec.kind in ("pixels", "tier"):
                payload["size"] = spec.value
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
            if spec.kind == "pixels":
                payload["size"] = spec.value
            return payload

        raise ValueError(f"Unsupported model family for {model}")

    async def _run_image_request(
        self,
        model: str,
        prompt: str,
        spec: OutputSpec,
        reference_images: Optional[List[Dict[str, Any]]] = None,
        emitter=None,
    ) -> List[str]:
        """Shared generate/edit path: build the payload, submit, resolve URLs."""
        payload = self._build_payload(model, prompt, spec, reference_images)
        family = _model_family(model)
        endpoint = _family_endpoint(family)
        print(
            f"DEBUG: [{'EDIT' if reference_images else 'GENERATE'}] "
            f"model={model}, spec={spec.kind}={spec.value} "
            f"(aspect={spec.aspect}, source={spec.source}), "
            f"images={len(reference_images or [])}, endpoint={endpoint}"
        )

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

    async def _generate_image(
        self, model: str, prompt: str, spec: OutputSpec, emitter=None
    ) -> List[str]:
        """Generate images from a text prompt. Returns image URLs."""
        return await self._run_image_request(model, prompt, spec, None, emitter)

    async def _edit_image(
        self,
        model: str,
        prompt: str,
        reference_images: List[Dict[str, Any]],
        spec: OutputSpec,
        emitter=None,
    ) -> List[str]:
        """Edit one or more reference images. Returns image URLs."""
        return await self._run_image_request(
            model, prompt, spec, reference_images, emitter
        )

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

    def _render(
        self,
        model: str,
        urls: List[str],
        spec: OutputSpec,
        editing: bool,
        reference_origin: Optional[str] = None,
    ) -> str:
        label = "edited" if editing else "generated"
        header = f"**✏️ Here is your {label} image:**\n" if editing else (
            "**🖼️ Here is your generated image:**\n"
        )
        if editing and reference_origin:
            # Makes it obvious that this turn continued an earlier picture
            # rather than starting over, which is otherwise easy to misread.
            header += f"*(edited from your {reference_origin})*\n"

        blocks: List[str] = []
        for idx, url in enumerate(urls, start=1):
            if len(urls) > 1:
                caption = f"Variation {idx}"
            else:
                caption = "Edited Image" if editing else "Generated Image"
            blocks.append(f"![{caption}]({url})")

        if spec.kind == "aspect":
            meta = (
                f"**Aspect:** `{spec.value}` | **Resolution:** "
                f"`{self.valves.vidu_resolution}` | **Model:** `{model}` | "
                f"**Outputs:** `{len(urls)}`"
            )
        elif spec.kind == "tier":
            meta = (
                f"**Size:** `{spec.value}` (model picks the ratio) | "
                f"**Model:** `{model}` | **Outputs:** `{len(urls)}`"
            )
        else:
            ratio = f" · **Aspect:** `{spec.aspect}`" if spec.aspect else ""
            meta = (
                f"**Size:** `{spec.value}`{ratio} | **Model:** `{model}` | "
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

            # An image on this message wins; otherwise continue the most recent
            # picture from the conversation so "change the colour to blue" edits
            # the previous result instead of generating a brand new image.
            images, reference_origin = await self._resolve_reference_images(
                messages, last_message
            )

            is_editing = bool(images)
            family = _model_family(model)

            if family == "unknown":
                yield self._error(
                    f"Model `{model}` does not match any known TokenHub image "
                    "family (Hunyuan, Seedream, Vidu, WAND-Vega)."
                )
                return

            spec = self._resolve_output(
                model,
                user_prompt,
                editing=is_editing,
                reference=self._reference_aspect(images) if is_editing else None,
            )

            if __event_emitter__:
                if spec.kind == "aspect":
                    detail = f"{spec.value}, {self.valves.vidu_resolution}"
                else:
                    detail = spec.status_value
                verb = "Editing" if is_editing else "Generating"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": (
                                f"{'✏️' if is_editing else '🎨'} {verb} image "
                                f"({detail}) using {model}…"
                            ),
                            "done": False,
                        },
                    }
                )

            if is_editing:
                urls = await self._edit_image(
                    model, user_prompt, images, spec, __event_emitter__
                )
            else:
                urls = await self._generate_image(
                    model, user_prompt, spec, __event_emitter__
                )

            yield self._render(
                model, urls, spec, is_editing, reference_origin
            )

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