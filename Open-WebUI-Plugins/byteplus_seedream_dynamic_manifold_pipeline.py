"""
title: BytePlus Seedream Dynamic Manifold Pipeline
author: dfirmans (v4.0 - seedream 5.5 ready)
version: 4.0.0
description: Dynamic multi-model image generation manifold for Open WebUI v0.11.0+ (tested on 0.12.x).
             Configure any number of Seedream/BytePlus image models via Valves as a comma-separated
             list — no code changes needed to add new models. Each model appears as a separate
             selectable model in the UI.
             Supports text-to-image, single-image editing, multi-reference blending, sequential
             batch generation (5.0 Lite), layer decomposition + rich-text OCR (5.5 / 5.0 Pro /
             5.0 Flash), native RGBA/transparent output (5.5), fast prompt optimization
             (5.0 Pro), size presets & explicit pixel sizes, reference images from uploads,
             pastes, and previously generated images.
             Updated for the BytePlus ModelArk Nov-2026 API: dola-seedream-5-5-261028,
             dola-seedream-5-0-pro-260628, dola-seedream-5-0-flash-260915, seedream-5-0-lite-260128.
required_open_webui_version: 0.11.0
requirements: pillow, httpx
"""

import base64
import io
import json
import re
import random
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from fastapi import Request
from PIL import Image
import httpx


# Ordered list of known resolution presets (used for clamping)
_PRESET_ORDER = ["1K", "1.5K", "2K", "3K", "4K"]

# Keywords mapped to each preset (matched case-insensitive with word boundaries)
_PRESET_KEYWORDS = {
    "1K": ["1k"],
    "1.5K": ["1.5k", "1,5k"],
    "2K": ["2k", "1440p"],
    "3K": ["3k"],
    "4K": ["4k", "4096p", "ultra hd", "uhd"],
}

# Keyword triggers for transparent-background output (checked case-insensitive)
_TRANSPARENT_KEYWORDS = [
    "transparent background",
    "background transparent",
    "transparent",
    "transparan",
    "alpha channel",
    "tanpa background",
    "latar belakang transparan",
]

# Keyword triggers for layer decomposition (without an explicit "layers:N" count)
_LAYER_KEYWORDS = [
    "layer decomposition",
    "decompose into layers",
    "decompose the image into layers",
    "split into layers",
    "pisah layer",
    "pisah menjadi layer",
    "layer decomp",
]

# Prompt directives supported by this pipe:
#   seed:42        — parsed but NOT sent (not part of the current BytePlus image API)
#   guidance:7.5   — parsed but NOT sent (not part of the current BytePlus image API)
#   layers:12      — layer decomposition, max_images=12 (5.5 / 5.0 Pro / 5.0 Flash, 1 input image)
#   transparent:yes|no — force background transparent / opaque (5.5 / 5.0 Pro / 5.0 Flash)
#   fast:yes|no    — optimize_prompt_options mode=fast (5.0 Pro & 4.0 only)
_DIRECTIVE_PATTERNS = [
    r"(?:seed|random[ _-]?seed)\s*[:=]\s*-?\d+",
    r"(?:guidance[ _-]?scale|guidance|cfg)\s*[:=]\s*\d+(?:\.\d+)?",
    r"(?:layers?|layer[ _-]?decomp)\s*[:=]\s*\d{1,3}",
    r"transparent\s*[:=]\s*(?:yes|no|true|false|1|0)",
    r"(?:fast|speed|optimize)\s*[:=]\s*(?:fast|standard|yes|no)",
]


class Pipe:
    """
    Dynamic manifold Pipe. Model list is user-configurable via Valves
    (one model per line), so adding/removing models needs no code changes.
    """

    # ---------------------------------------------------------------------
    # Valves — everything configurable here
    # ---------------------------------------------------------------------
    class Valves(BaseModel):
        api_key: str = Field(default="", description="BytePlus Ark API Key")
        base_url: str = Field(
            default="https://ark.ap-southeast.bytepluses.com/api/v3",
            description="BytePlus Ark API base URL",
        )

        # --- Dynamic model list (one model per line) ---
        # Format per line:  model_id|Display Name|flag1=value,flag2=value,...
        #
        # Available flags (values separated by commas; list-flags use ';'):
        #   seq=yes/no           supports sequential batch generation (5.0 Lite only)
        #   max_img=N            max output images for sequential generation
        #   img2img=yes/no       supports image-to-image editing / multi-reference
        #   max_ref=N            max reference images for img2img/blend
        #   layer=yes/no         supports layer_decomposition (5.5 / 5.0 Pro / 5.0 Flash)
        #   max_layers=N         max layers: 99 on 5.5, 16 on 5.0 Pro / Flash
        #   ocr=yes/no           supports layer_decomposition_options (max_images +
        #                        text_recognition) — currently 5.5 only
        #   background=VALUE     background param support & default:
        #                        auto (5.5), opaque (5.0 Pro/Flash), no (not supported)
        #   out_format=yes/no    output_format param supported (png/jpeg) — 5.x only
        #                        (4.5/4.0 are fixed jpeg, do not send)
        #   fast=yes/no          optimize_prompt_options.mode=fast allowed (5.0 Pro, 4.0)
        #   sizes=A;B;C          allowed resolution presets (1K/1.5K/2K/3K/4K)
        #   min_pixels=N         min total pixels for explicit WxH sizes (optional)
        #   max_pixels=N         max total pixels for explicit WxH sizes (optional)
        #   default_size=S       default size (e.g. 2K or 2048x2048)
        #
        # Legacy flag names from v3 are still accepted: stream=, has_guidance=,
        # transparency=yes/no (mapped to background=auto / background=no),
        # blend=yes/no, max_ref_images=, max_images=.
        #
        # Lines starting with # are ignored.
        models: str = Field(
            default=(
                "dola-seedream-5-5-261028|Seedream 5.5|seq=no,img2img=yes,max_ref=10,layer=yes,max_layers=99,ocr=yes,background=auto,out_format=yes,fast=no,min_pixels=921600,max_pixels=4624220,sizes=1K;1.5K;2K,default_size=2K\n"
                "dola-seedream-5-0-pro-260628|Seedream 5.0 Pro|seq=no,img2img=yes,max_ref=10,layer=yes,max_layers=16,ocr=no,background=opaque,out_format=yes,fast=yes,min_pixels=921600,max_pixels=4624220,sizes=1K;1.5K;2K,default_size=2K\n"
                "dola-seedream-5-0-flash-260915|Seedream 5.0 Flash|seq=no,img2img=yes,max_ref=10,layer=yes,max_layers=16,ocr=no,background=opaque,out_format=yes,fast=no,min_pixels=921600,max_pixels=4624220,sizes=1K;1.5K;2K,default_size=2K\n"
                "seedream-5-0-lite-260128|Seedream 5.0 Lite|seq=yes,max_img=15,img2img=yes,max_ref=14,layer=no,background=no,out_format=yes,fast=no,min_pixels=3686400,max_pixels=16777216,sizes=2K;3K;4K,default_size=2K"
            ),
            description=(
                "List of models, one per line. Format per line: "
                "model_id|Display Name|key=value,key=value,...\n"
                "Add or remove lines to change which models appear in the selector. "
                "No code changes needed.\n"
                "NOTE: if you upgraded from v3, the old value saved in the DB is kept — "
                "replace it with the current default to get the 5.5 / 5.0 Flash IDs."
            ),
        )

        # --- Global defaults ---
        output_format: str = Field(
            default="png",
            description="Output image format: png or jpeg (only sent to 5.x models)",
        )
        response_format: str = Field(
            default="url", description="Response format: url (24h expiry) or b64_json"
        )
        watermark: bool = Field(default=False, description="Add watermark to output")
        enable_size_detection: bool = Field(
            default=True,
            description="Auto-detect image size from prompt keywords",
        )
        enable_sequential_detection: bool = Field(
            default=True,
            description="Auto-enable batch sequence when prompt asks for multiple images",
        )
        enable_transparency_detection: bool = Field(
            default=True,
            description="Auto-set background=transparent when prompt asks for transparency",
        )
        default_max_images: int = Field(
            default=3,
            ge=1,
            le=15,
            description="Default number of images for sequential generation",
        )
        image_max_bytes: int = Field(
            default=30 * 1024 * 1024,
            description="Max raw bytes per base64 image payload (API limit: 30 MB)",
        )
        request_timeout: int = Field(
            default=300,
            description="API request timeout in seconds (layer decomposition can be slow)",
        )
        model_name_prefix: str = Field(
            default="Image / ",
            description="Prefix shown in the model selector",
        )
        send_safety_identifier: bool = Field(
            default=True,
            description="Send the Open WebUI user id as safety_identifier (abuse attribution)",
        )
        files_api_token: str = Field(
            default="",
            description="Optional Open WebUI API token used to fetch uploaded images when "
            "the request itself carries no Authorization header (e.g. direct API calls)",
        )
        debug: bool = Field(default=False, description="Print debug info to server logs")

    def __init__(self):
        self.valves = self.Valves()
        self._model_cache: Optional[List[Dict]] = None
        self._model_cache_hash: Optional[str] = None
        self.tooltips = [
            "Try: 'in the style of oil painting, cinematic lighting'",
            "Upload 1–10 images as reference — describe what to keep from each one",
            "Use '2K', '1.5K', '2048x1152' keywords to control size (5.x supports 1K/1.5K/2K)",
            "Seedream 5.5: add layers:12 (or 'layer decomposition') to split an image into editable layers",
            "Seedream 5.5: say 'transparent background' for native RGBA / sticker output",
            "Type '4 different variations' to auto-generate a batch (5.0 Lite)",
            "Add fast:yes to speed up 5.0 Pro generation (slightly lower quality)",
            "For editing: upload an image and say 'change the background to...'",
            "Follow-up edits: just ask — the last generated image is reused automatically",
            "Add new models anytime via the Valves — no code changes needed",
        ]

    # ---------------------------------------------------------------------
    # Model parser — converts Valves.models string into config dicts
    # ---------------------------------------------------------------------
    def _parse_models(self) -> List[Dict[str, Any]]:
        """Parse the models valve string into a list of model config dicts."""
        raw = self.valves.models.strip()
        if not raw:
            return []

        models = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 2:
                # Fallback: treat entire line as model_id, use id as name
                model_id = line.strip()
                name = model_id
                flags_str = ""
            else:
                model_id = parts[0].strip()
                name = parts[1].strip()
                flags_str = parts[2].strip() if len(parts) > 2 else ""

            # Parse flags — split on commas, keep list-values (which use ';') intact
            flags: Dict[str, Any] = {}
            if flags_str:
                for kv in flags_str.split(","):
                    kv = kv.strip()
                    if not kv or "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    # Type coercion (but keep ';' lists as strings for later parsing)
                    if ";" in v:
                        flags[k] = v
                    elif v.lower() in ("yes", "true", "1"):
                        flags[k] = True
                    elif v.lower() in ("no", "false", "0"):
                        flags[k] = False
                    elif re.fullmatch(r"-?\d+", v):
                        flags[k] = int(v)
                    else:
                        try:
                            flags[k] = float(v)
                        except ValueError:
                            flags[k] = v

            # Parse sizes list ( ';' separated )
            sizes = ["1K", "1.5K", "2K"]
            if "sizes" in flags and isinstance(flags["sizes"], str):
                parsed = [s.strip().upper() for s in flags["sizes"].split(";") if s.strip()]
                valid = [s for s in parsed if s in _PRESET_ORDER]
                if valid:
                    sizes = sorted(set(valid), key=_PRESET_ORDER.index)

            # Legacy v3 flag compatibility
            transparency = flags.get("transparency")
            background = flags.get(
                "background",
                ("auto" if transparency in (True, "yes") else "no") if transparency is not None else "no",
            )
            # 'no'/'false' get coerced to bool by the flag parser — normalize back
            if background is False:
                background = "no"
            elif background is True:
                background = "auto"
            background = str(background).lower()
            if background not in ("auto", "transparent", "opaque"):
                background = "no"
            has_guidance = flags.get("has_guidance", False)

            capabilities = {
                "text2img": True,
                "img2img": flags.get("img2img", True),
                "blend": flags.get("blend", flags.get("img2img", True)),
                "sequential": bool(flags.get("seq", flags.get("sequential", False))),
                "max_images": int(flags.get("max_img", flags.get("max_images", 1)) or 1),
                "max_ref_images": int(
                    flags.get("max_ref", flags.get("max_ref_images", 10)) or 10
                ),
                "layer_decomp": bool(flags.get("layer", False)),
                "max_layers": int(flags.get("max_layers", 16) or 16),
                "layer_options": bool(flags.get("ocr", False)),
                "background": background,
                "out_format": bool(flags.get("out_format", False)),
                "fast_mode": bool(flags.get("fast", False)),
                # guidance_scale is no longer part of the BytePlus image API;
                # keep the flag only to warn users who still set it.
                "has_guidance": bool(has_guidance),
                "sizes": sizes,
                "min_pixels": int(flags.get("min_pixels", 0) or 0),
                "max_pixels": int(flags.get("max_pixels", 0) or 0),
            }

            models.append(
                {
                    "id": model_id,
                    "name": name,
                    "capabilities": capabilities,
                    "default_size": str(flags.get("default_size", "2K")).upper(),
                }
            )

        if not models:
            # Ultimate fallback so there's always at least one model
            models.append(
                {
                    "id": "dola-seedream-5-5-261028",
                    "name": "Seedream 5.5 (default)",
                    "capabilities": {
                        "text2img": True,
                        "img2img": True,
                        "blend": True,
                        "sequential": False,
                        "max_images": 1,
                        "max_ref_images": 10,
                        "layer_decomp": True,
                        "max_layers": 99,
                        "layer_options": True,
                        "background": "auto",
                        "out_format": True,
                        "fast_mode": False,
                        "has_guidance": False,
                        "sizes": ["1K", "1.5K", "2K"],
                        "min_pixels": 921600,
                        "max_pixels": 4624220,
                    },
                    "default_size": "2K",
                }
            )

        return models

    def _get_models(self) -> List[Dict[str, Any]]:
        """Get parsed model list (cached by content hash of valve string)."""
        current_hash = hash(self.valves.models)
        if self._model_cache is None or self._model_cache_hash != current_hash:
            self._model_cache = self._parse_models()
            self._model_cache_hash = current_hash
        return self._model_cache

    # ---------------------------------------------------------------------
    # Manifold registration — called by Open WebUI to list available models
    # ---------------------------------------------------------------------
    async def pipes(self) -> List[Dict[str, str]]:
        """Register each configured model as a separate selectable model."""
        models = self._get_models()
        return [
            {
                "id": m["id"],
                "name": f"{self.valves.model_name_prefix}{m['name']}",
            }
            for m in models
        ]

    # ---------------------------------------------------------------------
    # Helpers — model lookup
    # ---------------------------------------------------------------------
    def _strip_model_prefix(self, model_id: str) -> str:
        """Extract bare model ID from a potentially prefixed selector value.

        Open WebUI addresses manifold sub-models as '<function_id>.<id>' and
        recovers the function id by splitting on the FIRST dot, so everything
        after the first dot is the sub-model id (it may contain dots itself).
        """
        if "." in model_id:
            return model_id.split(".", 1)[1]
        return model_id

    def _canonicalize_model_id(self, model_id: str) -> str:
        """Map legacy/renamed BytePlus model ids to their current form."""
        aliases = {
            # v3-era ids without the 'dola-' prefix (renamed by BytePlus)
            "seedream-5-0-pro-260628": "dola-seedream-5-0-pro-260628",
            "seedream-5-0-flash-260915": "dola-seedream-5-0-flash-260915",
            "seedream-5-5-261028": "dola-seedream-5-5-261028",
            # canonical lite id -> documented alias
            "seedream-5-0-260128": "seedream-5-0-lite-260128",
        }
        return aliases.get(model_id, model_id)

    def _get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Look up model config by its ID. Falls back to first model if unknown."""
        models = self._get_models()
        candidates = [model_id, self._canonicalize_model_id(model_id)]
        for candidate in candidates:
            for m in models:
                if candidate == m["id"]:
                    return m
                # Open WebUI prefixes sub-model ids — strip it if still present
                if candidate.endswith(f".{m['id']}"):
                    return m
        # Fallback to first model
        return models[0]

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def _validate_config(self) -> bool:
        return bool(self.valves.api_key)

    def _dbg(self, *args):
        if self.valves.debug:
            print("DEBUG[seedream]:", *args)

    # ---------------------------------------------------------------------
    # Image handling
    # ---------------------------------------------------------------------
    @staticmethod
    def _image_has_alpha(img: Image.Image) -> bool:
        if img.mode in ("RGBA", "LA"):
            return True
        if img.mode == "P" and "transparency" in img.info:
            return True
        return False

    @staticmethod
    def _flatten_to_rgb(img: Image.Image) -> Image.Image:
        """Composite an alpha image onto white so it can be saved as JPEG."""
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            return background
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    @staticmethod
    def _resize_image(img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        ratio = max_side / max(w, h)
        return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    def _encode_image_data_uri(self, img: Image.Image) -> str:
        """Convert a PIL image to a base64 data URI under image_max_bytes.

        Alpha images are kept as PNG (the API needs png for transparency);
        opaque images go through a JPEG quality ladder.
        """
        limit = self.valves.image_max_bytes

        # --- Alpha path: PNG ladder (resize step by step) ---
        if self._image_has_alpha(img):
            rgba = img.convert("RGBA") if img.mode != "RGBA" else img
            for max_side in (None, 4096, 3072, 2560, 2048, 1536, 1024):
                candidate = rgba if max_side is None else self._resize_image(rgba, max_side)
                buf = io.BytesIO()
                candidate.save(buf, format="PNG", optimize=True)
                raw = buf.getvalue()
                if len(raw) <= limit:
                    return f"data:image/png;base64,{base64.b64encode(raw).decode('utf-8')}"
            # Could not keep alpha under the limit — flatten and fall through
            img = self._flatten_to_rgb(rgba)
        elif img.mode != "RGB":
            img = self._flatten_to_rgb(img)

        # --- Opaque path: JPEG quality ladder ---
        def _encode_jpeg(image: Image.Image, quality: int) -> Tuple[str, int]:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
            return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('utf-8')}", len(raw)

        for q in (95, 90, 85, 80, 75, 70, 60, 50):
            data_url, raw_size = _encode_jpeg(img, q)
            if raw_size <= limit:
                return data_url

        for max_side in (2560, 2048, 1536):
            img = self._resize_image(img, max_side)
            for q in (85, 75, 65, 55, 45):
                data_url, raw_size = _encode_jpeg(img, q)
                if raw_size <= limit:
                    return data_url

        raise ValueError(
            f"Image cannot be compressed below {limit / 1024 / 1024:.1f} MB limit."
        )

    async def _resolve_image_bytes(
        self, src: str, __request__: Optional[Request] = None
    ) -> Optional[bytes]:
        """Resolve an image source into raw bytes.

        Supported sources:
          - data URIs (data:image/...;base64,...)
          - http(s) URLs (e.g. BytePlus output URLs from earlier turns)
          - Open WebUI server files: bare file id or /api/v1/files/<id>/content
            (fetched with the caller's Authorization header, or files_api_token)
        """
        src = (src or "").strip()
        if not src:
            return None

        # 1. Inline data URI
        if src.lower().startswith("data:"):
            try:
                return base64.b64decode(src.split(",", 1)[1])
            except Exception as e:
                self._dbg("data URI decode failed:", e)
                return None

        headers = {}
        auth = __request__.headers.get("authorization") if __request__ else None
        if auth:
            headers["Authorization"] = auth
        elif self.valves.files_api_token:
            headers["Authorization"] = f"Bearer {self.valves.files_api_token}"

        low = src.lower()
        if low.startswith("http://") or low.startswith("https://"):
            url = src
        else:
            base = str(__request__.base_url).rstrip("/") if __request__ else ""
            if not base:
                self._dbg("cannot resolve server file without request base url:", src)
                return None
            if src.startswith("/"):
                url = f"{base}{src}"
            else:
                # Bare Open WebUI file id (current frontend sends file.id as url)
                url = f"{base}/api/v1/files/{src}/content"

        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code == 200 and resp.content:
                return resp.content
            self._dbg(f"image fetch failed HTTP {resp.status_code}: {url[:120]}")
        except Exception as e:
            self._dbg(f"image fetch error ({url[:120]}): {e}")
        return None

    async def _extract_images_from_messages(
        self,
        messages: List[Dict[str, Any]],
        __request__: Optional[Request] = None,
        extra_files: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract reference images from the conversation, newest message first.

        Attached-image sources (used first):
          1. message['files'] entries of type 'image' (or raster content_type) —
             current Open WebUI puts uploads here with url = file id or data URI.
          2. multimodal content lists: {'type': 'image_url', 'image_url': {'url': ...}}
             — injected by Open WebUI middleware for every user message with images.
          3. markdown images with data URIs in any message.

        Fallback source (only when NO attached image was found, so a fresh
        upload always wins over older generated results):
          4. markdown images in assistant messages — http(s) or server file URLs
             of previously generated images (enables multi-turn editing).
        """
        attached: List[str] = []
        generated_urls: List[str] = []

        def _collect_from_files(files: Any):
            for f in files or []:
                if not isinstance(f, dict):
                    continue
                ctype = (f.get("content_type") or "").lower()
                if f.get("type") == "image" or ctype.startswith("image/"):
                    url = f.get("url") or f.get("data") or ""
                    if url:
                        attached.append(url)

        def _collect_from_content(content: Any, is_assistant: bool):
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in ("image_url", "input_image"):
                        url = item.get("image_url", {})
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        elif not isinstance(url, str):
                            url = ""
                        if url:
                            attached.append(url)
            elif isinstance(content, str):
                # Attached images embedded as markdown data URIs (any role)
                attached.extend(
                    re.findall(r"!\[[^\]]*\]\((data:image/[^)\s]+)\)", content)
                )
                if is_assistant:
                    # Previously generated results (fallback source only)
                    generated_urls.extend(
                        re.findall(
                            r"!\[[^\]]*\]\((https?://[^)\s]+|/api/v1/files/[^)\s]+)\)",
                            content,
                        )
                    )

        for msg in reversed(messages):
            role = msg.get("role")
            _collect_from_files(msg.get("files"))
            _collect_from_content(msg.get("content", ""), is_assistant=(role == "assistant"))

        # Top-level files key (some Open WebUI paths attach files to the body)
        if extra_files:
            _collect_from_files(extra_files)

        candidates = attached if attached else generated_urls
        if not candidates:
            return []

        # Resolve bytes (dedupe by source string first to skip double work)
        seen_src = set()
        raw_images: List[Image.Image] = []
        for src in candidates:
            if src in seen_src:
                continue
            seen_src.add(src)
            try:
                data = await self._resolve_image_bytes(src, __request__)
                if not data:
                    continue
                pil_img = Image.open(io.BytesIO(data))
                pil_img.load()
                raw_images.append(pil_img)
            except Exception as e:
                self._dbg("image load failed:", e)

        if not raw_images:
            return []

        # Deduplicate by pixel data (same upload often appears in files AND content)
        seen = set()
        unique = []
        for img in raw_images:
            try:
                key = img.tobytes()
            except Exception:
                key = id(img)
            if key not in seen:
                seen.add(key)
                unique.append(img)

        result = []
        for img in unique:
            try:
                result.append(
                    {
                        "pil": img,
                        "base64_url": self._encode_image_data_uri(img),
                        "has_alpha": self._image_has_alpha(img),
                    }
                )
            except Exception as e:
                self._dbg("image encode failed:", e)
        return result

    # ---------------------------------------------------------------------
    # Prompt parsing
    # ---------------------------------------------------------------------
    def _extract_prompt_text(self, body: dict, __metadata__: Optional[dict]) -> str:
        """Get the user's prompt text from the last user message.

        Prefers __metadata__['user_prompt'] (recommended by Open WebUI — the
        raw prompt before citation wrapping), falls back to scanning messages
        for the last 'user' message (body['messages'] may end with a tool
        message when native function calling is enabled).
        """
        prompt = (__metadata__ or {}).get("user_prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                text = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                text = str(content)
            if text.strip():
                return text.strip()
        return ""

    def _parse_directives(self, prompt: str) -> Dict[str, Any]:
        """Parse inline control directives from the prompt."""
        out: Dict[str, Any] = {
            "layers": None,
            "transparent": None,
            "fast": None,
            "seed": None,
            "guidance": None,
        }

        m = re.search(
            r"(?:seed|random[ _-]?seed)\s*[:=]\s*(-?\d+)", prompt, re.IGNORECASE
        )
        if m:
            out["seed"] = int(m.group(1))

        m = re.search(
            r"(?:guidance[ _-]?scale|guidance|cfg)\s*[:=]\s*(\d+(?:\.\d+)?)",
            prompt,
            re.IGNORECASE,
        )
        if m:
            out["guidance"] = float(m.group(1))

        m = re.search(r"(?:layers?|layer[ _-]?decomp)\s*[:=]\s*(\d{1,3})", prompt, re.IGNORECASE)
        if m:
            out["layers"] = int(m.group(1))

        m = re.search(
            r"transparent\s*[:=]\s*(yes|no|true|false|1|0)", prompt, re.IGNORECASE
        )
        if m:
            out["transparent"] = m.group(1).lower() in ("yes", "true", "1")

        m = re.search(
            r"(?:fast|speed|optimize)\s*[:=]\s*(fast|standard|yes|no)", prompt, re.IGNORECASE
        )
        if m:
            out["fast"] = m.group(1).lower() in ("fast", "yes")

        return out

    def _clean_prompt(self, prompt: str) -> str:
        """Remove control directives from the prompt."""
        for pattern in _DIRECTIVE_PATTERNS:
            prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE)
        return prompt.strip()

    def _detect_size(self, prompt: str, model_cfg: Dict) -> Tuple[str, Optional[str]]:
        """Detect the requested size from the prompt.

        Returns (size, note) where note explains any adjustment made.
        Supports:
          - resolution presets (1K / 1.5K / 2K / 3K / 4K), clamped to the model's
            allowed preset list
          - explicit pixel dimensions ('2048x1152'), validated against the
            model's total-pixel range
        """
        cap = model_cfg["capabilities"]
        default = model_cfg["default_size"]
        prompt_lower = prompt.lower()

        explicit: Optional[str] = None
        m = re.search(r"(\d{3,4})\s*[x×*]\s*(\d{3,4})", prompt_lower)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            total = w * h
            min_px = cap.get("min_pixels") or 0
            max_px = cap.get("max_pixels") or 0
            aspect_ok = 0.0625 <= (w / h if h else 0) <= 16
            if aspect_ok and (not max_px or total <= max_px) and (not min_px or total >= min_px):
                explicit = f"{w}x{h}"
            else:
                return default, (
                    f"Requested size {w}x{h} is outside {model_cfg['name']}'s supported "
                    f"range — using {default} instead."
                )

        preset: Optional[str] = None
        for size, keywords in _PRESET_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"(?<![\d.]){re.escape(kw)}(?![\d])", prompt_lower):
                    preset = size
                    break
            if preset:
                break

        size = explicit or preset or default

        if explicit:
            return size, None

        if size in cap["sizes"]:
            return size, None

        # Preset not supported by this model — clamp to the closest allowed one
        requested_idx = (
            _PRESET_ORDER.index(size) if size in _PRESET_ORDER else len(_PRESET_ORDER) - 1
        )
        allowed = cap["sizes"]
        fallback = None
        for candidate in reversed(allowed):
            if _PRESET_ORDER.index(candidate) <= requested_idx:
                fallback = candidate
                break
        if fallback is None:
            fallback = allowed[0]
        return fallback, (
            f"{model_cfg['name']} does not support {size} — using {fallback} instead "
            f"(supported: {', '.join(allowed)})."
        )

    def _detect_sequential_request(
        self, prompt: str, model_cfg: Dict, img_count: int
    ) -> Tuple[bool, int, Optional[str]]:
        """Detect if the user wants multiple images.

        Works for text-to-batch and image-to-batch (5.0 Lite supports both).
        Returns (enabled, count, note).
        """
        cap = model_cfg["capabilities"]
        if not cap["sequential"]:
            return False, 1, None

        cap_max = cap["max_images"] or 1
        if cap_max < 2:
            # Sequential needs at least 2 output images to be meaningful
            return False, 1, None

        prompt_lower = prompt.lower()
        count: Optional[int] = None

        m = re.search(
            r"(\d+)\s+(images?|pics?|pictures?|variations?|results?|versions?|shots?|"
            r"gambar|foto|variasi|versi|hasil|scenes?|frames?|storyboards?)",
            prompt_lower,
        )
        if m:
            count = int(m.group(1))
        else:
            seq_keywords = [
                "different angles",
                "different variations",
                "multiple images",
                "several images",
                "generate 3",
                "create 4",
                "morning, noon and night",
                "storyboard",
                "beberapa gambar",
            ]
            if any(k in prompt_lower for k in seq_keywords):
                count = self.valves.default_max_images

        if not count or count < 2:
            return False, 1, None

        note = None

        # Hard API rule for batch-capable models: input refs + outputs <= 15
        if img_count > 0:
            room = 15 - img_count
            if room < 2:
                return False, 1, (
                    f"Sequential batch disabled: {img_count} reference image(s) leave no "
                    f"room for batch output (refs + outputs must be <= 15)."
                )
            if count > room:
                note = f"Batch count clamped to {room} (refs + outputs must be <= 15)."
                count = room

        if count > cap_max:
            note = (note or "") + (
                f"{model_cfg['name']} supports at most {cap_max} image(s) per batch — clamped."
            ).strip()
            count = cap_max

        return True, max(2, min(count, cap_max)), note

    def _detect_transparency(self, prompt: str, directives: Dict) -> Optional[bool]:
        """Decide whether transparent background was requested."""
        if directives.get("transparent") is not None:
            return bool(directives["transparent"])
        if self.valves.enable_transparency_detection:
            prompt_lower = prompt.lower()
            if any(k in prompt_lower for k in _TRANSPARENT_KEYWORDS):
                return True
        return None

    def _detect_layer_decomposition(self, prompt: str, directives: Dict) -> Optional[int]:
        """Return requested max layer images (2+) when layer decomposition is asked.

        Returns None when not requested; returns the default count when triggered
        by keyword without an explicit 'layers:N' directive.
        """
        if directives.get("layers"):
            return directives["layers"]
        prompt_lower = prompt.lower()
        if any(k in prompt_lower for k in _LAYER_KEYWORDS):
            return 30  # API default for max_images
        return None

    # ---------------------------------------------------------------------
    # Payload builder
    # ---------------------------------------------------------------------
    def _build_payload(
        self,
        model_cfg: Dict[str, Any],
        prompt: str,
        images: List[str],
        size: str,
        background: Optional[str],
        layer_decomp: bool,
        layer_options: Optional[Dict[str, Any]],
        optimize_mode: Optional[str],
        seq_count: int,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Build the /images/generations payload for the given mode & capabilities."""
        cap = model_cfg["capabilities"]
        payload: Dict[str, Any] = {
            "model": model_cfg["id"],
            "prompt": prompt,
        }

        if images:
            payload["image"] = images[0] if len(images) == 1 else images

        if size:
            payload["size"] = size

        # output_format is only accepted by 5.x models (4.5/4.0 are fixed jpeg)
        if cap["out_format"]:
            payload["output_format"] = (
                "png" if background == "transparent" else self.valves.output_format
            )

        # background only applies to image generation (NOT layer decomposition)
        if background and not layer_decomp:
            payload["background"] = background

        # Prompt optimization mode (standard default; fast only on some models)
        if optimize_mode:
            payload["optimize_prompt_options"] = {"mode": optimize_mode}

        if layer_decomp:
            payload["layer_decomposition"] = True
            if layer_options:
                payload["layer_decomposition_options"] = layer_options
        elif cap["sequential"] and seq_count > 1:
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {"max_images": seq_count}

        payload["response_format"] = self.valves.response_format
        payload["watermark"] = self.valves.watermark

        if self.valves.send_safety_identifier and user_id:
            payload["safety_identifier"] = str(user_id)[:64]

        return payload

    # ---------------------------------------------------------------------
    # BytePlus API caller
    # ---------------------------------------------------------------------
    async def _call_images_generations(
        self, payload: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Call BytePlus /images/generations with native async httpx."""
        url = f"{self.valves.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }
        self._dbg("API call →", url, "| model:", payload.get("model"))

        timeout = httpx.Timeout(self.valves.request_timeout, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                msg = f"API call failed – HTTP {resp.status_code}"
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("message") or err_data.get("error", {}).get(
                        "message", resp.text
                    )
                    msg += f": {err_msg}"
                except Exception:
                    msg += f": {resp.text}"
                raise Exception(msg)

            data = resp.json()

        results = []
        for item in data.get("data", []):
            results.append(
                {
                    "url": item.get("url"),
                    "b64_json": item.get("b64_json"),
                    "size": item.get("size"),
                    "output_format": item.get("output_format"),
                    "z_index": item.get("z_index"),
                    "bounding_box": item.get("bounding_box"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "text_recognition": item.get("text_recognition"),
                }
            )

        usage = data.get("usage", {}) or {}
        if not results:
            err = data.get("error")
            if err:
                raise ValueError(f"API returned an error: {json.dumps(err, indent=2)}")
            raise ValueError(f"No image data in response: {json.dumps(data, indent=2)}")
        return results, usage

    # ---------------------------------------------------------------------
    # Response formatting
    # ---------------------------------------------------------------------
    def _result_image_url(self, res: Dict[str, Any]) -> Optional[str]:
        url = res.get("url")
        if not url and res.get("b64_json"):
            fmt = (res.get("output_format") or self.valves.output_format or "png").lower()
            url = f"data:image/{fmt};base64,{res['b64_json']}"
        return url

    def _build_response_markdown(
        self,
        results: List[Dict[str, Any]],
        usage: Dict[str, Any],
        model_cfg: Dict[str, Any],
        mode: str,
        size: str,
        background: Optional[str],
        optimize_mode: Optional[str],
        directives: Dict[str, Any],
        notes: List[str],
        img_count: int,
    ) -> str:
        """Build the assistant reply markdown for the generation result."""
        lines: List[str] = []

        icons = {
            "blend": "**🎨 Blended Result",
            "img2img": "**✏️ Edited Result",
            "layer": "**🧩 Layer Decomposition",
            "text2img": "**🖼️ Generated Result",
        }
        header = icons.get(mode, icons["text2img"])
        lines.append(f"{header} ({len(results)} output(s)):**\n")

        if mode == "layer":
            for idx, res in enumerate(results, start=1):
                url = self._result_image_url(res)
                if not url:
                    continue
                z = res.get("z_index")
                name = res.get("name") or (f"Base image" if idx == 1 else f"Layer {idx - 1}")
                label = f"{name} (z={z})" if z is not None else name
                lines.append(f"**{label}**\n")
                lines.append(f"![{label}]({url})\n")
                if res.get("description"):
                    lines.append(f"_Description:_ {res['description']}\n")
                tr = res.get("text_recognition")
                if tr and isinstance(tr, dict) and tr.get("text"):
                    ocr_text = tr["text"].replace("\n", " · ")
                    lines.append(f"_OCR text:_ `{ocr_text}`\n")
        else:
            for idx, res in enumerate(results, start=1):
                url = self._result_image_url(res)
                if not url:
                    continue
                label = f"Image {idx}" if len(results) > 1 else "Result"
                lines.append(f"![{label}]({url})\n")

        # Metadata line
        meta_parts = [
            f"**Size:** `{size}`",
            f"**Model:** `{model_cfg['name']}`",
            f"**Outputs:** `{len(results)}`",
        ]
        if img_count:
            meta_parts.append(f"**Refs:** `{img_count}`")
        if background:
            meta_parts.append(f"**Background:** `{background}`")
        if optimize_mode:
            meta_parts.append(f"**Optimize:** `{optimize_mode}`")
        if usage.get("total_tokens"):
            meta_parts.append(f"**Tokens:** `{usage.get('total_tokens')}`")
        lines.append("\n" + " | ".join(meta_parts))

        # Notes / adjustments surfaced to the user
        if directives.get("seed") is not None:
            notes.append(
                "seed:N was not sent — the current BytePlus image API does not expose a "
                "seed parameter for Seedream 5.x models."
            )
        if directives.get("guidance") is not None:
            notes.append(
                "guidance:N was not sent — the current BytePlus image API does not expose "
                "guidance_scale."
            )

        if notes:
            lines.append("\n**Notes:**")
            for note in notes:
                lines.append(f"- {note}")

        lines.append(f"\n💡 _Tip: {random.choice(self.tooltips)}_")

        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Main pipe entry point
    # ---------------------------------------------------------------------
    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: dict = None,
        __metadata__: dict = None,
        __event_emitter__=None,
        __task__: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        # 0. Skip background tasks (title/tags/query generation) — don't burn API credits
        if __task__:
            return

        # 1. Basic validation
        if not self._validate_config():
            yield (
                "❌ **Error:** BytePlus API key is missing. "
                "Configure it in the Pipe Valves."
            )
            return

        notes: List[str] = []
        model_cfg: Dict[str, Any] = {}

        try:
            # 2. Resolve which model was selected from the manifold
            raw_model = body.get("model", "")
            model_id = self._strip_model_prefix(raw_model)
            model_cfg = self._get_model_config(model_id)
            cap = model_cfg["capabilities"]

            self._dbg("Manifold model →", model_cfg["id"], f"({model_cfg['name']})")

            # 3. Extract prompt
            user_prompt = self._extract_prompt_text(body, __metadata__)
            if not user_prompt:
                yield "❌ **Error:** No text prompt found."
                return

            # 4. Parse control directives & clean the prompt
            directives = self._parse_directives(user_prompt)
            clean_prompt = self._clean_prompt(user_prompt)
            if not clean_prompt:
                yield "❌ **Error:** Prompt is empty after removing control directives."
                return

            # 5. Extract reference images (current message first, then history)
            images = await self._extract_images_from_messages(
                body.get("messages", []), __request__, extra_files=body.get("files")
            )
            image_urls = [img["base64_url"] for img in images]
            img_count = len(image_urls)
            alpha_inputs = sum(1 for img in images if img.get("has_alpha"))

            # Clamp reference images to the model's limit
            if img_count > cap["max_ref_images"]:
                notes.append(
                    f"Reference images clamped from {img_count} to {cap['max_ref_images']} "
                    f"({model_cfg['name']} limit)."
                )
                image_urls = image_urls[: cap["max_ref_images"]]
                alpha_inputs = sum(
                    1 for img in images[: len(image_urls)] if img.get("has_alpha")
                )
                img_count = len(image_urls)

            # 6. Determine mode
            requested_layers = self._detect_layer_decomposition(clean_prompt, directives)
            layer_mode = False
            if requested_layers is not None:
                if not cap["layer_decomp"]:
                    notes.append(
                        f"{model_cfg['name']} does not support layer decomposition — "
                        f"ignored (supported by Seedream 5.5 / 5.0 Pro / 5.0 Flash)."
                    )
                elif img_count == 0:
                    notes.append(
                        "Layer decomposition needs one input image — upload an image to decompose."
                    )
                elif img_count > 1:
                    notes.append(
                        "Layer decomposition accepts only a single input image — "
                        "using the first image only."
                    )
                    image_urls = image_urls[:1]
                    img_count = 1
                    alpha_inputs = 1 if images and images[0].get("has_alpha") else 0
                else:
                    layer_mode = True

            if layer_mode:
                mode = "layer"
            elif img_count == 0:
                mode = "text2img"
            elif img_count == 1:
                mode = "img2img"
            else:
                mode = "blend"

            if mode in ("img2img", "blend") and not cap["img2img"]:
                yield (
                    f"❌ **Error:** {model_cfg['name']} does not support image-to-image "
                    f"editing.\nTry a model with `img2img=yes` capability in your Valves "
                    f"models list."
                )
                return

            # 7. Size detection
            if self.valves.enable_size_detection:
                size, size_note = self._detect_size(clean_prompt, model_cfg)
                if size_note:
                    notes.append(size_note)
            else:
                size, size_note = model_cfg["default_size"], None

            # Layer decomposition only accepts presets 1K/1.5K/2K/auto — no pixel dims
            if layer_mode and "x" in size.lower():
                size = (
                    model_cfg["default_size"]
                    if model_cfg["default_size"] in cap["sizes"]
                    else cap["sizes"][-1]
                )
                notes.append(
                    f"Explicit pixel sizes are not supported for layer decomposition — using {size}."
                )

            # 8. Sequential (batch) detection — 5.0 Lite supports text- AND image-to-batch
            seq_enabled, seq_count = False, 1
            if not layer_mode and self.valves.enable_sequential_detection:
                seq_enabled, seq_count, seq_note = self._detect_sequential_request(
                    clean_prompt, model_cfg, img_count
                )
                if seq_note:
                    notes.append(seq_note)

            # 9. Background / transparency handling
            background: Optional[str] = None
            want_transparent = self._detect_transparency(clean_prompt, directives)
            if directives.get("transparent") is False:
                want_transparent = False

            if want_transparent and not layer_mode:
                if cap["background"] == "no":
                    notes.append(
                        f"{model_cfg['name']} does not support the background parameter — "
                        f"transparency ignored (supported by Seedream 5.5 / 5.0 Pro / 5.0 Flash)."
                    )
                elif mode == "text2img" and cap["background"] == "opaque":
                    notes.append(
                        f"{model_cfg['name']} only generates RGBA from image input "
                        f"(one transparent PNG reference) — transparency ignored for text-to-image."
                    )
                elif mode == "blend" and cap["background"] == "opaque":
                    notes.append(
                        f"{model_cfg['name']} RGBA output requires a single transparent PNG "
                        f"reference — transparency ignored for multi-image blending."
                    )
                elif mode == "img2img" and cap["background"] == "opaque" and alpha_inputs < 1:
                    notes.append(
                        f"{model_cfg['name']} needs one transparent PNG input for RGBA output — "
                        f"transparency ignored (use Seedream 5.5 for RGBA from any input)."
                    )
                else:
                    background = "transparent"
            elif directives.get("transparent") is False and cap["background"] != "no":
                background = "opaque"

            # 10. Prompt optimization mode (fast) — 5.0 Pro & 4.0 only
            optimize_mode: Optional[str] = None
            if directives.get("fast"):
                if cap["fast_mode"]:
                    optimize_mode = "fast"
                else:
                    notes.append(
                        f"fast:yes is not supported by {model_cfg['name']} — using standard "
                        f"(supported by Seedream 5.0 Pro)."
                    )

            # 11. Layer decomposition options (max_images / text_recognition — 5.5 only)
            layer_options: Optional[Dict[str, Any]] = None
            if layer_mode and cap["layer_options"]:
                max_layers = cap["max_layers"] or 99
                requested = max(2, min(requested_layers or 30, max_layers + 1))
                layer_options = {"max_images": requested}
                if requested_layers and requested_layers != requested:
                    notes.append(
                        f"layers:{requested_layers} clamped to {requested} "
                        f"(range 2–{max_layers + 1} for {model_cfg['name']})."
                    )

            # 12. Build payload and call API
            payload = self._build_payload(
                model_cfg=model_cfg,
                prompt=clean_prompt,
                images=image_urls,
                size=size,
                background=background,
                layer_decomp=layer_mode,
                layer_options=layer_options,
                optimize_mode=optimize_mode,
                seq_count=seq_count if seq_enabled else 1,
                user_id=(__user__ or {}).get("id"),
            )

            # Status emitter
            if __event_emitter__:
                if mode == "layer":
                    status = (
                        f"🧩 Decomposing image into layers at {size} "
                        f"using {model_cfg['name']}…"
                    )
                elif mode == "blend":
                    status = (
                        f"🎨 Blending {img_count} images at {size} "
                        f"using {model_cfg['name']}…"
                    )
                elif mode == "img2img":
                    extra = f" → {seq_count} image(s)" if seq_enabled else ""
                    status = (
                        f"✏️ Editing image at {size}{extra} using {model_cfg['name']}…"
                    )
                else:
                    extra = f" → {seq_count} image(s)" if seq_enabled else ""
                    status = (
                        f"🖼️ Generating image at {size}{extra} "
                        f"using {model_cfg['name']}…"
                    )
                await __event_emitter__(
                    {"type": "status", "data": {"description": status, "done": False}}
                )

            self._dbg(
                f"Mode={mode}, Images={img_count}, Size={size}, "
                f"Seq={seq_enabled}({seq_count}), Background={background}, "
                f"Layer={layer_mode}({layer_options}), Fast={optimize_mode}"
            )

            results, usage = await self._call_images_generations(payload)

            # 13. Build response
            markdown = self._build_response_markdown(
                results=results,
                usage=usage,
                model_cfg=model_cfg,
                mode=mode,
                size=size,
                background=background,
                optimize_mode=optimize_mode,
                directives=directives,
                notes=notes,
                img_count=img_count,
            )
            yield markdown

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "✅ Done", "done": True}}
                )

        except Exception as err:
            yield f"❌ **Error:**\n\n{str(err)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "❌ Error occurred", "done": True},
                    }
                )
