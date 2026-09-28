"""
title: BytePlus Seedream Dynamic Manifold Pipeline
author: dfirmans (v3.0 - dynamic manifold)
version: 3.0.0
description: Dynamic multi-model image generation manifold for Open WebUI v0.11.0+.
             Configure any number of Seedream/BytePlus image models via Valves
             as a comma-separated list — no code changes needed to add new models.
             Each model appears as a separate selectable model in the UI.
             Supports text-to-image, image-to-image editing, multi-image fusion,
             sequential batch generation, size presets, seed & guidance control.
required_open_webui_version: 0.11.0
requirements: pillow, httpx
"""

import base64
import io
import json
import re
import time
import uuid
import random
from typing import AsyncGenerator, Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import Request
from PIL import Image
import httpx


class Pipe:
    """
    Dynamic manifold Pipe. Model list is user-configurable via Valves
    (comma-separated), so adding/removing models needs no code changes.
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
        # Available flags:
        #   seq=yes/no          supports sequential batch generation
        #   img2img=yes/no      supports image-to-image editing
        #   blend=yes/no        supports multi-image blending
        #   stream=yes/no       supports streaming output
        #   has_guidance=yes/no supports guidance_scale parameter
        #   transparency=yes/no supports PNG transparency / alpha
        #   max_ref=N           max reference images for img2img/blend
        #   max_img=N           max output images for sequential generation
        #   default_size=S      default size (e.g. 2K, 1024x1024)
        #   default_guidance=N  default guidance_scale (only if has_guidance=yes)
        #
        # Lines starting with # are ignored.
        # Add/remove lines to change models — no code changes needed.
        models: str = Field(
            default=(
                "seedream-5-0-pro-260628|Seedream 5.0 Pro|seq=no,img2img=yes,blend=yes,stream=no,has_guidance=no,max_ref=10,max_img=1,default_size=2K,transparency=yes\n"
                "seedream-5-0-lite-260128|Seedream 5.0 Lite|seq=yes,img2img=yes,blend=yes,stream=yes,has_guidance=no,max_ref=14,max_img=15,default_size=2K,transparency=yes\n"
                "seedream-4-0-250828|Seedream 4.0|seq=yes,img2img=yes,blend=yes,stream=no,has_guidance=yes,max_ref=10,max_img=15,default_size=2K,transparency=no,default_guidance=5.5\n"
                "seedream-3-0-t2i-250415|Seedream 3.0|seq=no,img2img=no,blend=no,stream=no,has_guidance=yes,max_ref=0,max_img=1,default_size=1024x1024,transparency=no,default_guidance=2.5"
            ),
            description=(
                "List of models, one per line. Format per line: "
                "model_id|Display Name|key=value,key=value,...\n"
                "Add or remove lines to change which models appear in the selector. "
                "No code changes needed."
            ),
        )

        # --- Global defaults ---
        output_format: str = Field(
            default="png",
            description="Output image format: png (5.0+), jpeg (all)",
        )
        response_format: str = Field(
            default="url", description="Response format: url or b64_json"
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
        default_max_images: int = Field(
            default=3,
            ge=1,
            le=15,
            description="Default number of images for sequential generation",
        )
        image_max_bytes: int = Field(
            default=20 * 1024 * 1024,
            description="Max raw bytes per base64 image payload",
        )
        request_timeout: int = Field(
            default=180, description="API request timeout in seconds"
        )
        model_name_prefix: str = Field(
            default="Image / ",
            description="Prefix shown in the model selector",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._model_cache: Optional[List[Dict]] = None
        self._model_cache_hash: Optional[str] = None
        self.tooltips = [
            "Try: 'in the style of oil painting, cinematic lighting'",
            "Upload 2+ images for multi-reference fusion — describe what to keep from each",
            "Use '4K', '2K', 'portrait', 'landscape' keywords to control size",
            "Add seed:42 to your prompt for reproducible results",
            "Add guidance:7 to strengthen prompt adherence (1-10)",
            "Type '3 different variations' to auto-generate a batch",
            "For editing: upload an image and say 'change the background to...'",
            "Switch between models in the selector to compare quality",
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

            # Parse flags
            flags: Dict[str, Any] = {}
            if flags_str:
                for kv in flags_str.split(","):
                    kv = kv.strip()
                    if not kv or "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    # Type coercion
                    if v.lower() in ("yes", "true", "1"):
                        flags[k] = True
                    elif v.lower() in ("no", "false", "0"):
                        flags[k] = False
                    elif v.isdigit():
                        flags[k] = int(v)
                    else:
                        try:
                            flags[k] = float(v)
                        except ValueError:
                            flags[k] = v

            # Build capabilities dict with defaults
            has_guidance = flags.get("has_guidance", False)
            default_guidance = (
                flags.get("default_guidance", 5.5) if has_guidance else None
            )

            capabilities = {
                "text2img": True,
                "img2img": flags.get("img2img", True),
                "blend": flags.get("blend", True),
                "sequential": flags.get("seq", flags.get("sequential", True)),
                "streaming": flags.get("stream", flags.get("streaming", False)),
                "has_guidance": has_guidance,
                "transparency": flags.get("transparency", False),
                "max_ref_images": flags.get("max_ref", flags.get("max_ref_images", 10)),
                "max_images": flags.get("max_img", flags.get("max_images", 1)),
                "sizes": flags.get(
                    "sizes",
                    ["1K", "2K", "4K", "1024x1024", "768x1344", "1344x768"],
                ),
            }

            models.append(
                {
                    "id": model_id,
                    "name": name,
                    "capabilities": capabilities,
                    "default_size": flags.get("default_size", "2K"),
                    "default_guidance": default_guidance,
                }
            )

        if not models:
            # Ultimate fallback so there's always at least one model
            models.append(
                {
                    "id": "seedream-5-0-lite-260128",
                    "name": "Seedream 5.0 Lite (default)",
                    "capabilities": {
                        "text2img": True,
                        "img2img": True,
                        "blend": True,
                        "sequential": True,
                        "streaming": True,
                        "transparency": True,
                        "max_ref_images": 14,
                        "max_images": 15,
                        "sizes": ["1K", "2K", "3K", "4K"],
                    },
                    "default_size": "2K",
                    "default_guidance": 5.5,
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
    def _get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Look up model config by its ID. Falls back to first model if unknown."""
        models = self._get_models()
        for m in models:
            if model_id == m["id"]:
                return m
            # Open WebUI may prefix the model name with the pipe id — strip it
            if model_id.endswith(f".{m['id']}"):
                return m
        # Fallback to first model
        return models[0]

    def _strip_model_prefix(self, model_id: str) -> str:
        """Extract bare model ID from a potentially prefixed selector value."""
        if "." in model_id:
            # Handle "pipeid.modelid" format
            return model_id.split(".", 1)[1]
        return model_id

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def _validate_config(self) -> bool:
        return bool(self.valves.api_key)

    # ---------------------------------------------------------------------
    # Image handling
    # ---------------------------------------------------------------------
    def _compress_image_to_b64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 JPEG Data URI, keeping under max bytes."""
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

        def _encode(image: Image.Image, quality: int) -> tuple:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
            b64 = base64.b64encode(raw).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}", len(raw)

        def _resize(image: Image.Image, max_side: int) -> Image.Image:
            w, h = image.size
            if max(w, h) <= max_side:
                return image
            ratio = max_side / max(w, h)
            return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        # Pass 1: quality ladder at original size
        for q in (95, 90, 85, 80, 75, 70, 60, 50):
            data_url, raw_size = _encode(img, q)
            if raw_size <= self.valves.image_max_bytes:
                return data_url

        # Pass 2: resize to 2560px and retry
        img = _resize(img, 2560)
        for q in (85, 75, 65, 55, 45):
            data_url, raw_size = _encode(img, q)
            if raw_size <= self.valves.image_max_bytes:
                return data_url

        # Pass 3: hard fallback 1536px
        img = _resize(img, 1536)
        data_url, raw_size = _encode(img, 60)
        if raw_size <= self.valves.image_max_bytes:
            return data_url

        raise ValueError(
            f"Image cannot be compressed below {self.valves.image_max_bytes / 1024 / 1024:.1f} MB limit."
        )

    def _extract_images_from_message(
        self, message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract and compress images from an Open WebUI message."""
        raw_images: List[Image.Image] = []

        def _add(b64_url: str):
            try:
                b64_data = b64_url.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_data)
                pil_img = Image.open(io.BytesIO(image_bytes))
                pil_img.load()
                raw_images.append(pil_img)
            except Exception as e:
                print(f"DEBUG: Error processing image: {e}")

        # 1. From 'files' field
        for f in message.get("files", []):
            if isinstance(f, dict):
                data = f.get("data", "")
                if isinstance(data, str) and data.startswith("data:image"):
                    _add(data)

        # 2. From content list (multimodal)
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        _add(url)

        # 3. From markdown embedded images in string content
        if isinstance(content, str):
            for url in re.findall(
                r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)", content
            ):
                _add(url)

        if not raw_images:
            return []

        # Deduplicate by pixel bytes
        seen = set()
        unique = []
        for img in raw_images:
            key = img.tobytes()
            if key not in seen:
                seen.add(key)
                unique.append(img)

        result = []
        for img in unique:
            b64_url = self._compress_image_to_b64(img)
            result.append({"pil": img, "base64_url": b64_url})
        return result

    # ---------------------------------------------------------------------
    # Prompt parsing
    # ---------------------------------------------------------------------
    def _detect_size_from_prompt(self, prompt: str, model_cfg: Dict) -> str:
        """Detect size keywords in prompt and return the size string."""
        prompt_lower = prompt.lower()
        patterns = {
            "4K": ["4k", "4096x2048", "4096*2048", "ultra hd", "uhd"],
            "3K": ["3k", "3072x1536", "3072*1536"],
            "2K": ["2k", "2048x1024", "2048*1024", "1440p"],
            "1K": ["1k", "1024x512", "1024*512"],
            "1024x1024": ["1024x1024", "1024*1024", "square 1024", "square"],
            "768x1344": ["768x1344", "768*1344", "portrait 768", "portrait"],
            "1344x768": ["1344x768", "1344*768", "landscape 1344", "landscape"],
            "1536x1024": ["1536x1024", "1536*1024"],
            "1024x1536": ["1024x1536", "1024*1536"],
            "512x512": ["512x512", "512*512"],
            "2048x2048": ["2048x2048", "2048*2048"],
        }
        for size, keywords in patterns.items():
            if any(k in prompt_lower for k in keywords):
                return size
        return model_cfg["default_size"]

    def _extract_seed_from_prompt(self, prompt: str) -> Optional[int]:
        """Extract 'seed:123' or 'seed=123' from prompt, returning seed or None."""
        m = re.search(
            r"(?:seed|random[ _-]?seed)\s*[:=]\s*(-?\d+)", prompt, re.IGNORECASE
        )
        if m:
            return int(m.group(1))
        return None

    def _extract_guidance_from_prompt(self, prompt: str) -> Optional[float]:
        """Extract 'guidance:7.5' or 'cfg:7' from prompt."""
        m = re.search(
            r"(?:guidance[ _-]?scale|guidance|cfg)\s*[:=]\s*(\d+(?:\.\d+)?)",
            prompt,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))
        return None

    def _detect_sequential_request(self, prompt: str, model_cfg: Dict) -> tuple:
        """Detect if user wants multiple images from text prompt.

        Returns (enabled: bool, count: int).
        """
        cap = model_cfg["capabilities"]
        if not cap["sequential"]:
            return False, 1

        prompt_lower = prompt.lower()
        m = re.search(
            r"(\d+)\s+(images?|pics?|pictures?|variations?|results?|versions?|shots?)",
            prompt_lower,
        )
        if m:
            count = int(m.group(1))
            cap_max = cap["max_images"]
            return True, max(1, min(count, cap_max))

        seq_keywords = [
            "different angles",
            "different variations",
            "multiple images",
            "several images",
            "generate 3",
            "create 4",
            "morning, noon and night",
        ]
        if any(k in prompt_lower for k in seq_keywords):
            return True, min(self.valves.default_max_images, cap["max_images"])
        return False, 1

    def _clean_prompt(self, prompt: str) -> str:
        """Remove control keywords (seed:, guidance:, etc.) from prompt."""
        prompt = re.sub(
            r"(?:seed|random[ _-]?seed)\s*[:=]\s*-?\d+\s*",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        prompt = re.sub(
            r"(?:guidance[ _-]?scale|guidance|cfg)\s*[:=]\s*\d+(?:\.\d+)?\s*",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        return prompt.strip()

    # ---------------------------------------------------------------------
    # BytePlus API caller
    # ---------------------------------------------------------------------
    async def _call_images_generations(
        self, payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Call BytePlus /images/generations endpoint with native async httpx."""
        url = f"{self.valves.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }
        print(f"DEBUG: BytePlus API call → {url}")
        print(f"DEBUG: Model → {payload.get('model')}")
        print(f"DEBUG: Payload keys → {sorted(payload.keys())}")

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
                    "revised_prompt": item.get("revised_prompt"),
                }
            )
        if not results:
            raise ValueError(f"No image data in response: {json.dumps(data, indent=2)}")
        return results

    # ---------------------------------------------------------------------
    # Payload builder
    # ---------------------------------------------------------------------
    def _build_payload(
        self,
        prompt: str,
        images: List[str],
        mode: str,
        size: str,
        seed: Optional[int],
        guidance: Optional[float],
        seq_count: int,
        model_cfg: Dict,
    ) -> Dict[str, Any]:
        """Build the API payload based on mode and model capabilities."""
        cap = model_cfg["capabilities"]
        payload: Dict[str, Any] = {
            "model": model_cfg["id"],
            "prompt": prompt,
            "size": size,
            "output_format": self.valves.output_format,
            "response_format": self.valves.response_format,
            "watermark": self.valves.watermark,
        }

        if seed is not None:
            payload["seed"] = seed

        if guidance is not None and cap["has_guidance"]:
            payload["guidance_scale"] = guidance

        # Sequential image generation — only for models that support it
        if cap["sequential"] and mode == "text2img":
            if seq_count > 1:
                payload["sequential_image_generation"] = "auto"
                payload["sequential_image_generation_options"] = {
                    "max_images": seq_count
                }
            else:
                payload["sequential_image_generation"] = "disabled"

        # Image inputs
        if mode == "img2img":
            payload["image"] = images[0]
            if cap["sequential"]:
                payload["sequential_image_generation"] = "disabled"
            # For non-sequential models (5.0 Pro), just pass the image — no seq field

        elif mode == "blend":
            max_refs = cap["max_ref_images"]
            payload["image"] = images[:max_refs]
            if cap["sequential"]:
                payload["sequential_image_generation"] = "disabled"
            # For non-sequential models (5.0 Pro), native multi-image array

        return payload

    # ---------------------------------------------------------------------
    # SSE formatting
    # ---------------------------------------------------------------------
    def _format_chunk(self, content: str, model: str) -> str:
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def _format_stop_chunk(self, model: str) -> str:
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # ---------------------------------------------------------------------
    # Main pipe entry point
    # ---------------------------------------------------------------------
    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: dict,
        __event_emitter__=None,
    ) -> AsyncGenerator[str, None]:
        # 1. Basic validation
        if not self._validate_config():
            msg = (
                "❌ **Error:** BytePlus API key is missing. "
                "Configure it in the Pipe Valves."
            )
            yield self._format_chunk(msg, "seedream-manifold")
            yield self._format_stop_chunk("seedream-manifold")
            return

        try:
            # 2. Resolve which model was selected from the manifold
            raw_model = body.get("model", "")
            model_id = self._strip_model_prefix(raw_model)
            model_cfg = self._get_model_config(model_id)
            cap = model_cfg["capabilities"]

            print(f"DEBUG: Manifold model → {model_cfg['id']} ({model_cfg['name']})")

            # 3. Extract prompt
            last_message = body["messages"][-1]
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
                yield self._format_chunk(
                    "❌ **Error:** No text prompt found.", model_cfg["id"]
                )
                yield self._format_stop_chunk(model_cfg["id"])
                return

            # 4. Extract control parameters from prompt
            seed = self._extract_seed_from_prompt(user_prompt)
            guidance = self._extract_guidance_from_prompt(user_prompt)
            # Only apply guidance if the model supports it
            if cap["has_guidance"]:
                if guidance is None:
                    guidance = model_cfg["default_guidance"]
            else:
                guidance = None

            # Clean prompt of control keywords
            clean_prompt = self._clean_prompt(user_prompt)

            # 5. Extract images from current message, fallback to previous
            current_images = self._extract_images_from_message(last_message)
            if not current_images:
                for msg in reversed(body["messages"][:-1]):
                    found = self._extract_images_from_message(msg)
                    if found:
                        current_images = found
                        break

            image_b64s = [img["base64_url"] for img in current_images]
            img_count = len(image_b64s)

            # 6. Determine mode
            if img_count == 0:
                mode = "text2img"
            elif img_count == 1:
                mode = "img2img"
            else:
                mode = "blend"

            # Validate mode against model capabilities — give helpful error
            if mode == "img2img" and not cap["img2img"]:
                yield self._format_chunk(
                    f"❌ **Error:** {model_cfg['name']} does not support image-to-image editing.\n"
                    f"Try a model with `img2img=yes` capability, or add one to your Valves models list.",
                    model_cfg["id"],
                )
                yield self._format_stop_chunk(model_cfg["id"])
                return

            if mode == "blend" and not cap["blend"]:
                yield self._format_chunk(
                    f"❌ **Error:** {model_cfg['name']} does not support multi-image blending.\n"
                    f"Try a model with `blend=yes` capability, or add one to your Valves models list.",
                    model_cfg["id"],
                )
                yield self._format_stop_chunk(model_cfg["id"])
                return

            # Clamp reference image count
            if mode in ("img2img", "blend") and img_count > cap["max_ref_images"]:
                image_b64s = image_b64s[: cap["max_ref_images"]]
                img_count = len(image_b64s)

            # 7. Detect size & sequential request
            size = (
                self._detect_size_from_prompt(clean_prompt, model_cfg)
                if self.valves.enable_size_detection
                else model_cfg["default_size"]
            )
            seq_enabled, seq_count = False, 1
            if mode == "text2img" and self.valves.enable_sequential_detection:
                seq_enabled, seq_count = self._detect_sequential_request(
                    clean_prompt, model_cfg
                )

            # 8. Build payload and call API
            payload = self._build_payload(
                clean_prompt,
                image_b64s,
                mode,
                size,
                seed,
                guidance,
                seq_count if seq_enabled else 1,
                model_cfg,
            )

            # Status emitter
            if __event_emitter__:
                if mode == "blend":
                    status = (
                        f"🎨 Blending {img_count} images at {size} "
                        f"using {model_cfg['name']}…"
                    )
                elif mode == "img2img":
                    status = f"✏️ Editing image at {size} using {model_cfg['name']}…"
                else:
                    n_text = f" → {seq_count} image(s)" if seq_enabled else ""
                    status = (
                        f"🖼️ Generating image at {size}{n_text} "
                        f"using {model_cfg['name']}…"
                    )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status, "done": False},
                    }
                )

            print(
                f"DEBUG: Mode={mode}, Images={img_count}, Size={size}, "
                f"Seq={seq_enabled}({seq_count}), Seed={seed}, Guidance={guidance}"
            )

            # Native async API call — no thread pool overhead, fully non-blocking
            results = await self._call_images_generations(payload)

            # 9. Build response
            lines = []
            if mode == "blend":
                lines.append(f"**🎨 Blended Result ({len(results)} output(s)):**\n")
            elif mode == "img2img":
                lines.append(f"**✏️ Edited Result ({len(results)} output(s)):**\n")
            else:
                lines.append(f"**🖼️ Generated Result ({len(results)} output(s)):**\n")

            for idx, res in enumerate(results, start=1):
                label = f"Image {idx}" if len(results) > 1 else "Result"
                url = res.get("url")
                if not url and res.get("b64_json"):
                    fmt = self.valves.output_format
                    url = f"data:image/{fmt};base64,{res['b64_json']}"
                if url:
                    lines.append(f"![{label}]({url})\n")

            # Metadata line
            meta_parts = [
                f"**Size:** `{size}`",
                f"**Model:** `{model_cfg['name']}`",
                f"**Outputs:** `{len(results)}`",
            ]
            if seed is not None:
                meta_parts.append(f"**Seed:** `{seed}`")
            if guidance is not None:
                meta_parts.append(f"**Guidance:** `{guidance}`")
            lines.append("\n" + " | ".join(meta_parts))
            lines.append(f"\n💡 _Tip: {random.choice(self.tooltips)}_")

            yield self._format_chunk("\n".join(lines), model_cfg["id"])

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "✅ Done", "done": True}}
                )

        except Exception as err:
            yield self._format_chunk(
                f"❌ **Error:**\n\n{str(err)}", "seedream-manifold"
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "❌ Error occurred",
                            "done": True,
                        },
                    }
                )
        finally:
            yield self._format_stop_chunk("seedream-manifold")

