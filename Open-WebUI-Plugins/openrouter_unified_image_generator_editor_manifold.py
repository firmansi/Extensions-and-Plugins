"""
title: OpenRouter Unified Image Generator & Editor (Manifold)
author: adapted from dfirmans' WAN pipe
version: 3.1.0
description: >
  Unified Pipe for image generation and editing in Open WebUI via OpenRouter.
  Uses the dedicated OpenRouter Image API (`/api/v1/images`) for both
  text-to-image and image-to-image/reference workflows.
  This is a manifold pipe: set `models` to a comma-separated list of OpenRouter
  model IDs and each one will appear as a selectable model in Open WebUI.
required_open_webui_version: 0.11.4
requirements: httpx
"""

import re
import json
import time
import uuid
import random
from typing import AsyncGenerator, List, Dict, Any, Optional

import httpx
from pydantic import BaseModel, Field
from fastapi import Request


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
                "google/gemini-3.1-flash-image, "
                "google/gemini-3.1-flash-lite-image, "
                "google/gemini-2.5-flash-image, "
                "google/gemini-3-pro-image, "
                "openai/gpt-image-2, "
                "black-forest-labs/flux.2-pro, "
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
                "Fallback to legacy /chat/completions path if dedicated Image API fails."
            ),
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
            description="Default aspect ratio when auto-detection finds nothing.",
        )
        default_resolution: str = Field(
            default="1K",
            description="Default image resolution: 512, 1K, 2K, or 4K.",
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
            description="Output image format: png, jpeg, webp, or svg.",
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
            description="Stream partial images via SSE when the model supports it.",
        )

    def __init__(self):
        self.valves = self.Valves()
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
        "landscape": "16:9",
        "horizontal": "16:9",
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

    def _resolve_size_and_ratio(self, prompt: str) -> tuple:
        if not self.valves.enable_size_detection:
            return (
                self.valves.default_aspect_ratio,
                self.valves.default_resolution,
                self.valves.size,
            )
        aspect_ratio = (
            self._detect_aspect_ratio(prompt) or self.valves.default_aspect_ratio
        )
        resolution = self._detect_resolution(prompt) or self.valves.default_resolution
        size = self.valves.size
        return aspect_ratio, resolution, size

    # ------------------------------------------------------------------
    # OpenRouter API helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.api_key}",
            "HTTP-Referer": "https://openwebui.com",
            "X-Title": "Open WebUI Image Pipe",
        }

    def _build_provider_object(self) -> Optional[Dict[str, Any]]:
        provider = {}
        if self.valves.provider_only:
            provider["only"] = [
                s.strip() for s in self.valves.provider_only.split(",") if s.strip()
            ]
        if self.valves.provider_ignore:
            provider["ignore"] = [
                s.strip() for s in self.valves.provider_ignore.split(",") if s.strip()
            ]
        if self.valves.provider_order:
            provider["order"] = [
                s.strip() for s in self.valves.provider_order.split(",") if s.strip()
            ]
        if self.valves.provider_sort:
            provider["sort"] = self.valves.provider_sort.strip()
        provider["allow_fallbacks"] = self.valves.allow_fallbacks
        return provider if provider else None

    def _build_image_payload(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
        aspect_ratio: str,
        resolution: str,
        size: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
        }
        if self.valves.n and self.valves.n > 1:
            payload["n"] = self.valves.n
        if size:
            payload["size"] = size
        else:
            payload["aspect_ratio"] = aspect_ratio
            payload["resolution"] = resolution
        if self.valves.quality:
            payload["quality"] = self.valves.quality
        if self.valves.output_format:
            payload["output_format"] = self.valves.output_format
        if self.valves.output_compression is not None:
            payload["output_compression"] = self.valves.output_compression
        if self.valves.background:
            payload["background"] = self.valves.background
        if self.valves.seed is not None:
            payload["seed"] = self.valves.seed
        if self.valves.user:
            payload["user"] = self.valves.user
        if self.valves.enable_streaming:
            payload["stream"] = True
        if input_images:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": url}}
                for url in input_images[:16]
            ]
        provider = self._build_provider_object()
        if provider:
            payload["provider"] = provider
        return payload

    async def _generate_image_with_image_api(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
        aspect_ratio: str,
        resolution: str,
        size: Optional[str],
    ) -> List[str]:
        url = f"{self.valves.base_url}/images"
        payload = self._build_image_payload(
            model_id, prompt, input_images, aspect_ratio, resolution, size
        )
        print(
            f"DEBUG [Image API] POST {url}  model={model_id}  "
            f"ratio={aspect_ratio}  resolution={resolution}  size={size}  "
            f"input_images={len(input_images)}  n={self.valves.n}  "
            f"stream={payload.get('stream', False)}"
        )
        if payload.get("stream"):
            image_urls = await self._stream_image_api(url, payload)
            if image_urls:
                return image_urls
            print("DEBUG [Image API] Stream returned no image; falling back to buffered.")
        async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
        if resp.status_code != 200:
            raise Exception(f"OpenRouter HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        print(f"DEBUG [Image API] Response keys: {list(data.keys())}")
        image_urls = self._extract_images_from_response(data)
        if not image_urls:
            print(f"DEBUG [Image API] Full response: {data}")
            raise ValueError(
                "No images found in response. Check that the model supports /api/v1/images."
            )
        return image_urls

    @classmethod
    def _extract_images_from_response(
        cls, data: Dict[str, Any]
    ) -> List[str]:
        """Extract data-URLs from a buffered Image API response."""
        image_urls = []
        for entry in data.get("data", []):
            b64 = entry.get("b64_json") or entry.get("b64Json") or ""
            media_type = entry.get("media_type", "image/png")
            if b64:
                image_urls.append(f"data:{media_type};base64,{b64}")
            elif entry.get("url"):
                image_urls.append(entry["url"])
        return list(dict.fromkeys(image_urls))

    async def _stream_image_api(
        self, url: str, payload: Dict[str, Any]
    ) -> List[str]:
        """Parse the SSE stream emitted by /images when stream=true."""
        image_urls = []
        print("DEBUG [Image API] Streaming mode enabled")
        try:
            async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
                async with client.stream(
                    "POST", url, headers=self._headers(), json=payload
                ) as resp:
                    if resp.status_code != 200:
                        text = await resp.aread()
                        raise Exception(
                            f"OpenRouter HTTP {resp.status_code}: {text.decode(errors='replace')}"
                        )
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[len("data:"):].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        try:
                            event = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        event_type = event.get("type", "")
                        if event_type == "image_generation.completed":
                            b64 = event.get("b64_json") or event.get("b64Json") or ""
                            media_type = event.get("media_type", "image/png")
                            if b64:
                                image_urls.append(
                                    f"data:{media_type};base64,{b64}"
                                )
                        elif event_type == "error":
                            msg = (
                                event.get("error", {}).get("message", "")
                                if isinstance(event.get("error"), dict)
                                else str(event.get("error", ""))
                            )
                            raise Exception(f"Stream error: {msg}")
        except Exception as e:
            print(f"DEBUG [Image API] Stream failed: {e}")
            raise
        return image_urls

    def _build_legacy_payload(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
        aspect_ratio: str,
        resolution: str,
        size: Optional[str],
    ) -> Dict[str, Any]:
        if input_images:
            content = [{"type": "text", "text": prompt}]
            for img_data_url in input_images:
                content.append(
                    {"type": "image_url", "image_url": {"url": img_data_url}}
                )
        else:
            content = prompt
        messages = [{"role": "user", "content": content}]
        modalities = [
            m.strip() for m in self.valves.legacy_modalities.split(",") if m.strip()
        ]
        image_config = {"aspect_ratio": aspect_ratio}
        if size:
            image_config["size"] = size
        else:
            image_config["image_size"] = resolution
        if self.valves.quality:
            image_config["quality"] = self.valves.quality
        payload = {
            "model": model_id,
            "messages": messages,
            "modalities": modalities,
            "image_config": image_config,
            "stream": False,
        }
        provider = self._build_provider_object()
        if provider:
            payload["provider"] = provider
        return payload

    async def _generate_image_with_legacy_api(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
        aspect_ratio: str,
        resolution: str,
        size: Optional[str],
    ) -> List[str]:
        url = f"{self.valves.base_url}/chat/completions"
        payload = self._build_legacy_payload(
            model_id, prompt, input_images, aspect_ratio, resolution, size
        )
        print(
            f"DEBUG [Legacy API] POST {url}  model={model_id}  "
            f"ratio={aspect_ratio}  resolution={resolution}  size={size}  "
            f"input_images={len(input_images)}"
        )
        async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
        if resp.status_code != 200:
            raise Exception(f"OpenRouter HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        print(f"DEBUG [Legacy API] Response keys: {list(data.keys())}")
        image_urls = []
        try:
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                for img in message.get("images", []):
                    url_val = img.get("image_url", {}).get("url", "")
                    if url_val:
                        image_urls.append(url_val)
                if not image_urls:
                    msg_content = message.get("content", "")
                    if isinstance(msg_content, list):
                        for item in msg_content:
                            if item.get("type") == "image_url":
                                url_val = item.get("image_url", {}).get("url", "")
                                if url_val:
                                    image_urls.append(url_val)
        except Exception as e:
            print(f"DEBUG [Legacy API] Parse error: {e}")
            print(f"DEBUG [Legacy API] Full response: {data}")
            raise ValueError(f"Cannot parse image from response: {data}") from e
        if not image_urls:
            print(f"DEBUG [Legacy API] Full response: {data}")
            raise ValueError(
                "No images found in legacy response. Check legacy_modalities and model support."
            )
        return image_urls

    async def _generate_image(
        self,
        model_id: str,
        prompt: str,
        input_images: List[str],
        aspect_ratio: str,
        resolution: str,
        size: Optional[str],
    ) -> List[str]:
        errors = []
        try:
            return await self._generate_image_with_image_api(
                model_id, prompt, input_images, aspect_ratio, resolution, size
            )
        except Exception as e:
            errors.append(f"Image API error: {e}")
            if not self.valves.enable_legacy_chat_fallback:
                raise
        try:
            return await self._generate_image_with_legacy_api(
                model_id, prompt, input_images, aspect_ratio, resolution, size
            )
        except Exception as e:
            errors.append(f"Legacy API error: {e}")
            raise Exception(
                "Failed to generate image with both Image API and legacy API.\n"
                + "\n".join(errors)
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
        "gpt-image",
        "seedream",
        "recraft",
        "riverflow",
        "qwen-image",
        "muse",
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
                "❌ **Error:** OpenRouter API key is missing. "
                "Please set it in **Valves → api_key**.",
            )
            yield self._stop_chunk(model_id)
            return

        if not model_id:
            yield self._chunk(
                model_id, "❌ **Error:** Could not determine model ID from request."
            )
            yield self._stop_chunk(model_id)
            return

        try:
            last_msg = body["messages"][-1]
            content = last_msg.get("content", "")
            if isinstance(content, list):
                prompt = "".join(
                    item["text"] for item in content if item.get("type") == "text"
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
                        f"Image-to-image editing requires a model that accepts image uploads. "
                        f"Please switch to one of these:\n\n"
                        f"- `google/gemini-3.1-flash-image`\n"
                        f"- `google/gemini-2.5-flash-image`\n"
                        f"- `openai/gpt-image-2`\n"
                        f"- `bytedance-seed/seedream-5-0-pro`\n"
                        f"- `qwen/qwen-image-3`\n\n"
                        f"Or remove the uploaded image to use text-to-image generation.",
                    )
                    yield self._stop_chunk(model_id)
                    return
                elif support is None:
                    print(
                        f"WARNING: Model '{model_id}' is not in the known compatibility list. "
                        f"Proceeding with image input — API may return an error."
                    )

            aspect_ratio, resolution, size = self._resolve_size_and_ratio(prompt)

            if is_editing:
                status_msg = (
                    f"✏️ Editing image with `{model_id}` — "
                    f"ratio: **{aspect_ratio}** | resolution: **{resolution}**…"
                )
            else:
                status_msg = (
                    f"🎨 Generating image with `{model_id}` — "
                    f"ratio: **{aspect_ratio}** | resolution: **{resolution}**…"
                )

            print(
                f"DEBUG: model={model_id}  prompt={prompt[:80]}  "
                f"ratio={aspect_ratio}  resolution={resolution}  size={size}  images={len(input_images)}"
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": False},
                    }
                )

            image_urls = await self._generate_image(
                model_id, prompt, input_images, aspect_ratio, resolution, size
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"✅ Done! ({aspect_ratio} · {resolution})",
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
                f"**Ratio:** `{aspect_ratio}` | **Resolution:** `{resolution}` | "
                f"**Model:** `{model_id}`\n\n"
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

