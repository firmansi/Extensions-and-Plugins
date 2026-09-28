"""
BytePlus SeeDream Image Generation & Editing (Async Native)
author: converted for BytePlus, optimized for OpenWebUI 0.5+
version: 4.3-async
description: >
  Single-model pipe for BytePlus SeeDream in Open WebUI.
  Supports text-to-image, image-to-image editing, and multi-image reference
  with smart size detection.
  FIXED: Uses httpx async native instead of OpenAI SDK to bypass FastAPI request timeouts.
  Set the model via the `model_name` valve (e.g. seedream-5-0-260128).
required_open_webui_version: 0.5.0
requirements: httpx, pillow
"""

import asyncio
import random
import re
from typing import AsyncGenerator, Dict, Any, List, Literal, Optional

import httpx
from fastapi import Request
from pydantic import BaseModel, Field

# ─────────────────────────── Type alias ─────────────────────────────

ImageSizeType = Literal["1K", "2K", "4K"]

# Models that support sequential_image_generation + stream
_SEQ_STREAM_MODELS = {
    "seedream-5-0-260128",
    "seedream-4-5-251128",
    "seedream-4-0-251128",
}

# Models that accept style / strength parameters
_STYLE_MODELS = {"seedream-5-0-260128"}


# ────────────────────────────── Pipe ────────────────────────────────


class Pipe:
    """
    Single-model OpenWebUI Function Pipe — BytePlus SeeDream.

    ASYNC NATIVE (httpx):
    * Uses httpx.AsyncClient for truly async HTTP calls
    * No thread pool executor needed - keeps event loop free
    * Respects custom timeout settings without SDK interference
    * Yields plain str -> appended as assistant text directly (OpenWebUI 0.5+)
    """

    # Valves

    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description="BytePlus ARK API key",
        )
        base_url: str = Field(
            default="https://ark.bytedance.net/api/v3",
            description="BytePlus ARK API base URL",
        )
        model_name: str = Field(
            default="seedream-5-0-260128",
            description=(
                "SeeDream model ID to use. "
                "Options: seedream-5-0-260128, seedream-4-5-251128, "
                "seedream-4-0-251128, seedream-3-0-t2i"
            ),
        )
        enable_size_detection: bool = Field(
            default=True,
            description="Auto-detect image size from prompt keywords (1K / 2K / 4K)",
        )
        watermark: bool = Field(
            default=False,
            description="Add watermark to generated images",
        )
        seed: int = Field(
            default=-1,
            description="Random seed (-1 = random each time)",
        )
        sequential_image_generation: str = Field(
            default="disabled",
            description="Batch generation mode: 'auto' or 'disabled' (SeeDream 4.x/5.x only)",
        )
        stream: bool = Field(
            default=False,
            description="Enable streaming output (SeeDream 4.0/4.5 only)",
        )
        guidance_scale: float = Field(
            default=2.5,
            description="Prompt adherence 1-10 (SeeDream 3.0 only)",
        )
        response_format: str = Field(
            default="url",
            description="Response format: 'url' or 'b64_json'",
        )
        style: str = Field(
            default="",
            description="Style tag (SeeDream 5.0 only, optional)",
        )
        strength: float = Field(
            default=1.0,
            description="Edit strength 0.0-2.0",
        )
        timeout: int = Field(
            default=300,
            description=(
                "Request timeout in seconds. Image generation can take 1-3 minutes. "
                "Increase if you still get timeout errors."
            ),
        )

    # Init

    def __init__(self):
        self.valves = self.Valves()
        self.tooltips = [
            "Try adjusting the style, e.g., 'in the style of Van Gogh'.",
            "How about generating the same image from a different perspective?",
            "What about changing the time of day? Try 'at sunset' or 'at night'.",
            "Add a new element, e.g., 'add a small fox in the foreground'.",
            "You can also remove elements you don't like.",
            "Change the color palette, e.g., 'make the colors warmer'.",
            "Change the mood - try 'make it more mysterious'.",
            "Try different resolutions: 2K for quality or 4K for maximum detail.",
            "Upload multiple images to combine or reference different elements!",
            "Edit your image by uploading it and describing the changes you want.",
        ]

    # Helpers

    def _validate_config(self) -> bool:
        return bool(self.valves.api_key)

    def _extract_images_from_message(self, message: Dict[str, Any]) -> List[str]:
        """Return deduplicated base64 data-URL images from a single message dict."""
        found: List[str] = []

        for f in message.get("files", []):
            if isinstance(f, dict) and str(f.get("data", "")).startswith("data:image"):
                found.append(f["data"])

        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        found.append(url)

        if isinstance(content, str):
            for url in re.findall(
                r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)", content
            ):
                found.append(url)

        return list(dict.fromkeys(found))

    def _detect_size_from_prompt(self, prompt: str) -> ImageSizeType:
        pl = prompt.lower()
        if any(w in pl for w in ["4k", "4096", "ultra high", "maximum quality"]):
            return "4K"
        if any(w in pl for w in ["2k", "2048", "high quality", "hd"]):
            return "2K"
        if any(
            w in pl
            for w in [
                "1k",
                "1024",
                "standard",
                "normal",
                "quick",
                "fast",
                "draft",
                "preview",
            ]
        ):
            return "1K"
        return "2K"

    def _extract_text_from_message(self, message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        return content or ""

    def _build_extra_body(self, reference_images: Optional[List[str]] = None) -> dict:
        model = self.valves.model_name
        extra: dict = {"watermark": self.valves.watermark}

        if reference_images:
            extra["image"] = (
                reference_images[0] if len(reference_images) == 1 else reference_images
            )

        if self.valves.seed != -1:
            extra["seed"] = self.valves.seed

        if model in _SEQ_STREAM_MODELS:
            extra["sequential_image_generation"] = (
                self.valves.sequential_image_generation
            )
            if self.valves.stream:
                extra["stream"] = True

        if "seedream-3-0" in model:
            extra["guidance_scale"] = self.valves.guidance_scale

        if model in _STYLE_MODELS:
            if self.valves.style:
                extra["style"] = self.valves.style
            if self.valves.strength:
                extra["strength"] = self.valves.strength

        return extra

    async def _call_api_async(
        self,
        prompt: str,
        size: ImageSizeType,
        reference_images: Optional[List[str]] = None,
    ) -> str:
        """
        Async HTTP call to BytePlus using httpx.
        Truly non-blocking - no run_in_executor needed.
        Respects self.valves.timeout without SDK interference.
        """
        extra_body = self._build_extra_body(reference_images)

        payload = {
            "model": self.valves.model_name,
            "prompt": prompt,
            "size": size,
            "response_format": self.valves.response_format,
        }
        payload.update(extra_body)

        print(
            f"[SeeDream] model={self.valves.model_name} size={size} "
            f"refs={len(reference_images) if reference_images else 0} "
            f"timeout={self.valves.timeout}s"
        )

        headers = {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }

        # BytePlus ARK uses /images/generations endpoint (OpenAI-compatible)
        endpoint = f"{self.valves.base_url}/images/generations"

        print(f"[SeeDream] Calling {endpoint}")

        try:
            async with httpx.AsyncClient(timeout=self.valves.timeout, verify=False) as client:
                resp = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                )
                print(f"[SeeDream] Response status: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            print(f"[SeeDream] HTTP {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"[SeeDream] Request failed: {e}")
            raise

        if self.valves.response_format == "url":
            return data["data"][0]["url"]
        return f"data:image/jpeg;base64,{data['data'][0]['b64_json']}"

    # Main pipe

    async def pipe(
        self,
        body: dict,
        __request__: Request = None,
        __user__: dict = None,
        __event_emitter__=None,
    ) -> AsyncGenerator[str, None]:
        """
        Entry point called by OpenWebUI for every user message.
        Yields plain strings - OpenWebUI 0.5+ appends them as assistant content.
        """

        # Config guard
        if not self._validate_config():
            yield "Error: BytePlus ARK API key is missing. Please set it in Valves."
            return

        # Extract messages
        messages: list = body.get("messages", [])
        if not messages:
            yield "Error: No messages in request body."
            return

        # Extract prompt
        user_prompt = self._extract_text_from_message(messages[-1]).strip()
        if not user_prompt:
            yield "Error: No text prompt found. Please type something."
            return

        # Extract reference images from most recent message that has any
        reference_images: List[str] = []
        for msg in reversed(messages):
            imgs = self._extract_images_from_message(msg)
            if imgs:
                reference_images = imgs
                break

        # Determine mode
        n_refs = len(reference_images)
        if n_refs == 0:
            mode, emoji, mode_label = "generate", "🎨", "generated"
        elif n_refs == 1:
            mode, emoji, mode_label = "edit", "✏️", "edited"
        else:
            mode, emoji, mode_label = (
                "multi-ref",
                "🖼️",
                f"generated from {n_refs} references",
            )

        # Detect size
        size: ImageSizeType = (
            self._detect_size_from_prompt(user_prompt)
            if self.valves.enable_size_detection
            else "2K"
        )

        # Status event
        if __event_emitter__:
            if mode == "generate":
                status_text = (
                    f"{emoji} Creating {size} image with {self.valves.model_name}..."
                )
            elif mode == "edit":
                status_text = (
                    f"{emoji} Editing image ({size}) with {self.valves.model_name}..."
                )
            else:
                status_text = f"{emoji} Compositing {n_refs} refs ({size}) with {self.valves.model_name}..."
            await __event_emitter__(
                {"type": "status", "data": {"description": status_text, "done": False}}
            )

        # Generate - fully async, no thread pool needed
        try:
            image_url = await self._call_api_async(
                user_prompt,
                size,
                reference_images or None,
            )
        except asyncio.TimeoutError:
            print(f"[SeeDream] Request timeout after {self.valves.timeout}s")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Request timeout", "done": True},
                    }
                )
            yield (
                f"BytePlus Request Timeout:\n\n"
                f"Generation took longer than {self.valves.timeout} seconds. "
                f"Try increasing the timeout in Valves or try a smaller image size (1K instead of 4K)."
            )
            return
        except Exception as err:
            print(f"[SeeDream] API error: {err}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "API error", "done": True},
                    }
                )
            yield f"BytePlus API Error:\n\n```\n{err}\n```"
            return

        # Build response
        meta_parts = [
            f"**Size:** `{size}`",
            f"**Model:** `{self.valves.model_name}`",
        ]
        if mode != "generate":
            meta_parts.append(f"**Mode:** `{mode}`")
        if self.valves.seed != -1:
            meta_parts.append(f"**Seed:** `{self.valves.seed}`")

        tip = random.choice(self.tooltips)

        yield (
            f"**{emoji} Here is your {mode_label} image:**\n\n"
            f"![Generated Image]({image_url})\n\n"
            + " | ".join(meta_parts)
            + f"\n\n_Tip: {tip}_"
        )

        # Done status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Image {mode_label} successfully ({size})",
                        "done": True,
                    },
                }
            )
