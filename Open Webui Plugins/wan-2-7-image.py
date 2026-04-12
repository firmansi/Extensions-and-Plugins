"""
title: Qwen WAN Unified Image Generator & Editor
author: dfirmans (unified by assistant)
version: 5.0
description: >
  Unified pipe for Qwen WAN-2.7 image generation and editing in Open WebUI.
  - Text-to-image: uses wan2.7-image-pro (sync) for single image generation
  - Image-to-image / editing: uses wan2.7-image-pro (sync) when an image is uploaded
  - Image set generation: uses wan2.7-image-pro (sync) with sequential mode for multiple images
  Automatically detects whether an input image is present and routes accordingly.
  Size can be auto-detected from prompt keywords or overridden via Valves.
required_open_webui_version: 0.5.0
requirements: pillow, requests
"""

import base64
import io
import re
import time
import uuid
import json
import random
import asyncio
import requests
from typing import AsyncGenerator, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from fastapi import Request
from PIL import Image

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

WANModelType = Literal[
    "wan2.7-image-pro",  # Latest model supporting all features
]

ImageSizeType = Literal[
    "2K",  # New size option for WAN-2.7
    "1024*1024",
    "1440*810",
    "810*1440",
    "1440*1080",
    "1080*1440",
]


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------


class Pipe:
    """
    Unified Qwen WAN pipe.

    Routing logic:
      - If the conversation contains an uploaded image  → Image-to-image editing
        (wan2.6-image via async API, endpoint: image-generation/generation)
      - Otherwise                                       → Text-to-image generation
        (wan2.6-t2i via sync API, or other models via async API,
         endpoint: text2image/image-synthesis)
    """

    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description="Alibaba Cloud DashScope API key",
        )
        base_url: str = Field(
            default="https://dashscope-intl.aliyuncs.com/api/v1",
            description="DashScope API base URL (Singapore region)",
        )

        # --- Model settings ---
        model: str = Field(
            default="wan2.7-image-pro",
            description=(
                "Model for image generation and editing. "
                "'wan2.7-image-pro' supports all features including sequential generation."
            ),
        )

        # --- Size detection settings ---
        enable_size_detection: bool = Field(
            default=True,
            description=(
                "Auto-detect image size from prompt keywords "
                "(e.g. 'landscape', '16:9', 'portrait', 'square'). "
                "When disabled, the default size below is always used."
            ),
        )
        default_size: str = Field(
            default="2K",
            description=(
                "Default size for image generation when auto-detection is disabled "
                "or no keyword is matched. "
                "Options: 2K, 1024*1024 (1:1), 1440*810 (16:9), 810*1440 (9:16), "
                "1440*1080 (4:3), 1080*1440 (3:4)."
            ),
        )

        # --- Sequential generation settings ---
        enable_sequential: bool = Field(
            default=False,
            description=(
                "Enable sequential image generation for consistent character/object across multiple images. "
                "Useful for image sets with consistent features."
            ),
        )

        # --- Generation settings ---
        num_images: int = Field(
            default=1,
            description=(
                "Number of images to generate. Max 4 for sequential generation, 1 for regular generation."
            ),
        )

        enable_thinking_mode: bool = Field(
            default=True,
            description="Enable thinking mode for better image generation",
        )

        enable_watermark: bool = Field(
            default=False,
            description="Enable watermark on generated images",
        )

        # --- Shared settings ---
        enable_prompt_extend: bool = Field(
            default=True,
            description="Enable automatic prompt extension for better results",
        )

    # -----------------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------------

    def __init__(self):
        self.valves = self.Valves()
        self.tooltips = [
            "Try adjusting the style, e.g., 'in the style of Van Gogh'.",
            "How about generating the same scene from a different perspective?",
            "What about changing the time of day? Try 'at sunset' or 'at night'.",
            "Add a new element, e.g., 'add a small fox in the foreground'.",
            "You can also ask to remove elements you don't like.",
            "Change the color palette, e.g., 'make the colors warmer and more saturated'.",
            "Try changing the mood: 'make it more mysterious'.",
            "Try different aspect ratios: 16:9 for widescreen, 9:16 for vertical.",
            "For image sets, describe multiple scenes in one prompt to maintain consistency.",
        ]

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_config(self) -> bool:
        return bool(self.valves.api_key)

    # -----------------------------------------------------------------------
    # Image extraction helpers
    # -----------------------------------------------------------------------

    def _extract_images_from_message(self, message: Dict[str, Any]) -> List[str]:
        """
        Return a deduplicated list of data-URL strings (data:image/…;base64,…)
        found inside a single Open WebUI message dict.
        """
        images: List[str] = []

        # 1. 'files' field (Open WebUI file uploads)
        for file_item in message.get("files", []):
            if isinstance(file_item, dict):
                data = file_item.get("data", "")
                if data.startswith("data:image"):
                    images.append(data)

        # 2. content list with image_url items
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        images.append(url)

        # 3. Markdown-embedded base64 images inside string content
        if isinstance(content, str):
            for data_url in re.findall(r"\((data:image/[^;]+;base64,[^)]+)\)", content):
                images.append(data_url)

        # Deduplicate while preserving order
        return list(dict.fromkeys(images))

    def _find_images_in_conversation(self, messages: List[Dict]) -> List[str]:
        """Search messages in reverse order and return the first batch of images found."""
        for message in reversed(messages):
            imgs = self._extract_images_from_message(message)
            if imgs:
                return imgs
        return []

    # -----------------------------------------------------------------------
    # Size detection helpers
    # -----------------------------------------------------------------------

    def _detect_size(self, prompt: str) -> str:
        """
        Resolve the output size for image generation.

        Priority:
          1. If enable_size_detection is OFF  → return default_size valve
          2. Explicit size string in prompt   → matched value
          3. Orientation keywords             → inferred value
          4. Fallback                         → default_size valve
        """
        if not self.valves.enable_size_detection:
            print(
                f"DEBUG [size] detection disabled, using default: {self.valves.default_size}"
            )
            return self.valves.default_size

        p = prompt.lower()

        # Check for 2K size
        if "2k" in p or "2 k" in p:
            print(f"DEBUG [size] 2K match → 2K")
            return "2K"

        # Explicit size / aspect-ratio strings
        size_patterns = {
            "1440*810": ["1440x810", "1440*810", "16:9", "16/9", "sixteen by nine"],
            "810*1440": ["810x1440", "810*1440", "9:16", "9/16", "nine by sixteen"],
            "1440*1080": ["1440x1080", "1440*1080", "4:3", "4/3", "four by three"],
            "1080*1440": ["1080x1440", "1080*1440", "3:4", "3/4", "three by four"],
            "1024*1024": ["1024x1024", "1024*1024", "1:1", "square"],
        }
        for size, patterns in size_patterns.items():
            if any(pat in p for pat in patterns):
                print(f"DEBUG [size] explicit match → {size}")
                return size

        # Orientation keywords
        vertical_kw = [
            "vertical",
            "portrait",
            "tall",
            "mobile",
            "phone",
            "instagram story",
            "story",
            "tiktok",
            "reels",
        ]
        if any(w in p for w in vertical_kw):
            size = "1080*1440" if any(w in p for w in ["3:4", "3/4"]) else "810*1440"
            print(f"DEBUG [size] vertical keyword → {size}")
            return size

        widescreen_kw = [
            "widescreen",
            "landscape",
            "horizontal",
            "wide",
            "banner",
            "header",
            "desktop",
            "monitor",
        ]
        if any(w in p for w in widescreen_kw):
            size = "1440*1080" if any(w in p for w in ["4:3", "4/3"]) else "1440*810"
            print(f"DEBUG [size] widescreen keyword → {size}")
            return size

        square_kw = ["square", "1:1", "instagram post", "profile picture", "avatar"]
        if any(w in p for w in square_kw):
            print("DEBUG [size] square keyword → 1024*1024")
            return "1024*1024"

        # Fallback to valve default
        print(f"DEBUG [size] no match, using default: {self.valves.default_size}")
        return self.valves.default_size

    # -----------------------------------------------------------------------
    # Universal image generation: synchronous (wan2.7-image-pro)
    # -----------------------------------------------------------------------

    def _generate_image_sync(
        self,
        prompt: str,
        images: List[str] = [],
        size: str = "2K",
        n: int = 1,
        sequential: bool = False,
    ) -> List[str]:
        """Call the synchronous multimodal-generation endpoint for WAN-2.7 Image Pro."""
        url = f"{self.valves.base_url}/services/aigc/multimodal-generation/generation"

        # Build content array: text first, then images if any
        content_payload: List[Dict] = [{"text": prompt}]
        for img_data_url in images:
            content_payload.append({"image": img_data_url})

        payload = {
            "model": self.valves.model,
            "input": {"messages": [{"role": "user", "content": content_payload}]},
            "parameters": {
                "size": size,
                "n": n,
                "watermark": self.valves.enable_watermark,
                "thinking_mode": self.valves.enable_thinking_mode,
                "prompt_extend": self.valves.enable_prompt_extend,
            },
        }

        # Add sequential parameter if enabled
        if sequential:
            payload["parameters"]["enable_sequential"] = True

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.api_key}",
        }

        print(
            f"DEBUG [IMG-sync] POST {url}  size={size}  n={n}  sequential={sequential}"
        )
        # Increase timeout for synchronous call
        resp = requests.post(
            url, headers=headers, json=payload, timeout=300
        )  # 5 minutes timeout

        if resp.status_code != 200:
            raise Exception(
                f"Image generation sync HTTP {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        print(f"DEBUG [IMG-sync] Response: {data}")  # Debug response structure

        try:
            # Extract all image URLs from response
            images = []

            # Try format 1: choices → message → content
            for choice in data["output"]["choices"]:
                for content_item in choice["message"]["content"]:
                    if "image" in content_item:
                        images.append(content_item["image"])

            # If no images found in first format, try alternative format
            if not images:
                # Try format 2: results → url (for cases where API returns different structure)
                results = data["output"].get("results", [])
                for result in results:
                    if "url" in result:
                        images.append(result["url"])

            # If still no images found, print the response structure for debugging
            if not images:
                print(f"DEBUG: Response structure: {data}")

            return images
        except (KeyError, IndexError, TypeError) as e:
            print(f"DEBUG: Error parsing response: {e}")
            print(f"DEBUG: Full response: {data}")
            raise ValueError(f"Cannot parse image URL from sync response: {data}")

    # -----------------------------------------------------------------------
    # Response parsing helpers
    # -----------------------------------------------------------------------

    def _parse_image_urls(self, result_json: Dict) -> List[str]:
        """
        Parse multiple image URLs from response based on the actual API response structure.
        """
        output = result_json.get("output", {})

        # Shape 1: choices → message → content (for sync calls)
        images = []
        for choice in output.get("choices", []):
            for item in choice.get("message", {}).get("content", []):
                if "image" in item:
                    images.append(item["image"])

        # If no images found in the first format, try alternative format (for async results)
        if not images:
            results = output.get("results", [])
            for result in results:
                if "url" in result:
                    images.append(result["url"])

        return images

    # -----------------------------------------------------------------------
    # SSE formatting helpers
    # -----------------------------------------------------------------------

    def _chunk(self, content: str) -> str:
        return f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'unified', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': content}, 'finish_reason': None}]})}\n\n"

    def _stop_chunk(self) -> str:
        return f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'unified', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: dict,
        __event_emitter__=None,
    ) -> AsyncGenerator[str, None]:
        """
        Route the request to text-to-image or image-to-image depending on
        whether an input image is present in the conversation.
        """

        # --- Validate API key ---
        if not self._validate_config():
            yield self._chunk(
                "❌ Error: DashScope API key is missing. Please set it in Valves."
            )
            yield self._stop_chunk()
            return

        try:
            # ----------------------------------------------------------------
            # 1. Extract the latest user prompt
            # ----------------------------------------------------------------
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
                yield self._chunk("❌ Error: No text prompt found in the last message.")
                yield self._stop_chunk()
                return

            # ----------------------------------------------------------------
            # 2. Detect mode: image editing vs image set generation vs text-to-image
            # ----------------------------------------------------------------
            input_images = self._find_images_in_conversation(body["messages"])

            # Check if this is image editing (has input images) or sequential generation (has keywords suggesting multiple images)
            is_editing = len(input_images) > 0

            # Check for sequential generation keywords
            sequential_keywords = [
                "set",
                "sequence",
                "series",
                "multiple",
                "first image",
                "second image",
                "third image",
                "fourth image",
                "part 1",
                "part 2",
                "part 3",
                "part 4",
                "image 1",
                "image 2",
                "image 3",
                "image 4",
                "consistency",
                "consistent",
            ]
            is_sequential = self.valves.enable_sequential or any(
                keyword in prompt.lower() for keyword in sequential_keywords
            )

            # Determine number of images to generate
            n_images = min(self.valves.num_images, 4)  # Max 4 for sequential generation
            if (
                is_sequential
                and "first image" in prompt.lower()
                and "second image" in prompt.lower()
            ):
                # Count images mentioned in prompt
                image_mentions = sum(
                    1
                    for keyword in [
                        "first image",
                        "second image",
                        "third image",
                        "fourth image",
                        "image 1",
                        "image 2",
                        "image 3",
                        "image 4",
                        "part 1",
                        "part 2",
                        "part 3",
                        "part 4",
                    ]
                    if keyword in prompt.lower()
                )
                n_images = max(n_images, min(image_mentions, 4))

            if is_editing:
                mode = "editing"
            elif is_sequential:
                mode = "sequential"
                n_images = min(n_images, 4)  # Limit to 4 for sequential generation
            else:
                mode = "generation"
                n_images = 1  # Single image generation

            print(
                f"DEBUG: mode={mode}  prompt={prompt[:80]}  images={len(input_images)}  n={n_images}"
            )

            # ----------------------------------------------------------------
            # 3. Determine image size
            # ----------------------------------------------------------------
            size = self._detect_size(prompt)
            model_label = self.valves.model
            if mode == "editing":
                status_msg = f"✏️ Editing image ({size}) with {model_label}…"
            elif mode == "sequential":
                status_msg = (
                    f"🎬 Generating {n_images}-image set ({size}) with {model_label}…"
                )
            else:
                status_msg = f"🎨 Generating {size} image with {model_label}…"

            # ----------------------------------------------------------------
            # 4. Emit "working" status
            # ----------------------------------------------------------------
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": False},
                    }
                )

            # ----------------------------------------------------------------
            # 5. Actual API call
            # ----------------------------------------------------------------
            image_urls = []
            try:
                if mode == "editing":
                    # Image editing mode - pass input images
                    image_urls = self._generate_image_sync(
                        prompt=prompt,
                        images=input_images,
                        size=size,
                        n=n_images,
                        sequential=False,  # No sequential mode for editing
                    )
                elif mode == "sequential":
                    # Sequential generation mode
                    image_urls = self._generate_image_sync(
                        prompt=prompt,
                        images=[],  # No input images for sequential generation
                        size=size,
                        n=n_images,
                        sequential=True,
                    )
                else:
                    # Regular generation mode
                    image_urls = self._generate_image_sync(
                        prompt=prompt,
                        images=[],  # No input images for regular generation
                        size=size,
                        n=n_images,
                        sequential=False,
                    )
            except Exception as e:
                raise e

            # ----------------------------------------------------------------
            # 6. Emit completion status
            # ----------------------------------------------------------------
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"✅ Done! ({size})",
                            "done": True,
                        },
                    }
                )

            # ----------------------------------------------------------------
            # 7. Build and yield final response
            # ----------------------------------------------------------------
            op_text = (
                "edited"
                if mode == "editing"
                else ("generated set of" if mode == "sequential" else "generated")
            )
            tip = random.choice(self.tooltips)

            if len(image_urls) == 1:
                # Single image response
                response = (
                    f"**🖼️ Here is your {op_text} image:**\n\n"
                    f"![Result]({image_urls[0]})\n\n"
                    f"**Size:** `{size}` | **Model:** `{model_label}`\n\n"
                    f"💡 _Tip: {tip}_"
                )
            else:
                # Multiple images response
                images_md = "\n\n".join(
                    [f"![Result {i + 1}]({url})" for i, url in enumerate(image_urls)]
                )
                response = (
                    f"**🖼️ Here is your {op_text} image set:**\n\n"
                    f"{images_md}\n\n"
                    f"**Size:** `{size}` | **Count:** `{len(image_urls)}` | **Model:** `{model_label}`\n\n"
                    f"💡 _Tip: {tip}_"
                )
            yield self._chunk(response)

        except Exception as err:
            print(f"ERROR: {err}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "❌ Error occurred", "done": True},
                    }
                )
            yield self._chunk(f"❌ **Error:** {str(err)}")

        finally:
            yield self._stop_chunk()
