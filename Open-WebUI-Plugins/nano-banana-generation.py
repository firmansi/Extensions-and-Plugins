"""
title: Gemini 2.5 Flash Image Generation & Editing
author: Dimitri Senhupen
version: 0.6.1
description: Generates and edits images with Gemini 2.5 Flash Image. Supports text-to-image, iterative image editing, and provides random tips.
required_open_webui_version: 0.5.0
requirements: google-genai, pillow, requests
"""
import base64
import io
import re
import time
import uuid
import json
import random
from typing import AsyncGenerator, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import Request
from PIL import Image

# Google Gemini API Imports
from google import genai
from google.genai import types

# Open WebUI Internal Imports
from open_webui.models.users import Users

class Pipe:
    """Implements the Pipe class for Open WebUI with image generation and editing."""

    class Valves(BaseModel):
        api_key: str = Field(
            default="", description="Your Google AI API key for Gemini"
        )
        model_name: str = Field(
            default="gemini-2.5-flash-image-preview",
            description="The Gemini model name for image generation",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.client = None
        # List of random tooltips for a dynamic user experience
        self.tooltips = [
            "Try adjusting the style of the image, e.g., 'in the style of Van Gogh'.",
            "How about generating the same image from a different perspective?",
            "Okay, that looks great! You can also try to select a new crop or focus area.",
            "What about changing the time of day? Try 'at sunset' or 'at night'.",
            "Add a new element, for example, 'add a small fox in the foreground'.",
            "You can also remove elements that you don't like.",
            "Change the color palette, e.g., 'make the colors warmer and more saturated'.",
            "Let's change the mood of the image, try 'make it more mysterious'.",
        ]

    def _initialize_client(self) -> bool:
        if not self.valves.api_key:
            return False
        try:
            self.client = genai.Client(api_key=self.valves.api_key)
            return True
        except Exception:
            return False

    def _extract_images_from_message(
        self, message: Dict[str, Any]
    ) -> List[Image.Image]:
        images = []
        if "files" in message and isinstance(message["files"], list):
            for file_item in message.get("files", []):
                if isinstance(file_item, dict) and file_item.get("data", "").startswith(
                    "data:image"
                ):
                    try:
                        b64_data = file_item["data"].split(",")[1]
                        image_bytes = base64.b64decode(b64_data)
                        images.append(Image.open(io.BytesIO(image_bytes)))
                    except Exception as e:
                        print(f"DEBUG: Error processing 'files' image: {e}")
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        try:
                            b64_data = url.split(",")[1]
                            image_bytes = base64.b64decode(b64_data)
                            images.append(Image.open(io.BytesIO(image_bytes)))
                        except Exception as e:
                            print(
                                f"DEBUG: Error processing 'image_url' image: {e}"
                            )
        if isinstance(content, str):
            markdown_images = re.findall(
                r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)", content
            )
            for data_url in markdown_images:
                try:
                    b64_data = data_url.split(",")[1]
                    image_bytes = base64.b64decode(b64_data)
                    images.append(Image.open(io.BytesIO(image_bytes)))
                except Exception as e:
                    print(f"DEBUG: Error processing Markdown image: {e}")
        if images:
            unique_images = []
            seen_images = set()
            for img in images:
                img_bytes = img.tobytes()
                if img_bytes not in seen_images:
                    unique_images.append(img)
                    seen_images.add(img_bytes)
            return unique_images
        return images

    def _prepare_content_for_gemini(
        self, prompt: str, images: List[Image.Image]
    ) -> List[types.Content]:
        parts = []
        for image in images:
            img_byte_arr = io.BytesIO()
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(img_byte_arr, format="JPEG")
            parts.append(
                types.Part.from_bytes(
                    data=img_byte_arr.getvalue(), mime_type="image/jpeg"
                )
            )
        parts.append(types.Part.from_text(text=prompt))
        return [types.Content(role="user", parts=parts)]

    def _format_chunk(self, content: str) -> str:
        chunk_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.valves.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(chunk_data)}\n\n"

    def _format_stop_chunk(self) -> str:
        stop_chunk_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.valves.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return f"data: {json.dumps(stop_chunk_data)}\n\n"

    async def pipe(
        self, body: dict, __request__: Request, __user__: dict, __event_emitter__=None
    ) -> AsyncGenerator[str, None]:
        if not self._initialize_client():
            yield self._format_chunk("Error: API key is missing or invalid.")
            yield self._format_stop_chunk()
            return
        try:
            last_message = body["messages"][-1]
            user_prompt = (
                "".join(
                    [
                        item["text"]
                        for item in last_message["content"]
                        if item["type"] == "text"
                    ]
                )
                if isinstance(last_message.get("content"), list)
                else last_message.get("content", "")
            )
            if not user_prompt.strip():
                yield self._format_chunk("Error: No text prompt found.")
                yield self._format_stop_chunk()
                return
            input_images = []
            for message in reversed(body["messages"]):
                extracted = self._extract_images_from_message(message)
                if extracted:
                    input_images.extend(extracted)
                    break
            operation_type = "edit" if input_images else "generate"
            if __event_emitter__:
                status_msg = (
                    "🎨 Creating image..."
                    if operation_type == "generate"
                    else "✏️ Editing image..."
                )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": False},
                    }
                )
            contents = self._prepare_content_for_gemini(user_prompt, input_images)
            generate_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
            response_parts = []
            image_count = 0
            for chunk in self.client.models.generate_content_stream(
                model=self.valves.model_name, contents=contents, config=generate_config
            ):
                if hasattr(chunk, "text") and chunk.text:
                    response_parts.append(chunk.text)
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    for part in chunk.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.data:
                            image_count += 1
                            mime_type = part.inline_data.mime_type or "image/png"
                            b64_data = base64.b64encode(part.inline_data.data).decode(
                                "utf-8"
                            )
                            operation_text = (
                                "generated"
                                if operation_type == "generate"
                                else "edited"
                            )
                            image_markdown = f"\n\n**🖼️ Here is your {operation_text} image:**\n\n![Image-{image_count}](data:{mime_type};base64,{b64_data})\n"
                            response_parts.append(image_markdown)
            if not response_parts:
                response_parts.append(
                    "⚠️ No output was generated. Please try rephrasing your request."
                )
            
            if image_count > 0:
                random_tip = random.choice(self.tooltips)
                response_parts.append(f"\n\n💡 _Tip: {random_tip}_")

            final_content = "".join(response_parts)
            yield self._format_chunk(final_content)
            
            if __event_emitter__:
                final_status = f"✅ Done - {image_count} image(s) processed"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": final_status, "done": True},
                    }
                )
        except Exception as err:
            error_message = f"An unexpected error occurred: {str(err)}"
            yield self._format_chunk(f"❌ **Error:**\n\n{error_message}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "❌ An error occurred", "done": True},
                    }
                )
        finally:
            yield self._format_stop_chunk()