"""
title: BytePlus SeeDream Image Generation & Editing
author: converted for BytePlus
version: 3.0
description: Generates and edits images with BytePlus SeeDream 5.0 API. Supports text-to-image, image-to-image editing, and multi-image reference with smart size detection.
required_open_webui_version: 0.5.0
requirements: openai, pillow
"""

import base64
import io
import re
import time
import uuid
import json
import random
from typing import AsyncGenerator, List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field
from fastapi import Request
from PIL import Image
from openai import OpenAI

# Image size options based on BytePlus API documentation
ImageSizeType = Literal[
    "1K",  # 1024x1024
    "2K",  # 2048x2048
    "4K",  # 4096x4096
]

# BytePlus SeeDream models
ModelType = Literal[
    "seedream-5-0-260128",
    "seedream-4-5-251128",
    "seedream-4-0-251128",
    "seedream-3-0-t2i",
]

# Sequential generation mode
SequentialMode = Literal[
    "auto",
    "disabled",
]


class Pipe:
    """Implements the Pipe class for Open WebUI with BytePlus SeeDream image generation and editing."""

    class Valves(BaseModel):
        api_key: str = Field(default="", description="Your BytePlus ARK API key")
        base_url: str = Field(
            default="https://ark.bytedance.net/api/v3",
            description="BytePlus ARK API base URL",
        )
        model_name: str = Field(
            default="seedream-5-0-260128",
            description="The SeeDream model name for image generation",
        )
        enable_size_detection: bool = Field(
            default=True,
            description="Enable automatic size detection from prompt keywords",
        )
        watermark: bool = Field(
            default=False,
            description="Add watermark to generated images",
        )
        seed: int = Field(
            default=-1,
            description="Random seed for image generation (-1 for random)",
        )
        sequential_image_generation: str = Field(
            default="disabled",
            description="Batch generation mode: 'auto' or 'disabled'",
        )
        stream: bool = Field(
            default=False,
            description="Enable streaming output mode (for seedream-4.5 and 4.0)",
        )
        guidance_scale: float = Field(
            default=2.5,
            description="Control adherence to prompt (1-10, only for seedream-3.0)",
        )
        response_format: str = Field(
            default="url",
            description="Response format: 'url' or 'b64_json'",
        )
        style: str = Field(
            default="",
            description="Style parameter for SeeDream 5 (optional)",
        )
        strength: float = Field(
            default=1.0,
            description="Strength parameter for image editing (0.0-2.0)",
        )

    def __init__(self):
        self.valves = self.Valves()
        # List of random tooltips for a dynamic user experience
        self.tooltips = [
            "Try adjusting the style of the image, e.g., 'in the style of Van Gogh'.",
            "How about generating the same image from a different perspective?",
            "What about changing the time of day? Try 'at sunset' or 'at night'.",
            "Add a new element, for example, 'add a small fox in the foreground'.",
            "You can also remove elements that you don't like.",
            "Change the color palette, e.g., 'make the colors warmer and more saturated'.",
            "Let's change the mood of the image, try 'make it more mysterious'.",
            "Try different resolutions like 2K for higher quality or 4K for maximum detail.",
            "Upload multiple images to combine or reference different elements!",
            "Edit your image by uploading it and describing the changes you want.",
        ]

    def _validate_config(self) -> bool:
        """Validate that API key is configured."""
        if not self.valves.api_key:
            return False
        return True

    def _extract_images_from_message(self, message: Dict[str, Any]) -> List[str]:
        """
        Extract images from message content and convert to base64 data URLs.
        Returns list of base64 data URLs.
        """
        image_data_urls = []

        # Extract from 'files' field
        if "files" in message and isinstance(message["files"], list):
            for file_item in message.get("files", []):
                if isinstance(file_item, dict) and file_item.get("data", "").startswith(
                    "data:image"
                ):
                    try:
                        image_data_urls.append(file_item["data"])
                    except Exception as e:
                        print(f"DEBUG: Error processing 'files' image: {e}")

        # Extract from content with image_url
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        try:
                            image_data_urls.append(url)
                        except Exception as e:
                            print(f"DEBUG: Error processing 'image_url' image: {e}")

        # Extract from markdown images in string content
        if isinstance(content, str):
            markdown_images = re.findall(
                r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)", content
            )
            for data_url in markdown_images:
                try:
                    image_data_urls.append(data_url)
                except Exception as e:
                    print(f"DEBUG: Error processing Markdown image: {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img_url in image_data_urls:
            if img_url not in seen:
                unique_images.append(img_url)
                seen.add(img_url)

        return unique_images

    def _detect_size_from_prompt(self, prompt: str) -> ImageSizeType:
        """Detect desired image size from prompt keywords."""
        prompt_lower = prompt.lower()

        # Check for explicit size mentions first
        if any(
            word in prompt_lower
            for word in ["4k", "4096", "ultra high", "maximum quality"]
        ):
            return "4K"

        if any(word in prompt_lower for word in ["2k", "2048", "high quality", "hd"]):
            return "2K"

        if any(word in prompt_lower for word in ["1k", "1024", "standard", "normal"]):
            return "1K"

        # Check for quality keywords
        quality_keywords = {
            "4K": ["ultra", "maximum", "highest", "best quality", "ultra hd"],
            "2K": ["high", "hd", "quality", "detailed"],
            "1K": ["quick", "fast", "draft", "preview"],
        }

        for size, keywords in quality_keywords.items():
            if any(word in prompt_lower for word in keywords):
                return size

        # Default fallback to 2K
        return "2K"

    def _generate_image_with_byteplus(
        self,
        prompt: str,
        size: ImageSizeType,
        reference_images: Optional[List[str]] = None,
    ) -> str:
        """
        Generate image using BytePlus SeeDream API.

        :param prompt: Text prompt for image generation
        :param size: Image size
        :param reference_images: List of base64 image data URLs for image-to-image or multi-image reference
        :return: URL of the generated image or base64 data
        """
        try:
            # Initialize OpenAI client with BytePlus endpoint
            client = OpenAI(
                base_url=self.valves.base_url,
                api_key=self.valves.api_key,
            )

            # Prepare extra_body parameters
            extra_body = {
                "watermark": self.valves.watermark,
            }

            # Add reference images if provided
            if reference_images:
                if len(reference_images) == 1:
                    # Single image reference (image-to-image editing)
                    extra_body["image"] = reference_images[0]
                    print(f"DEBUG: Using single image reference for editing")
                else:
                    # Multiple images reference - SeeDream 5 uses list format for image replacement
                    # For multiple reference images, use the "image" parameter as a list
                    extra_body["image"] = reference_images
                    print(
                        f"DEBUG: Using {len(reference_images)} images as reference for image replacement"
                    )

            # Add seed if not random
            if self.valves.seed != -1:
                extra_body["seed"] = self.valves.seed

            # Add sequential generation for seedream-4.5, seedream-4.0, and seedream-5.0
            if self.valves.model_name in [
                "seedream-4-5-251128",
                "seedream-4-0-251128",
                "seedream-5-0-260128",
            ]:
                extra_body["sequential_image_generation"] = (
                    self.valves.sequential_image_generation
                )

                # Add streaming if enabled
                if self.valves.stream:
                    extra_body["stream"] = True

            # Add guidance_scale for seedream-3.0
            if "seedream-3-0" in self.valves.model_name:
                extra_body["guidance_scale"] = self.valves.guidance_scale

            # Add SeeDream 5 specific parameters if needed
            if self.valves.model_name == "seedream-5-0-260128":
                # SeeDream 5 might have additional parameters
                if hasattr(self.valves, "style") and self.valves.style:
                    extra_body["style"] = self.valves.style
                if hasattr(self.valves, "strength") and self.valves.strength:
                    extra_body["strength"] = self.valves.strength

                # Handle multiple reference images for SeeDream 5 image-to-image functionality
                if reference_images and len(reference_images) > 1:
                    # For multiple reference images, use the "image" parameter as a list
                    extra_body["image"] = reference_images

            print(
                f"DEBUG: Generating with model: {self.valves.model_name}, size: {size}"
            )
            print(
                f"DEBUG: Extra parameters: {json.dumps({k: v if k != 'image' else f'[{len(v)} images]' if isinstance(v, list) else '[1 image]' for k, v in extra_body.items()}, indent=2)}"
            )

            # Generate image
            images_response = client.images.generate(
                model=self.valves.model_name,
                prompt=prompt,
                size=size,
                response_format=self.valves.response_format,
                extra_body=extra_body,
            )

            # Return URL or base64 based on response format
            if self.valves.response_format == "url":
                return images_response.data[0].url
            else:
                # For b64_json format
                return f"data:image/jpeg;base64,{images_response.data[0].b64_json}"

        except Exception as e:
            raise Exception(f"BytePlus API error: {str(e)}")

    def _generate_image_from_references(
        self,
        prompt: str,
        size: ImageSizeType,
        reference_images: List[str],
    ) -> str:
        """
        Generate image using multiple reference images with specific prompt for image replacement.
        This method implements the functionality from the example code you provided.

        :param prompt: Text prompt for image generation
        :param size: Image size
        :param reference_images: List of base64 image data URLs or image URLs for reference
        :return: URL of the generated image or base64 data
        """
        try:
            # Initialize OpenAI client with BytePlus endpoint (using the configured base URL)
            client = OpenAI(
                base_url=self.valves.base_url,  # Use the configured base URL
                api_key=self.valves.api_key,
            )

            # Prepare the extra_body with multiple reference images
            # Handle both base64 encoded images and URLs
            processed_images = []
            for img in reference_images:
                if img.startswith("data:image"):
                    # If it's a base64 data URL, it can be passed directly
                    processed_images.append(img)
                else:
                    # If it's already a URL, pass it as is
                    processed_images.append(img)

            extra_body = {
                "image": processed_images,  # Pass the list of reference images
                "watermark": self.valves.watermark,  # Use configured watermark setting
                "sequential_image_generation": "disabled",  # Disable sequential generation
            }

            # Add seed if not random
            if self.valves.seed != -1:
                extra_body["seed"] = self.valves.seed

            print(
                f"DEBUG: Generating image from {len(processed_images)} reference images with prompt: '{prompt}'"
            )
            print(f"DEBUG: Reference images: {processed_images}")

            # Generate image using multiple reference images
            images_response = client.images.generate(
                model=self.valves.model_name,  # Use the configured model name
                prompt=prompt,
                size=size,
                response_format=self.valves.response_format,  # Use configured response format
                extra_body=extra_body,
            )

            # Return URL or base64 based on response format
            if self.valves.response_format == "url":
                return images_response.data[0].url
            else:
                # For b64_json format
                return f"data:image/jpeg;base64,{images_response.data[0].b64_json}"

        except Exception as e:
            raise Exception(
                f"BytePlus API error in reference-based generation: {str(e)}"
            )

            # Prepare the extra_body with multiple reference images
            # Handle both base64 encoded images and URLs
            processed_images = []
            for img in reference_images:
                if img.startswith("data:image"):
                    # If it's a base64 data URL, it can be passed directly
                    processed_images.append(img)
                else:
                    # If it's already a URL, pass it as is
                    processed_images.append(img)

            extra_body = {
                "image": processed_images,  # Pass the list of reference images
                "watermark": self.valves.watermark,  # Use configured watermark setting
                "sequential_image_generation": "disabled",  # Disable sequential generation
            }

            # Add seed if not random
            if self.valves.seed != -1:
                extra_body["seed"] = self.valves.seed

            print(
                f"DEBUG: Generating image from {len(processed_images)} reference images with prompt: '{prompt}'"
            )
            print(f"DEBUG: Reference images: {processed_images}")

            # Generate image using multiple reference images
            images_response = client.images.generate(
                model=self.valves.model_name,  # Use the configured model name
                prompt=prompt,
                size=size,
                response_format=self.valves.response_format,  # Use configured response format
                extra_body=extra_body,
            )

            # Return URL or base64 based on response format
            if self.valves.response_format == "url":
                return images_response.data[0].url
            else:
                # For b64_json format
                return f"data:image/jpeg;base64,{images_response.data[0].b64_json}"

        except Exception as e:
            raise Exception(
                f"BytePlus API error in reference-based generation: {str(e)}"
            )

    def _format_chunk(self, content: str) -> str:
        """Format content as SSE chunk."""
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
        """Format stop chunk."""
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
        """Main pipe method for handling image generation requests."""

        if not self._validate_config():
            yield self._format_chunk(
                "❌ Error: BytePlus ARK API key is missing or invalid."
            )
            yield self._format_stop_chunk()
            return

        try:
            # Extract user prompt
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
                yield self._format_chunk("❌ Error: No text prompt found.")
                yield self._format_stop_chunk()
                return

            # Check for input images (for editing mode or multi-image reference)
            reference_images = []
            for message in reversed(body["messages"]):
                extracted = self._extract_images_from_message(message)
                if extracted:
                    reference_images.extend(extracted)
                    break  # Only get images from the most recent message with images

            # Determine operation type
            if len(reference_images) == 0:
                operation_type = "generate"
                operation_emoji = "🎨"
            elif len(reference_images) == 1:
                operation_type = "edit"
                operation_emoji = "✏️"
            else:
                operation_type = "multi-reference"
                operation_emoji = "🖼️"

            # Detect image size from prompt
            if self.valves.enable_size_detection:
                image_size = self._detect_size_from_prompt(user_prompt)
                print(f"DEBUG: Detected size from prompt: {image_size}")
            else:
                image_size = "2K"  # Default 2K

            # Emit status
            if __event_emitter__:
                if operation_type == "generate":
                    status_msg = f"{operation_emoji} Creating {image_size} image with {self.valves.model_name}..."
                elif operation_type == "edit":
                    status_msg = f"{operation_emoji} Editing image ({image_size}) with {self.valves.model_name}..."
                else:
                    status_msg = f"{operation_emoji} Generating image from {len(reference_images)} references ({image_size}) with {self.valves.model_name}..."

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": False},
                    }
                )

            # Generate image
            print(
                f"DEBUG: Operation type: {operation_type}, Model: {self.valves.model_name}, Size: {image_size}"
            )

            # Check if we have multiple reference images and should use the reference-based generation
            if len(reference_images) >= 2:
                # Use the new reference-based generation method for multiple images
                image_url = self._generate_image_from_references(
                    user_prompt, image_size, reference_images
                )
            else:
                # Use the original generation method for single image or text-only generation
                image_url = self._generate_image_with_byteplus(
                    user_prompt,
                    image_size,
                    reference_images if reference_images else None,
                )

            # Format response based on operation type
            if operation_type == "generate":
                operation_text = "generated"
            elif operation_type == "edit":
                operation_text = "edited"
            else:
                operation_text = f"generated from {len(reference_images)} references"

            response_parts = [
                f"**{operation_emoji} Here is your {operation_text} image:**\n\n",
                f"![Generated Image]({image_url})\n\n",
                f"**Size:** `{image_size}` | **Model:** `{self.valves.model_name}`",
            ]

            # Add operation type info
            if operation_type != "generate":
                response_parts.append(f" | **Mode:** `{operation_type}`")

            # Add seed info if not random
            if self.valves.seed != -1:
                response_parts.append(f" | **Seed:** `{self.valves.seed}`")

            response_parts.append("\n")

            # Add random tip
            random_tip = random.choice(self.tooltips)
            response_parts.append(f"\n💡 _Tip: {random_tip}_")

            final_content = "".join(response_parts)
            yield self._format_chunk(final_content)

            # Emit completion status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"✅ Image {operation_text} successfully ({image_size})",
                            "done": True,
                        },
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
