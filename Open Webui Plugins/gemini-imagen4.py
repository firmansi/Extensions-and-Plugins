"""
title: Gemini-Imagen-4-Fast
description: Functions for generating images using Google's Imagen 4 Fast model via Gemini AI
author: Mike
version: 1.0.0
"""
import base64
import json
import logging
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Awaitable, Literal
import httpx
from fastapi import Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

class Pipe:
    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(default="", description="Your Google Gemini API key")
        API_BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com/v1beta/models/imagegeneration@004-fast:generateContent",
            description="Gemini API base URL for Imagen 4 Fast"
        )
        IMAGE_COUNT: int = Field(
            default=1, description="Number of images to generate (1-4)", ge=1, le=4
        )
        ASPECT_RATIO: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = Field(
            default="1:1", description="Aspect ratio of the generated image"
        )
        REQUEST_TIMEOUT: int = Field(
            default=600, description="Request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.emitter: Optional[Callable[[dict], Awaitable[None]]] = None

    async def emit_status(
        self, message: str, done: bool = False, show_in_chat: bool = False
    ):
        """Emit status updates to the client."""
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )
        # Only return a message for the chat if show_in_chat is True
        if show_in_chat:
            if done:
                return f"**✅ {message}**\n\n"
            else:
                return f"**⏳ {message}**\n\n"
        return ""

    async def pipes(self) -> List[dict]:
        return [{"id": "imagen-4.0-fast-generate-001", "name": "Gemini Imagen 4 Fast"}]

    async def pipe(
        self,
        body: dict,
        user: Optional[dict] = None,
        request: Optional[Request] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse:
        self.emitter = __event_emitter__

        async def stream_response():
            try:
                model = body.get("model", "imagen-4.0-fast-generate-001")
                messages = body.get("messages", [])
                is_stream = body.get("stream", False)
                
                # Extract prompt from messages
                prompt = self._extract_prompt(messages)
                
                if not self.valves.GEMINI_API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: Gemini API key not provided. Please set the GEMINI_API_KEY valve.",
                    )
                    return

                await self.emit_status("🖼️ Preparing image generation...")
                
                # Determine dimensions based on aspect ratio
                width, height = self._get_dimensions(self.valves.ASPECT_RATIO)
                
                # Prepare request payload
                request_body = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "generation_config": {
                        "temperature": 0.4,
                        "top_p": 1,
                        "top_k": 32,
                        "max_output_tokens": 8192,
                        "candidate_count": self.valves.IMAGE_COUNT,
                        "size": {"width": width, "height": height},
                    },
                    "safety_settings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        }
                    ],
                    "tools": [
                        {
                            "function_declarations": [
                                {
                                    "name": "generate_image",
                                    "description": "Generate an image based on a prompt",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "prompt": {
                                                "type": "string",
                                                "description": "The prompt to generate an image for"
                                            },
                                            "size": {
                                                "type": "object",
                                                "properties": {
                                                    "width": {
                                                        "type": "integer"
                                                    },
                                                    "height": {
                                                        "type": "integer"
                                                    }
                                                }
                                            }
                                        },
                                        "required": ["prompt", "size"]
                                    }
                                }
                            ]
                        }
                    ]
                }

                # Make the request to Gemini API
                await self.emit_status("🔄 Generating image... This may take a moment.")
                
                async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                    response = await client.post(
                        f"{self.valves.API_BASE_URL}?key={self.valves.GEMINI_API_KEY}",
                        json=request_body,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code != 200:
                        error_message = f"Error from Gemini API: {response.status_code} - {response.text}"
                        error_status = await self.emit_status(
                            f"❌ An error occurred", True, True
                        )
                        yield self._format_data(
                            is_stream=is_stream,
                            content=f"{error_status}{error_message}",
                        )
                        return
                        
                    response_data = response.json()
                    
                    # Process images
                    await self.emit_status("✅ Image processing complete!", True)
                    
                    image_markdown = []
                    if "candidates" in response_data and response_data["candidates"]:
                        for i, candidate in enumerate(response_data["candidates"]):
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "inlineData" in part and "data" in part["inlineData"]:
                                        mime_type = part["inlineData"].get("mimeType", "image/png")
                                        data = part["inlineData"]["data"]
                                        image_markdown.append(
                                            f"![image_{i+1}](data:{mime_type};base64,{data})"
                                        )
                    
                    # Combine all images into one response
                    content = "\n\n".join(image_markdown)
                    
                    if not image_markdown:
                        content = "No images were generated. Please try again with a different prompt."
                    
                    # Send response
                    if is_stream:
                        yield self._format_data(
                            is_stream=True,
                            model=model,
                            content=content,
                        )
                        # Send a final message with usage if available
                        yield self._format_data(
                            is_stream=True,
                            model=model,
                            content=None,
                        )
                    else:
                        yield self._format_data(
                            is_stream=False,
                            model=model,
                            content=content,
                        )
                        
            except Exception as err:
                error_status = await self.emit_status(
                    f"❌ An error occurred", True, True
                )
                yield self._format_data(
                    is_stream=body.get("stream", False),
                    content=f"{error_status}Error processing image request: {str(err)}",
                )
                
        return StreamingResponse(stream_response())

    def _extract_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Extract prompt from messages."""
        prompt = ""
        # Consider only the latest user message for simplicity
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            # If content is a list (mixed content format)
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt += item.get("text", "") + " "
            # If content is a string
            elif isinstance(content, str):
                prompt += content
                # Remove any image markdown
                pattern = r"!\[[^\]]*\]\(([^)]+)\)"
                prompt = re.sub(pattern, "", prompt)
            # We only need the most recent user message
            break
        return prompt.strip()
    
    def _get_dimensions(self, aspect_ratio: str) -> tuple[int, int]:
        """Get width and height based on aspect ratio."""
        if aspect_ratio == "1:1":
            return 1024, 1024
        elif aspect_ratio == "16:9":
            return 1344, 768
        elif aspect_ratio == "9:16":
            return 768, 1344
        elif aspect_ratio == "4:3":
            return 1024, 768
        elif aspect_ratio == "3:4":
            return 768, 1024
        else:
            return 1024, 1024  # Default to square

    def _format_data(
        self,
        is_stream: bool,
        model: str = "",
        content: Optional[str] = "",
        usage: Optional[dict] = None,
    ) -> str:
        """Format the response data in the expected OpenAI-compatible format."""
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if content is not None:
            data["choices"] = [
                {
                    "finish_reason": "stop" if not is_stream else None,
                    "index": 0,
                    "delta" if is_stream else "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ]
        if usage:
            data["usage"] = usage
        if is_stream:
            return f"data: {json.dumps(data)}\n\n"
        else:
            return json.dumps(data)