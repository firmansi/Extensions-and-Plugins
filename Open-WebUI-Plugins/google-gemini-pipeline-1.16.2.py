"""
title: Google Gemini Pipeline
author: owndev, olivier-lacroix
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.16.2
required_open_webui_version: 0.11.0
license: Apache License 2.0
description: Highly optimized Google Gemini pipeline with advanced image and video generation capabilities, intelligent compression, and streamlined processing workflows. (Updated for OpenWebUI 0.11.0 + Native Server-Side Tool Calling & Gemini 3.x Fixes)
"""

import os
import re
import time
import asyncio
import base64
import hashlib
import logging
import io
import uuid
import json
import inspect
import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError
from typing import List, Union, Optional, Dict, Any, Tuple, AsyncIterator, Callable
from pydantic_core import core_schema
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from open_webui.env import SRC_LOG_LEVELS
from open_webui.internal.db import get_async_db_context
from fastapi import Request, UploadFile, BackgroundTasks
from open_webui.routers.files import upload_file
from open_webui.models.chats import Chats
from open_webui.models.users import UserModel, Users
from starlette.datastructures import Headers

ASPECT_RATIO_OPTIONS: List[str] = [
    "default",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

RESOLUTION_OPTIONS: List[str] = [
    "default",
    "1K",
    "2K",
    "4K",
]

VIDEO_ASPECT_RATIO_OPTIONS: List[str] = [
    "default",
    "16:9",
    "9:16",
]

VIDEO_RESOLUTION_OPTIONS: List[str] = [
    "default",
    "720p",
    "1080p",
    "4k",
]

VIDEO_DURATION_OPTIONS: List[str] = [
    "default",
    "4",
    "5",
    "6",
    "8",
]

VIDEO_PERSON_GENERATION_OPTIONS: List[str] = [
    "default",
    "allow_all",
    "allow_adult",
    "dont_allow",
]


class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None

        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        if not value or value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:
            return value

        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        if not value or not value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:
            return value[len("encrypted:") :]

        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


class Pipe:
    """
    Pipeline for interacting with Google Gemini models.
    """

    class UserValves(BaseModel):
        IMAGE_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for image generation.",
            json_schema_extra={"enum": ASPECT_RATIO_OPTIONS},
        )
        IMAGE_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_RESOLUTION", "default"),
            description="Default resolution for image generation.",
            json_schema_extra={"enum": RESOLUTION_OPTIONS},
        )
        VIDEO_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for video generation (16:9 landscape or 9:16 portrait).",
            json_schema_extra={"enum": VIDEO_ASPECT_RATIO_OPTIONS},
        )
        VIDEO_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_RESOLUTION", "default"),
            description="Default resolution for video generation (720p, 1080p, or 4k).",
            json_schema_extra={"enum": VIDEO_RESOLUTION_OPTIONS},
        )
        VIDEO_GENERATION_DURATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_DURATION", "default"),
            description="Default duration in seconds for video generation (4, 5, 6, or 8 - availability varies by model).",
            json_schema_extra={"enum": VIDEO_DURATION_OPTIONS},
        )

    class Valves(BaseModel):
        BASE_URL: str = Field(
            default=os.getenv(
                "GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/"
            ),
            description="Base URL for the Google Generative AI API.",
        )
        GOOGLE_API_KEY: EncryptedStr = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI (used if USE_VERTEX_AI is false).",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_VERSION: str = Field(
            default=os.getenv("GOOGLE_API_VERSION", "v1beta"),
            description="API version to use for Google Generative AI (e.g., v1beta, v1).",
        )
        STREAMING_ENABLED: bool = Field(
            default=os.getenv("GOOGLE_STREAMING_ENABLED", "true").lower() == "true",
            description="Enable streaming responses (set false to force non-streaming mode).",
        )
        NATIVE_TOOL_CALLING_ENABLED: bool = Field(
            default=os.getenv("GOOGLE_NATIVE_TOOL_CALLING_ENABLED", "true").lower()
            == "true",
            description="Enable native OpenWebUI tool calling. When enabled and tools are available, streaming is temporarily disabled so tools can be executed server-side.",
        )
        INCLUDE_THOUGHTS: bool = Field(
            default=os.getenv("GOOGLE_INCLUDE_THOUGHTS", "true").lower() == "true",
            description="Enable Gemini thoughts outputs (set false to disable).",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("GOOGLE_THINKING_BUDGET", "-1")),
            description="Thinking budget for Gemini 2.5 models (0=disabled, -1=dynamic, 1-32768=fixed token limit). Not used for Gemini 3 models which use THINKING_LEVEL instead.",
        )
        THINKING_LEVEL: str = Field(
            default=os.getenv("GOOGLE_THINKING_LEVEL", ""),
            description="Thinking level for Gemini 3 models. Ignored for other models.",
        )
        USE_VERTEX_AI: bool = Field(
            default=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true",
            description="Whether to use Google Cloud Vertex AI instead of the Google Generative AI API.",
        )
        VERTEX_PROJECT: str | None = Field(
            default=os.getenv("GOOGLE_CLOUD_PROJECT"),
            description="The Google Cloud project ID to use with Vertex AI.",
        )
        VERTEX_LOCATION: str = Field(
            default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            description="The Google Cloud region to use with Vertex AI.",
        )
        VERTEX_AI_RAG_STORE: str | None = Field(
            default=os.getenv("GOOGLE_VERTEX_AI_RAG_STORE"),
            description="Vertex AI RAG Store path for grounding. Only used when USE_VERTEX_AI is true.",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("GOOGLE_USE_PERMISSIVE_SAFETY", "false").lower()
            == "true",
            description="Use permissive safety settings for content generation.",
        )
        MODEL_CACHE_TTL: int = Field(
            default=int(os.getenv("GOOGLE_MODEL_CACHE_TTL", "600")),
            description="Time in seconds to cache the model list before refreshing",
        )
        RETRY_COUNT: int = Field(
            default=int(os.getenv("GOOGLE_RETRY_COUNT", "2")),
            description="Number of times to retry API calls on temporary failures",
        )
        DEFAULT_SYSTEM_PROMPT: str = Field(
            default=os.getenv("GOOGLE_DEFAULT_SYSTEM_PROMPT", ""),
            description="Default system prompt applied to all chats. Leave empty to disable.",
        )
        ENABLE_FORWARD_USER_INFO_HEADERS: bool = Field(
            default=os.getenv(
                "GOOGLE_ENABLE_FORWARD_USER_INFO_HEADERS", "false"
            ).lower()
            == "true",
            description="Whether to forward user information headers.",
        )
        MODEL_ADDITIONAL: str = Field(
            default=os.getenv("GOOGLE_MODEL_ADDITIONAL", ""),
            description="A comma-separated list of model IDs to manually add to the list of available models.",
        )
        MODEL_WHITELIST: str = Field(
            default=os.getenv("GOOGLE_MODEL_WHITELIST", ""),
            description="A comma-separated list of model IDs to show in the models list.",
        )
        USE_ENTERPRISE_WEB_SEARCH: bool = Field(
            default=os.getenv("GOOGLE_USE_ENTERPRISE_WEB_SEARCH", "false").lower()
            == "true",
            description="Whether to use Enterprise Web Search instead of standard Google search when grounding is enabled.",
        )

        IMAGE_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for image generation.",
            json_schema_extra={"enum": ASPECT_RATIO_OPTIONS},
        )
        IMAGE_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_RESOLUTION", "default"),
            description="Default resolution for image generation.",
            json_schema_extra={"enum": RESOLUTION_OPTIONS},
        )
        IMAGE_MAX_SIZE_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_MAX_SIZE_MB", "15.0")),
            description="Maximum image size in MB before compression is applied",
        )
        IMAGE_MAX_DIMENSION: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_MAX_DIMENSION", "2048")),
            description="Maximum width or height in pixels before resizing",
        )
        IMAGE_COMPRESSION_QUALITY: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_COMPRESSION_QUALITY", "85")),
            description="JPEG compression quality",
        )
        IMAGE_ENABLE_OPTIMIZATION: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ENABLE_OPTIMIZATION", "true").lower()
            == "true",
            description="Enable intelligent image optimization for API compatibility",
        )
        IMAGE_PNG_COMPRESSION_THRESHOLD_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_PNG_THRESHOLD_MB", "0.5")),
            description="PNG files above this size (MB) will be converted to JPEG for better compression",
        )
        IMAGE_HISTORY_MAX_REFERENCES: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_HISTORY_MAX_REFERENCES", "5")),
            description="Maximum total number of images (history + current message) to include in a generation call",
        )
        IMAGE_ADD_LABELS: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ADD_LABELS", "true").lower() == "true",
            description="If true, add small text labels like [Image 1] before each image part.",
        )
        IMAGE_DEDUP_HISTORY: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_DEDUP_HISTORY", "true").lower() == "true",
            description="If true, deduplicate identical images (by hash) when constructing history context",
        )
        IMAGE_HISTORY_FIRST: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_HISTORY_FIRST", "true").lower() == "true",
            description="If true, history images precede current message images.",
        )

        VIDEO_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for video generation.",
            json_schema_extra={"enum": VIDEO_ASPECT_RATIO_OPTIONS},
        )
        VIDEO_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_RESOLUTION", "default"),
            description="Default resolution for video generation.",
            json_schema_extra={"enum": VIDEO_RESOLUTION_OPTIONS},
        )
        VIDEO_GENERATION_DURATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_DURATION", "default"),
            description="Default duration in seconds for video generation.",
            json_schema_extra={"enum": VIDEO_DURATION_OPTIONS},
        )
        VIDEO_GENERATION_NEGATIVE_PROMPT: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_NEGATIVE_PROMPT", ""),
            description="Default negative prompt for video generation.",
        )
        VIDEO_GENERATION_PERSON_GENERATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_PERSON_GENERATION", "default"),
            description="Controls generation of people in videos.",
            json_schema_extra={"enum": VIDEO_PERSON_GENERATION_OPTIONS},
        )
        VIDEO_GENERATION_ENHANCE_PROMPT: bool = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_ENHANCE_PROMPT", "true").lower()
            == "true",
            description="Enable prompt enhancement for video generation.",
        )
        VIDEO_POLL_INTERVAL: int = Field(
            default=int(os.getenv("GOOGLE_VIDEO_POLL_INTERVAL", "10")),
            description="Polling interval in seconds when waiting for video generation.",
        )
        VIDEO_POLL_TIMEOUT: int = Field(
            default=int(os.getenv("GOOGLE_VIDEO_POLL_TIMEOUT", "600")),
            description="Maximum time in seconds to wait for video generation before timing out (0=no limit).",
        )

    async def _gather_history_images(
        self,
        messages: List[Dict[str, Any]],
        last_user_msg: Dict[str, Any],
        optimization_stats: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        history_images: List[Dict[str, Any]] = []
        for msg in messages:
            if msg is last_user_msg:
                continue
            if msg.get("role") not in {"user", "assistant"}:
                continue
            _p, parts = await self._extract_images_from_message(
                msg, stats_list=optimization_stats
            )
            if parts:
                history_images.extend(parts)
        return history_images

    def _deduplicate_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.valves.IMAGE_DEDUP_HISTORY:
            return images
        seen: set[str] = set()
        result: List[Dict[str, Any]] = []
        for part in images:
            try:
                data = part["inline_data"]["data"]
                h = hashlib.sha256(data.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
            except Exception as e:
                self.log.debug(f"Skipping image in deduplication due to error: {e}")
            result.append(part)
        return result

    def _combine_system_prompts(
        self, user_system_prompt: Optional[str]
    ) -> Optional[str]:
        default_prompt = self.valves.DEFAULT_SYSTEM_PROMPT.strip()
        user_prompt = user_system_prompt.strip() if user_system_prompt else ""

        if default_prompt and user_prompt:
            return f"{default_prompt}\n\n{user_prompt}"
        elif default_prompt:
            return default_prompt
        elif user_prompt:
            return user_prompt
        return None

    def _apply_order_and_limit(
        self,
        history: List[Dict[str, Any]],
        current: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[bool]]:
        history_first = self.valves.IMAGE_HISTORY_FIRST
        limit = max(1, self.valves.IMAGE_HISTORY_MAX_REFERENCES)
        combined: List[Dict[str, Any]] = []
        reused_flags: List[bool] = []

        def append(parts: List[Dict[str, Any]], reused: bool):
            for p in parts:
                if len(combined) >= limit:
                    break
                combined.append(p)
                reused_flags.append(reused)

        if history_first:
            append(history, True)
            append(current, False)
        else:
            append(current, False)
            append(history, True)
        return combined, reused_flags

    async def _emit_image_stats(
        self,
        ordered_stats: List[Dict[str, Any]],
        reused_flags: List[bool],
        total_limit: int,
        __event_emitter__: Callable,
    ) -> None:
        if not ordered_stats:
            return
        for idx, stat in enumerate(ordered_stats, start=1):
            reused = reused_flags[idx - 1] if idx - 1 < len(reused_flags) else False
            stat_copy = dict(stat) if stat else {}
            stat_copy.update({"index": idx, "reused": reused})
            if stat and stat.get("original_size_mb") is not None:
                desc = f"Image {idx}: {stat['original_size_mb']:.2f}MB -> {stat['final_size_mb']:.2f}MB"
                if stat.get("quality") is not None:
                    desc += f" (Q{stat['quality']})"
            else:
                desc = f"Image {idx}: (no metrics)"
            reasons = stat.get("reasons") if stat else None
            if reasons:
                desc += " | " + ", ".join(reasons[:3])
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_optimization",
                        "description": desc,
                        "index": idx,
                        "done": False,
                        "details": stat_copy,
                    },
                }
            )
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "action": "image_optimization",
                    "description": f"{len(ordered_stats)} image(s) processed (limit {total_limit}).",
                    "done": True,
                },
            }
        )

    async def _build_image_generation_contents(
        self,
        messages: List[Dict[str, Any]],
        __event_emitter__: Callable,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        user_system_instruction = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        system_instruction = self._combine_system_prompts(user_system_instruction)

        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_msg:
            raise ValueError("No user message found")

        optimization_stats: List[Dict[str, Any]] = []
        history_images = await self._gather_history_images(
            messages, last_user_msg, optimization_stats
        )
        prompt, current_images = await self._extract_images_from_message(
            last_user_msg, stats_list=optimization_stats
        )

        history_images = self._deduplicate_images(history_images)
        current_images = self._deduplicate_images(current_images)

        combined, reused_flags = self._apply_order_and_limit(
            history_images, current_images
        )

        if not prompt and not combined:
            raise ValueError("No prompt or images provided")
        if not prompt and combined:
            prompt = "Analyze and describe the provided images."

        ordered_stats: List[Dict[str, Any]] = []
        if optimization_stats:
            hash_map: Dict[str, Dict[str, Any]] = {}
            for s in optimization_stats:
                fh = s.get("final_hash")
                if fh and fh not in hash_map:
                    hash_map[fh] = s
            for part in combined:
                try:
                    fh = hashlib.sha256(
                        part["inline_data"]["data"].encode()
                    ).hexdigest()
                    ordered_stats.append(hash_map.get(fh) or {})
                except Exception:
                    ordered_stats.append({})

        await self._emit_image_stats(
            ordered_stats,
            reused_flags,
            self.valves.IMAGE_HISTORY_MAX_REFERENCES,
            __event_emitter__,
        )

        if combined:
            mapping = [
                {
                    "index": i + 1,
                    "label": (
                        f"Image {i + 1}" if self.valves.IMAGE_ADD_LABELS else str(i + 1)
                    ),
                    "reused": reused_flags[i],
                    "origin": "history" if reused_flags[i] else "current",
                }
                for i in range(len(combined))
            ]
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_reference_map",
                        "description": f"{len(combined)} image(s) included (limit {self.valves.IMAGE_HISTORY_MAX_REFERENCES}).",
                        "images": mapping,
                        "done": True,
                    },
                }
            )

        parts: List[Dict[str, Any]] = []

        final_prompt = prompt
        if system_instruction and prompt:
            final_prompt = f"{system_instruction}\n\n{prompt}"
        elif system_instruction and not prompt:
            final_prompt = system_instruction

        if final_prompt:
            parts.append({"text": final_prompt})
        if self.valves.IMAGE_ADD_LABELS:
            for idx, part in enumerate(combined, start=1):
                parts.append({"text": f"[Image {idx}]"})
                parts.append(part)
        else:
            parts.extend(combined)

        return [{"role": "user", "parts": parts}], None

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Google Gemini: "
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0

    def _get_client(self) -> genai.Client:
        self._validate_api_key()

        if self.valves.USE_VERTEX_AI:
            return genai.Client(
                vertexai=True,
                project=self.valves.VERTEX_PROJECT,
                location=self.valves.VERTEX_LOCATION,
            )
        else:
            headers = {}
            if (
                self.valves.ENABLE_FORWARD_USER_INFO_HEADERS
                and hasattr(self, "user")
                and self.user
            ):

                def sanitize_header_value(value: Any, max_length: int = 255) -> str:
                    if value is None:
                        return ""
                    sanitized = re.sub(r"[\x00-\x1F\x7F]", "", str(value))
                    sanitized = sanitized.strip()
                    return (
                        sanitized[:max_length]
                        if len(sanitized) > max_length
                        else sanitized
                    )

                user_attrs = {
                    "X-OpenWebUI-User-Name": sanitize_header_value(
                        getattr(self.user, "name", None)
                    ),
                    "X-OpenWebUI-User-Id": sanitize_header_value(
                        getattr(self.user, "id", None)
                    ),
                    "X-OpenWebUI-User-Email": sanitize_header_value(
                        getattr(self.user, "email", None)
                    ),
                    "X-OpenWebUI-User-Role": sanitize_header_value(
                        getattr(self.user, "role", None)
                    ),
                }
                headers = {k: v for k, v in user_attrs.items() if v not in (None, "")}
            options = types.HttpOptions(
                api_version=self.valves.API_VERSION,
                base_url=self.valves.BASE_URL,
                headers=headers,
            )
            return genai.Client(
                api_key=EncryptedStr.decrypt(self.valves.GOOGLE_API_KEY),
                http_options=options,
            )

    def _validate_api_key(self) -> None:
        if self.valves.USE_VERTEX_AI:
            if not self.valves.VERTEX_PROJECT:
                raise ValueError(
                    "VERTEX_PROJECT is not set. Please provide the Google Cloud project ID."
                )
        else:
            if not self.valves.GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Please provide the API key in the environment variables or valves."
                )

    def strip_prefix(self, model_name: str) -> str:
        stripped = re.sub(r"^(?:.*/|[^.]*\.)", "", model_name)
        return stripped

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        current_time = time.time()
        if (
            not force_refresh
            and self._model_cache is not None
            and (current_time - self._model_cache_time) < self.valves.MODEL_CACHE_TTL
        ):
            return self._model_cache

        try:
            client = self._get_client()
            models = list(client.models.list())

            additional = self.valves.MODEL_ADDITIONAL
            if additional:
                existing_model_names = {self.strip_prefix(m.name) for m in models}
                additional_ids = set(re.findall(r"[^,\s]+", additional))

                for model_id in additional_ids.difference(existing_model_names):
                    models.append(types.Model(name=f"models/{model_id}"))

            available_models = []
            for model in models:
                actions = model.supported_actions
                model_id_stripped = self.strip_prefix(model.name)
                is_content_model = actions is None or "generateContent" in actions
                is_video_model = (
                    actions is not None and "generateVideos" in actions
                ) or model_id_stripped.startswith("veo-")
                if is_content_model or is_video_model:
                    model_id = model_id_stripped
                    model_name = model.display_name or model_id

                    supports_image_generation = self._check_image_generation_support(
                        model_id
                    )
                    if supports_image_generation:
                        model_name += " 🎨"

                    supports_video_generation = self._check_video_generation_support(
                        model_id
                    )
                    if supports_video_generation:
                        model_name += " 🎬"

                    available_models.append(
                        {
                            "id": model_id,
                            "name": model_name,
                            "image_generation": supports_image_generation,
                            "video_generation": supports_video_generation,
                        }
                    )

            model_map = {model["id"]: model for model in available_models}

            whitelist = self.valves.MODEL_WHITELIST
            if whitelist:
                whitelisted_ids = set(re.findall(r"[^,\s]+", whitelist))
                filtered_models = {
                    k: v for k, v in model_map.items() if k in whitelisted_ids
                }
            else:
                filtered_models = {
                    k: v
                    for k, v in model_map.items()
                    if k.startswith("gemini-") or k.startswith("veo-")
                }

            self._model_cache = list(filtered_models.values())
            self._model_cache_time = current_time
            return self._model_cache

        except Exception as e:
            self.log.exception(f"Could not fetch models from Google: {str(e)}")
            return [{"id": "error", "name": f"Could not fetch models: {str(e)}"}]

    def _check_image_generation_support(self, model_id: str) -> bool:
        image_generation_models = [
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-image-preview",
            "gemini-3-flash-image",
            "gemini-3-flash-image-preview",
            "gemini-3.1-flash-image-preview",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-image-lite",
            "gemini-3.7-flash-image",
            "gemini-3.7-flash-image-preview",
            "gemini-3-pro-image",
            "gemini-3-pro-image-preview",
        ]

        for pattern in image_generation_models:
            if model_id == pattern or pattern in model_id:
                return True

        if "image" in model_id.lower():
            return True

        return False

    def _is_gemini_3_family_model(self, model_id: str) -> bool:
        """Detect any Gemini 3.x model, including 3, 3.1, 3.5, 3.7, etc."""
        model_lower = model_id.lower()
        if model_lower.startswith("gemini-3-"):
            return True
        if model_lower.startswith("gemini-3."):
            return True
        return bool(re.search(r"\bgemini-3(?:\.\d+)?-", model_lower))

    def _is_gemini_3_image_model(self, model_id: str) -> bool:
        return self._is_gemini_3_family_model(
            model_id
        ) and self._check_image_generation_support(model_id)

    def _check_image_config_support(self, model_id: str) -> bool:
        return self._is_gemini_3_image_model(model_id)

    def _check_thinking_support(self, model_id: str) -> bool:
        non_thinking_models = [
            "gemini-2.5-flash-image-preview",
            "gemini-2.5-flash-image",
        ]
        for pattern in non_thinking_models:
            if model_id == pattern or pattern in model_id:
                return False

        if self._is_gemini_3_image_model(model_id):
            return True

        if "image" in model_id.lower() and (
            "generation" in model_id.lower() or "preview" in model_id.lower()
        ):
            return False

        return True

    def _check_thinking_level_support(self, model_id: str) -> bool:
        return self._is_gemini_3_family_model(model_id)

    def _get_supported_thinking_levels(self, model_id: str) -> List[str]:
        model_lower = model_id.lower()
        if model_lower.startswith("gemini-3.1-flash-image") or model_lower.startswith(
            "gemini-3.7-flash-image"
        ):
            return ["minimal", "high"]
        if self._is_gemini_3_family_model(model_id):
            return ["low", "high"]
        return []

    def _coerce_thinking_level(
        self, requested_level: str, supported_levels: List[str]
    ) -> Optional[str]:
        if not supported_levels:
            return None

        level_rank = {"minimal": 0, "low": 1, "medium": 2, "high": 3}
        requested_rank = level_rank.get(requested_level)
        if requested_rank is None:
            return None

        supported_ranks = sorted(
            (level_rank[level], level)
            for level in supported_levels
            if level in level_rank
        )
        if not supported_ranks:
            return None

        best_rank, best_level = min(
            supported_ranks,
            key=lambda item: (abs(item[0] - requested_rank), -item[0]),
        )
        return best_level

    def _validate_thinking_level(self, level: str, model_id: str = "") -> Optional[str]:
        if not level:
            return None

        normalized = level.strip().lower()
        valid_levels = ["minimal", "low", "medium", "high"]

        if normalized not in valid_levels:
            return None

        supported_levels = self._get_supported_thinking_levels(model_id)
        if not supported_levels or normalized in supported_levels:
            return normalized

        coerced_level = self._coerce_thinking_level(normalized, supported_levels)
        if coerced_level:
            return coerced_level

        return None

    def _validate_thinking_budget(self, budget: int) -> int:
        if budget == -1:
            return -1
        if budget == 0:
            return 0
        if budget > 0:
            if budget > 32768:
                return 32768
            return budget
        return -1

    def _validate_aspect_ratio(self, aspect_ratio: str) -> Optional[str]:
        if not aspect_ratio or aspect_ratio == "default":
            return None
        normalized = aspect_ratio.strip()
        valid_ratios = [r for r in ASPECT_RATIO_OPTIONS if r != "default"]
        if normalized in valid_ratios:
            return normalized
        return "1:1"

    def _validate_resolution(self, resolution: str) -> Optional[str]:
        if not resolution or resolution.lower() == "default":
            return None
        normalized = resolution.strip().upper()
        valid_resolutions = [r for r in RESOLUTION_OPTIONS if r.lower() != "default"]
        if normalized in valid_resolutions:
            return normalized
        return "2K"

    def _check_video_generation_support(self, model_id: str) -> bool:
        model_lower = model_id.lower()
        return model_lower.startswith("veo-") or (
            "veo" in model_lower and "generate" in model_lower
        )

    @staticmethod
    def _is_open_webui_image_tool(tool_name: str) -> bool:
        return tool_name in {"generate_image", "edit_image"}

    @staticmethod
    def _image_data_hash(image_data: Any) -> str:
        if isinstance(image_data, bytes):
            return hashlib.sha256(image_data).hexdigest()
        return hashlib.sha256(str(image_data).encode("utf-8")).hexdigest()

    async def _emit_generated_image_files(
        self,
        image_files: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable],
    ) -> bool:
        if not image_files or not __event_emitter__:
            return False
        try:
            await __event_emitter__(
                {
                    "type": "files",
                    "data": {"files": image_files},
                }
            )
            return True
        except Exception:
            return False

    async def _emit_generated_video_files(
        self,
        video_files: List[Dict[str, Any]],
        __event_emitter__: Optional[Callable],
    ) -> bool:
        if not video_files or not __event_emitter__:
            return False
        try:
            await __event_emitter__(
                {
                    "type": "files",
                    "data": {"files": video_files},
                }
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _build_generated_image_file(
        content_url: str,
        mime_type: str,
        name: str = "Generated Image",
    ) -> Dict[str, Any]:
        return {
            "type": "image",
            "url": content_url,
            "content_type": mime_type,
            "name": name,
            "meta": {"content_type": mime_type},
        }

    @staticmethod
    def _build_generated_video_file(
        file_id: str,
        content_url: str,
        filename: str,
        mime_type: str,
        size: int,
    ) -> Dict[str, Any]:
        return {
            "id": file_id,
            "type": "file",
            "url": content_url,
            "name": filename,
            "filename": filename,
            "size": size,
            "content_type": mime_type,
            "meta": {
                "content_type": mime_type,
                "size": size,
            },
        }

    def _check_veo_3_1_support(self, model_id: str) -> bool:
        return "veo-3.1" in model_id.lower()

    def _get_veo_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        model_lower = model_id.lower()
        is_fast = "fast" in model_lower

        if "veo-3.1" in model_lower:
            return {
                "version": "3.1",
                "is_fast": is_fast,
                "supports_enhance_prompt": False,
                "supports_resolution": True,
                "valid_resolutions": ["720p", "1080p", "4k"],
                "valid_durations": [4, 6, 8],
                "max_videos": 1,
                "supports_reference_images": True,
                "supports_last_frame": True,
                "supports_extension": True,
            }
        if "veo-3" in model_lower:
            return {
                "version": "3",
                "is_fast": is_fast,
                "supports_enhance_prompt": False,
                "supports_resolution": True,
                "valid_resolutions": ["720p", "1080p"],
                "valid_durations": [8],
                "max_videos": 1,
                "supports_reference_images": False,
                "supports_last_frame": True,
                "supports_extension": False,
            }
        if "veo-2" in model_lower:
            return {
                "version": "2",
                "is_fast": False,
                "supports_enhance_prompt": False,
                "supports_resolution": False,
                "valid_resolutions": [],
                "valid_durations": [5, 6, 8],
                "max_videos": 2,
                "supports_reference_images": False,
                "supports_last_frame": True,
                "supports_extension": False,
            }
        return {
            "version": "unknown",
            "is_fast": is_fast,
            "supports_enhance_prompt": False,
            "supports_resolution": False,
            "valid_resolutions": [],
            "valid_durations": [8],
            "max_videos": 1,
            "supports_reference_images": False,
            "supports_last_frame": False,
            "supports_extension": False,
        }

    def _validate_video_aspect_ratio(self, aspect_ratio: str) -> Optional[str]:
        if not aspect_ratio or aspect_ratio == "default":
            return None
        normalized = aspect_ratio.strip()
        valid = [r for r in VIDEO_ASPECT_RATIO_OPTIONS if r != "default"]
        if normalized in valid:
            return normalized
        return None

    def _validate_video_resolution(self, resolution: str) -> Optional[str]:
        if not resolution or resolution.lower() == "default":
            return None
        normalized = resolution.strip().lower()
        valid = [r for r in VIDEO_RESOLUTION_OPTIONS if r.lower() != "default"]
        if normalized in valid:
            return normalized
        return None

    def _validate_video_duration(self, duration: str) -> Optional[int]:
        if not duration or duration.lower() == "default":
            return None
        valid = {int(d) for d in VIDEO_DURATION_OPTIONS if d != "default"}
        try:
            val = int(duration)
            if val in valid:
                return val
        except (ValueError, TypeError):
            pass
        return None

    def _build_video_generation_config(
        self,
        body: Dict[str, Any],
        __user__: Optional[dict] = None,
        model_id: str = "",
    ) -> types.GenerateVideosConfig:
        caps = self._get_veo_model_capabilities(model_id)

        user_ar = self._get_user_valve_value(__user__, "VIDEO_GENERATION_ASPECT_RATIO")
        aspect_ratio = self._validate_video_aspect_ratio(
            body.get(
                "aspect_ratio", user_ar or self.valves.VIDEO_GENERATION_ASPECT_RATIO
            )
        )

        user_res = self._get_user_valve_value(__user__, "VIDEO_GENERATION_RESOLUTION")
        resolution = self._validate_video_resolution(
            body.get("resolution", user_res or self.valves.VIDEO_GENERATION_RESOLUTION)
        )

        user_dur = self._get_user_valve_value(__user__, "VIDEO_GENERATION_DURATION")
        duration_seconds = self._validate_video_duration(
            body.get("duration", user_dur or self.valves.VIDEO_GENERATION_DURATION)
        )

        negative_prompt = (
            body.get("negative_prompt", self.valves.VIDEO_GENERATION_NEGATIVE_PROMPT)
            or None
        )

        person_generation_raw = body.get(
            "person_generation", self.valves.VIDEO_GENERATION_PERSON_GENERATION
        )
        person_generation = None
        if person_generation_raw and person_generation_raw != "default":
            valid_person_values = [
                v for v in VIDEO_PERSON_GENERATION_OPTIONS if v != "default"
            ]
            if person_generation_raw in valid_person_values:
                person_generation = person_generation_raw

        enhance_prompt = body.get(
            "enhance_prompt", self.valves.VIDEO_GENERATION_ENHANCE_PROMPT
        )

        number_of_videos_raw = body.get("number_of_videos", 1)
        try:
            number_of_videos = int(number_of_videos_raw)
        except (ValueError, TypeError):
            number_of_videos = 1

        config_params: Dict[str, Any] = {
            "number_of_videos": min(max(number_of_videos, 1), caps["max_videos"]),
        }

        if caps["supports_enhance_prompt"] and enhance_prompt:
            config_params["enhance_prompt"] = enhance_prompt

        if aspect_ratio:
            config_params["aspect_ratio"] = aspect_ratio

        if resolution and caps["supports_resolution"]:
            if resolution in caps["valid_resolutions"]:
                config_params["resolution"] = resolution

        if duration_seconds:
            if duration_seconds in caps["valid_durations"]:
                config_params["duration_seconds"] = duration_seconds

        if negative_prompt:
            config_params["negative_prompt"] = negative_prompt
        if person_generation:
            config_params["person_generation"] = person_generation

        return types.GenerateVideosConfig(**config_params)

    def pipes(self) -> List[Dict[str, str]]:
        try:
            self.name = "Google Gemini: "
            return self.get_google_models()
        except ValueError as e:
            return [{"id": "error", "name": str(e)}]
        except Exception as e:
            return [{"id": "error", "name": f"An unexpected error occurred: {str(e)}"}]

    def _prepare_model_id(self, model_id: str) -> str:
        original_model_id = model_id
        model_id = self.strip_prefix(model_id)
        valid_prefixes = ("gemini-", "veo-")

        if not model_id.startswith(valid_prefixes):
            models_list = self.get_google_models()
            found_model = next(
                (m["id"] for m in models_list if m["name"] == original_model_id), None
            )
            if found_model and found_model.startswith(valid_prefixes):
                model_id = found_model
            else:
                if not model_id.startswith(valid_prefixes):
                    raise ValueError(
                        f"Invalid or unsupported Google model ID or name: '{original_model_id}'"
                    )
        return model_id

    def _prepare_content(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        user_system_message = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        system_message = self._combine_system_prompts(user_system_message)

        contents = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue

            content = message.get("content", "")
            parts = []

            if isinstance(content, list):
                parts.extend(self._process_multimodal_content(content))
            elif isinstance(content, str):
                parts.append({"text": content})
            else:
                continue

            api_role = "model" if role == "assistant" else "user"
            if parts:
                contents.append({"role": api_role, "parts": parts})

        return contents, system_message

    def _process_multimodal_content(
        self, content_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        parts = []

        for item in content_list:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")

                if image_url.startswith("data:image"):
                    try:
                        optimized_image = self._optimize_image_for_api(image_url)
                        header, encoded = optimized_image.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]

                        if mime_type not in [
                            "image/jpeg",
                            "image/png",
                            "image/webp",
                            "image/heic",
                            "image/heif",
                        ]:
                            parts.append(
                                {"text": f"[Image type {mime_type} not supported]"}
                            )
                            continue

                        if len(encoded) > 15 * 1024 * 1024:
                            parts.append(
                                {
                                    "text": "[Image too large for processing - please use a smaller image]"
                                }
                            )
                            continue

                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded,
                                }
                            }
                        )
                    except Exception:
                        parts.append({"text": "[Image data could not be processed]"})
                else:
                    parts.append({"text": f"[Image URL not processed: {image_url}]"})

        return parts

    async def _extract_images_from_message(
        self,
        message: Dict[str, Any],
        *,
        stats_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        content = message.get("content", "")
        text_segments: List[str] = []
        image_parts: List[Dict[str, Any]] = []

        def _add_image(data_url: str):
            try:
                optimized = self._optimize_image_for_api(data_url, stats_list)
                header, b64 = optimized.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0]
                image_parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            except Exception:
                pass

        md_pattern = re.compile(
            r"!\[[^\]]*\]\((data:image[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)"
        )

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    text_segments.append(txt)
                    for match in md_pattern.finditer(txt):
                        url = match.group(1)
                        if url.startswith("data:"):
                            _add_image(url)
                        else:
                            b64 = await self._fetch_file_as_base64(url)
                            if b64:
                                _add_image(b64)
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        _add_image(url)
                    elif "/files/" in url or "/api/v1/files/" in url:
                        b64 = await self._fetch_file_as_base64(url)
                        if b64:
                            _add_image(b64)
        elif isinstance(content, str):
            text_segments.append(content)
            for match in md_pattern.finditer(content):
                url = match.group(1)
                if url.startswith("data:"):
                    _add_image(url)
                else:
                    b64 = await self._fetch_file_as_base64(url)
                    if b64:
                        _add_image(b64)

        prompt_text = " ".join(s.strip() for s in text_segments if s.strip())
        return prompt_text, image_parts

    def _optimize_image_for_api(
        self, image_data: str, stats_list: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        if not self.valves.IMAGE_ENABLE_OPTIMIZATION:
            return image_data

        max_size_mb = self.valves.IMAGE_MAX_SIZE_MB
        max_dimension = self.valves.IMAGE_MAX_DIMENSION
        base_quality = self.valves.IMAGE_COMPRESSION_QUALITY
        png_threshold = self.valves.IMAGE_PNG_COMPRESSION_THRESHOLD_MB

        try:
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]
            else:
                encoded = image_data
                mime_type = "image/png"

            image_bytes = base64.b64decode(encoded)
            original_size_mb = len(image_bytes) / (1024 * 1024)
            base64_size_mb = len(encoded) / (1024 * 1024)

            reasons: List[str] = []
            if original_size_mb > max_size_mb:
                reasons.append(f"size > {max_size_mb} MB")
            if base64_size_mb > max_size_mb * 1.4:
                reasons.append("base64 overhead")
            if mime_type == "image/png" and original_size_mb > png_threshold:
                reasons.append(f"PNG > {png_threshold}MB")

            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                resized_flag = False
                if width > max_dimension or height > max_dimension:
                    reasons.append(f"dimensions > {max_dimension}px")

                if not reasons:
                    if stats_list is not None:
                        stats_list.append(
                            {
                                "original_size_mb": round(original_size_mb, 4),
                                "final_size_mb": round(original_size_mb, 4),
                                "quality": None,
                                "format": mime_type.split("/")[-1].upper(),
                                "resized": False,
                                "reasons": ["no_optimization_needed"],
                                "final_hash": hashlib.sha256(
                                    encoded.encode()
                                ).hexdigest(),
                            }
                        )
                    return image_data

                if img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(
                        img,
                        mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None,
                    )
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                if width > max_dimension or height > max_dimension:
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_flag = True

                if original_size_mb > 5.0:
                    quality_levels = [
                        base_quality,
                        base_quality - 10,
                        base_quality - 20,
                        base_quality - 30,
                        base_quality - 40,
                        max(base_quality - 50, 25),
                    ]
                elif original_size_mb > 2.0:
                    quality_levels = [
                        base_quality,
                        base_quality - 5,
                        base_quality - 15,
                        base_quality - 25,
                        max(base_quality - 35, 35),
                    ]
                else:
                    quality_levels = [
                        min(base_quality + 5, 95),
                        base_quality,
                        base_quality - 10,
                        max(base_quality - 20, 50),
                    ]

                quality_levels = [max(1, min(100, q)) for q in quality_levels]

                for quality in quality_levels:
                    output_buffer = io.BytesIO()
                    format_type = (
                        "JPEG"
                        if original_size_mb > png_threshold or "jpeg" in mime_type
                        else "PNG"
                    )
                    output_mime = f"image/{format_type.lower()}"

                    img.save(
                        output_buffer,
                        format=format_type,
                        quality=quality,
                        optimize=True,
                    )
                    output_bytes = output_buffer.getvalue()
                    output_size_mb = len(output_bytes) / (1024 * 1024)

                    if output_size_mb <= max_size_mb:
                        optimized_b64 = base64.b64encode(output_bytes).decode("utf-8")
                        if stats_list is not None:
                            stats_list.append(
                                {
                                    "original_size_mb": round(original_size_mb, 4),
                                    "final_size_mb": round(output_size_mb, 4),
                                    "quality": quality,
                                    "format": format_type,
                                    "resized": resized_flag,
                                    "reasons": reasons,
                                    "final_hash": hashlib.sha256(
                                        optimized_b64.encode()
                                    ).hexdigest(),
                                }
                            )
                        return f"data:{output_mime};base64,{optimized_b64}"

                output_buffer = io.BytesIO()
                img.save(output_buffer, format="JPEG", quality=15, optimize=True)
                output_bytes = output_buffer.getvalue()
                output_size_mb = len(output_bytes) / (1024 * 1024)
                optimized_b64 = base64.b64encode(output_bytes).decode("utf-8")

                if stats_list is not None:
                    stats_list.append(
                        {
                            "original_size_mb": round(original_size_mb, 4),
                            "final_size_mb": round(output_size_mb, 4),
                            "quality": 15,
                            "format": "JPEG",
                            "resized": resized_flag,
                            "reasons": reasons + ["fallback_min_quality"],
                            "final_hash": hashlib.sha256(
                                optimized_b64.encode()
                            ).hexdigest(),
                        }
                    )
                return f"data:image/jpeg;base64,{optimized_b64}"

        except Exception:
            if image_data.startswith("data:"):
                if stats_list is not None:
                    stats_list.append(
                        {
                            "original_size_mb": None,
                            "final_size_mb": None,
                            "quality": None,
                            "format": None,
                            "resized": False,
                            "reasons": ["optimization_failed"],
                            "final_hash": (
                                hashlib.sha256(encoded.encode()).hexdigest()
                                if "encoded" in locals()
                                else None
                            ),
                        }
                    )
                return image_data
            return f"data:image/jpeg;base64,{encoded if 'encoded' in locals() else image_data}"

    async def _fetch_file_as_base64(self, file_url: str) -> Optional[str]:
        try:
            if "/api/v1/files/" in file_url:
                fid = file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
            else:
                fid = file_url.split("/files/")[-1].split("/")[0].split("?")[0]

            from pathlib import Path
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage

            file_obj = await Files.get_file_by_id(fid)
            if file_obj and file_obj.path:
                file_path = await asyncio.to_thread(Storage.get_file, file_obj.path)
                file_path = Path(file_path)
                if file_path.is_file():
                    async with aiofiles.open(file_path, "rb") as fp:
                        raw = await fp.read()
                    enc = base64.b64encode(raw).decode()
                    mime = file_obj.meta.get("content_type", "image/png")
                    return f"data:{mime};base64,{enc}"
        except Exception:
            pass
        return None

    async def _upload_image_with_status(
        self,
        image_data: Any,
        mime_type: str,
        __request__: Request,
        __user__: dict,
        __event_emitter__: Callable,
    ) -> str:
        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Uploading generated image to your library...",
                        "done": False,
                    },
                }
            )

            self.user = user = await Users.get_user_by_id(__user__["id"])

            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            image_url = await self._upload_image(
                __request__=__request__,
                user=user,
                image_data=image_data_b64,
                mime_type=mime_type,
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Image uploaded successfully!",
                        "done": True,
                    },
                }
            )

            return image_url

        except Exception:
            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Using inline image (upload failed)",
                        "done": True,
                    },
                }
            )

            return f"data:{mime_type};base64,{image_data_b64}"

    async def _upload_image(
        self, __request__: Request, user: UserModel, image_data: str, mime_type: str
    ) -> str:
        try:
            try:
                decoded_data = base64.b64decode(image_data)
            except Exception:
                try:
                    missing_padding = len(image_data) % 4
                    if missing_padding:
                        image_data += "=" * (4 - missing_padding)
                    decoded_data = base64.b64decode(image_data)
                except Exception:
                    return f"data:{mime_type};base64,{image_data}"

            bio = io.BytesIO(decoded_data)
            bio.seek(0)

            extension = "png"
            if "jpeg" in mime_type or "jpg" in mime_type:
                extension = "jpg"
            elif "webp" in mime_type:
                extension = "webp"
            elif "gif" in mime_type:
                extension = "gif"

            filename = f"gemini-generated-{uuid.uuid4().hex}.{extension}"

            async with get_async_db_context() as db:
                up_obj = await upload_file(
                    request=__request__,
                    background_tasks=BackgroundTasks(),
                    file=UploadFile(
                        file=bio,
                        filename=filename,
                        headers=Headers({"content-type": mime_type}),
                    ),
                    process=False,
                    user=user,
                    metadata={
                        "mime_type": mime_type,
                        "source": "gemini_image_generation",
                    },
                    db=db,
                )

            return __request__.app.url_path_for("get_file_content_by_id", id=up_obj.id)

        except Exception:
            return f"data:{mime_type};base64,{image_data}"

    async def _upload_video(
        self,
        __request__: Request,
        user: UserModel,
        video_data: bytes,
        mime_type: str = "video/mp4",
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        bio = io.BytesIO(video_data)
        bio.seek(0)

        extension = "mp4"
        if "webm" in mime_type:
            extension = "webm"

        filename = f"veo-generated-{uuid.uuid4().hex}.{extension}"

        async with get_async_db_context() as db:
            up_obj = await upload_file(
                request=__request__,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=bio,
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                process=False,
                user=user,
                metadata={"mime_type": mime_type, "source": "veo_video_generation"},
                db=db,
            )

            if chat_id and message_id:
                try:
                    await Chats.insert_chat_files(
                        chat_id=chat_id,
                        message_id=message_id,
                        file_ids=[up_obj.id],
                        user_id=user.id,
                        db=db,
                    )
                except Exception:
                    pass

        content_url = str(
            __request__.app.url_path_for("get_file_content_by_id", id=up_obj.id)
        )
        return content_url, self._build_generated_video_file(
            file_id=up_obj.id,
            content_url=content_url,
            filename=filename,
            mime_type=mime_type,
            size=len(video_data),
        )

    async def _upload_video_with_status(
        self,
        video_data: bytes,
        mime_type: str,
        __request__: Request,
        __user__: dict,
        __event_emitter__: Callable,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "video_upload",
                        "description": "Uploading generated video to your library...",
                        "done": False,
                    },
                }
            )

            self.user = user = await Users.get_user_by_id(__user__["id"])
            chat_id = __metadata__.get("chat_id") if __metadata__ else None
            message_id = __metadata__.get("message_id") if __metadata__ else None
            video_url, file_entry = await self._upload_video(
                __request__=__request__,
                user=user,
                video_data=video_data,
                mime_type=mime_type,
                chat_id=chat_id,
                message_id=message_id,
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "video_upload",
                        "description": "Video uploaded successfully!",
                        "done": True,
                    },
                }
            )
            return file_entry, video_url

        except Exception:
            video_data_b64 = base64.b64encode(video_data).decode("utf-8")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "video_upload",
                        "description": "Using inline video (upload failed)",
                        "done": True,
                    },
                }
            )
            return None, f"data:{mime_type};base64,{video_data_b64}"

    def _get_user_valve_value(
        self, __user__: Optional[dict], valve_name: str
    ) -> Optional[str]:
        if __user__ and "valves" in __user__:
            value = getattr(__user__["valves"], valve_name, None)
            if value and value != "default":
                return value
        return None

    def _configure_generation(
        self,
        body: Dict[str, Any],
        system_instruction: Optional[str],
        __metadata__: Dict[str, Any],
        __tools__: dict[str, Any] | None = None,
        __user__: Optional[dict] = None,
        enable_image_generation: bool = False,
        model_id: str = "",
    ) -> types.GenerateContentConfig:
        gen_config_params = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_output_tokens": body.get("max_tokens"),
            "stop_sequences": body.get("stop") or None,
            "system_instruction": system_instruction,
        }

        if enable_image_generation:
            gen_config_params["response_modalities"] = ["TEXT", "IMAGE"]

            if self._check_image_config_support(model_id):
                user_aspect_ratio = self._get_user_valve_value(
                    __user__, "IMAGE_GENERATION_ASPECT_RATIO"
                )
                aspect_ratio = body.get(
                    "aspect_ratio",
                    user_aspect_ratio or self.valves.IMAGE_GENERATION_ASPECT_RATIO,
                )

                user_resolution = self._get_user_valve_value(
                    __user__, "IMAGE_GENERATION_RESOLUTION"
                )
                resolution = body.get(
                    "resolution",
                    user_resolution or self.valves.IMAGE_GENERATION_RESOLUTION,
                )

                validated_aspect_ratio = self._validate_aspect_ratio(aspect_ratio)
                validated_resolution = self._validate_resolution(resolution)

                if validated_aspect_ratio or validated_resolution:
                    try:
                        image_config_params = {}
                        if validated_aspect_ratio:
                            image_config_params["aspect_ratio"] = validated_aspect_ratio
                        if validated_resolution:
                            image_config_params["image_size"] = validated_resolution
                        gen_config_params["image_config"] = types.ImageConfig(
                            **image_config_params
                        )
                    except Exception:
                        pass

        if self._check_thinking_support(model_id):
            try:
                thinking_config_params: Dict[str, Any] = {}

                include_thoughts = body.get("include_thoughts", True)
                if not self.valves.INCLUDE_THOUGHTS:
                    include_thoughts = False
                thinking_config_params["include_thoughts"] = include_thoughts

                if self._check_thinking_level_support(model_id):
                    reasoning_effort = body.get("reasoning_effort")
                    validated_level = None

                    if reasoning_effort:
                        validated_level = self._validate_thinking_level(
                            reasoning_effort, model_id
                        )

                    if not validated_level:
                        validated_level = self._validate_thinking_level(
                            self.valves.THINKING_LEVEL, model_id
                        )

                    if validated_level:
                        thinking_config_params["thinking_level"] = validated_level
                else:
                    body_thinking_budget = body.get("thinking_budget")
                    validated_budget = None

                    if body_thinking_budget is not None:
                        validated_budget = self._validate_thinking_budget(
                            body_thinking_budget
                        )

                    if validated_budget is None:
                        validated_budget = self._validate_thinking_budget(
                            self.valves.THINKING_BUDGET
                        )

                    if validated_budget == 0:
                        thinking_config_params["thinking_budget"] = 0
                    elif validated_budget is not None and validated_budget > 0:
                        thinking_config_params["thinking_budget"] = validated_budget
                    else:
                        thinking_config_params["thinking_budget"] = -1

                gen_config_params["thinking_config"] = types.ThinkingConfig(
                    **thinking_config_params
                )
            except Exception:
                pass

        if self.valves.USE_PERMISSIVE_SAFETY:
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
                ),
            ]
            gen_config_params |= {"safety_settings": safety_settings}

        features = __metadata__.get("features", {})
        params = __metadata__.get("params", {})
        tools = []

        if features.get("google_search_tool", False):
            if self.valves.USE_ENTERPRISE_WEB_SEARCH:
                tools.append(
                    types.Tool(enterprise_web_search=types.EnterpriseWebSearch())
                )
            else:
                tools.append(types.Tool(google_search=types.GoogleSearch()))
            tools.append(types.Tool(url_context=types.UrlContext()))

        if features.get("vertex_ai_search", False) or (
            self.valves.USE_VERTEX_AI
            and (self.valves.VERTEX_AI_RAG_STORE or os.getenv("VERTEX_AI_RAG_STORE"))
        ):
            vertex_rag_store = (
                params.get("vertex_rag_store")
                or self.valves.VERTEX_AI_RAG_STORE
                or os.getenv("VERTEX_AI_RAG_STORE")
            )
            if vertex_rag_store:
                tools.append(
                    types.Tool(
                        retrieval=types.Retrieval(
                            vertex_ai_search=types.VertexAISearch(
                                datastore=vertex_rag_store
                            )
                        )
                    )
                )

        function_declarations: List[types.FunctionDeclaration] = []
        native_tool_specs: Dict[str, Any] = {}

        def _clean_tool_schema(schema: Any) -> Any:
            if isinstance(schema, dict):
                allowed_keys = {
                    "type",
                    "description",
                    "properties",
                    "required",
                    "items",
                    "enum",
                }
                cleaned = {}
                for k, v in schema.items():
                    if k not in allowed_keys:
                        continue
                    cleaned[k] = _clean_tool_schema(v)

                properties = cleaned.get("properties")
                required = cleaned.get("required")
                if isinstance(properties, dict) and isinstance(required, list):
                    cleaned["required"] = [r for r in required if r in properties]
                return cleaned
            if isinstance(schema, list):
                return [_clean_tool_schema(v) for v in schema]
            return schema

        if __tools__ and self.valves.NATIVE_TOOL_CALLING_ENABLED:
            # FIX: OpenWebUI mengirimkan __tools__ sebagai DICT {tool_id: {"spec": ..., "callable": ...}}
            # bukan list objek. Iterasi .values() agar mendapatkan tool sebenarnya.
            tools_iterable = (
                __tools__.values() if isinstance(__tools__, dict) else __tools__
            )
            for tool in tools_iterable:
                # FIX: tool bisa berupa dict (OpenWebUI) atau objek dengan atribut .spec
                spec = (
                    tool.get("spec")
                    if isinstance(tool, dict)
                    else getattr(tool, "spec", None)
                )
                if not spec:
                    continue
                name = spec.get("name")
                if not name:
                    continue
                if enable_image_generation and self._is_open_webui_image_tool(name):
                    continue
                try:
                    parameters = (
                        spec.get("parameters") or spec.get("parameters_schema") or {}
                    )
                    parameters = _clean_tool_schema(parameters)
                    function_declarations.append(
                        types.FunctionDeclaration(
                            name=name,
                            description=spec.get("description", ""),
                            parameters=parameters if parameters else None,
                        )
                    )
                    native_tool_specs[name] = tool
                except Exception:
                    continue

        if "tools" in body and isinstance(body.get("tools"), list):
            for tool in body["tools"]:
                if tool.get("type") == "function":
                    func_data = tool.get("function", {})
                    name = func_data.get("name")
                    if not name or name in native_tool_specs:
                        continue
                    if enable_image_generation and self._is_open_webui_image_tool(name):
                        continue
                    parameters = func_data.get("parameters")
                    parameters = _clean_tool_schema(parameters) if parameters else None
                    function_declarations.append(
                        types.FunctionDeclaration(
                            name=name,
                            description=func_data.get("description", ""),
                            parameters=parameters,
                        )
                    )
                    # FIX: Mode native OpenWebUI meneruskan spec tool via body["tools"],
                    # sedangkan callable-nya hanya tersedia di __tools__.
                    # Tautkan keduanya berdasarkan nama agar tool loop bisa mengeksekusinya.
                    if __tools__:
                        candidate_tools = (
                            __tools__.values()
                            if isinstance(__tools__, dict)
                            else __tools__
                        )
                        for candidate_tool in candidate_tools:
                            candidate_spec = (
                                candidate_tool.get("spec")
                                if isinstance(candidate_tool, dict)
                                else getattr(candidate_tool, "spec", None)
                            )
                            if candidate_spec and candidate_spec.get("name") == name:
                                native_tool_specs[name] = candidate_tool
                                break

        if function_declarations:
            cleaned_declarations = []
            for fd in function_declarations:
                try:
                    fd_dict = fd.model_dump(exclude_none=True)
                except AttributeError:
                    try:
                        fd_dict = fd.dict(exclude_none=True)
                    except AttributeError:
                        fd_dict = {
                            "name": getattr(fd, "name", ""),
                            "description": getattr(fd, "description", ""),
                            "parameters": getattr(fd, "parameters", None),
                        }

                def _clean_decl_schema(obj: Any) -> Any:
                    if isinstance(obj, dict):
                        out = {}
                        for k, v in obj.items():
                            if k in (
                                "additional_properties",
                                "additionalProperties",
                                "examples",
                                "$schema",
                                "default",
                                "title",
                            ):
                                continue
                            out[k] = _clean_decl_schema(v)
                        if "properties" in out and "required" in out:
                            props = out.get("properties")
                            req = out.get("required")
                            if isinstance(props, dict) and isinstance(req, list):
                                out["required"] = [r for r in req if r in props]
                        return out
                    if isinstance(obj, list):
                        return [_clean_decl_schema(v) for v in obj]
                    return obj

                params = fd_dict.get("parameters")
                if params:
                    fd_dict["parameters"] = _clean_decl_schema(params)
                cleaned_declarations.append(types.FunctionDeclaration(**fd_dict))

            tools.append(types.Tool(function_declarations=cleaned_declarations))
            self.log.debug(
                f"Registered {len(cleaned_declarations)} function declaration(s)"
            )

        if tools:
            gen_config_params["tools"] = tools

        gen_config = types.GenerateContentConfig(
            **{k: v for k, v in gen_config_params.items() if v is not None}
        )
        setattr(gen_config, "_openwebui_native_tools", native_tool_specs)
        return gen_config

    @staticmethod
    def _format_grounding_chunks_as_sources(
        grounding_chunks: list[types.GroundingChunk],
    ):
        formatted_sources = []
        for chunk in grounding_chunks:
            if hasattr(chunk, "retrieved_context") and chunk.retrieved_context:
                context = chunk.retrieved_context
                formatted_sources.append(
                    {
                        "source": {
                            "name": getattr(context, "title", None) or "Document",
                            "type": "vertex_ai_search",
                            "uri": getattr(context, "uri", None),
                        },
                        "document": [getattr(context, "chunk_text", None) or ""],
                        "metadata": [
                            {"source": getattr(context, "title", None) or "Document"}
                        ],
                    }
                )
            elif hasattr(chunk, "web") and chunk.web:
                context = chunk.web
                uri = context.uri
                title = context.title or "Source"

                formatted_sources.append(
                    {
                        "source": {
                            "name": title,
                            "type": "web_search_results",
                            "url": uri,
                        },
                        "document": ["Click the link to view the content."],
                        "metadata": [{"source": title}],
                    }
                )
        return formatted_sources

    def _get_native_tools_from_config(
        self, generation_config: types.GenerateContentConfig
    ) -> Dict[str, Any]:
        """Retrieve the native OpenWebUI tool callables stored on the config."""
        return getattr(generation_config, "_openwebui_native_tools", {}) or {}

    @staticmethod
    def _serialize_tool_result(result: Any) -> Any:
        """Best-effort serialize a tool result so Gemini can consume it."""
        if result is None:
            return {"result": None}
        if isinstance(result, bool):
            return {"result": result}
        if isinstance(result, (int, float)):
            return {"result": result}
        # FIX: string mungkin berupa JSON hasil tool OpenWebUI — coba parse ke dict/list.
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                parsed = result
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        if isinstance(result, list):
            return {"result": result}
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        try:
            return json.loads(json.dumps(result, default=str))
        except Exception:
            return {"result": str(result)}

    async def _execute_native_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        native_tools: Dict[str, Any],
        __event_emitter__: Callable,
        tool_call_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single native OpenWebUI tool by name and return a Gemini FunctionResponse."""
        tool = native_tools.get(tool_name)
        if not tool:
            return {
                "name": tool_name,
                "response": {
                    "error": f"Native tool '{tool_name}' is not available in this session."
                },
                "id": tool_call_id,
            }

        # FIX: tool bisa berupa dict (OpenWebUI: {"spec":..., "callable":...}) atau objek
        callable_ref = (
            tool.get("callable")
            if isinstance(tool, dict)
            else getattr(tool, "callable", None)
        )
        if not callable_ref:
            return {
                "name": tool_name,
                "response": {
                    "error": f"Native tool '{tool_name}' does not expose a callable."
                },
                "id": tool_call_id,
            }

        try:

            async def _invoke(fn, params):
                if inspect.iscoroutinefunction(fn) or asyncio.iscoroutinefunction(fn):
                    return await fn(params)
                maybe_coro = fn(params)
                if inspect.isawaitable(maybe_coro):
                    return await maybe_coro
                return maybe_coro

            try:
                # FIX: Konvensi OpenWebUI — tool dipanggil dengan keyword arguments,
                # jadi coba **args terlebih dahulu, fallback ke positional dict.
                result = await _invoke(callable_ref, **args)
            except TypeError as te:
                try:
                    result = await _invoke(callable_ref, args)
                except Exception:
                    raise te
        except Exception as e:
            return {
                "name": tool_name,
                "response": {"error": f"Tool execution failed: {e}"},
                "id": tool_call_id,
            }

        return {
            "name": tool_name,
            "response": self._serialize_tool_result(result),
            "id": tool_call_id,
        }

    @staticmethod
    def _normalize_function_args(args: Any) -> Dict[str, Any]:
        if args is None:
            return {}
        if isinstance(args, dict):
            return dict(args)
        if hasattr(args, "to_json_dict"):
            try:
                return dict(args.to_json_dict())
            except Exception:
                pass
        if hasattr(args, "fields") and isinstance(args.fields, dict):
            try:
                return {
                    k: Pipe._normalize_function_args(v) for k, v in args.fields.items()
                }
            except Exception:
                pass
        if hasattr(args, "items") and callable(getattr(args, "items", None)):
            try:
                return {k: Pipe._normalize_function_args(v) for k, v in args.items()}
            except Exception:
                pass
        if hasattr(args, "__iter__") and not isinstance(args, (str, bytes)):
            try:
                result = {}
                for key in args:
                    result[key] = Pipe._normalize_function_args(
                        getattr(args, key, None)
                    )
                return result
            except Exception:
                pass
        try:
            return json.loads(json.dumps(args, default=str))
        except Exception:
            return {"args": str(args)}

    @staticmethod
    def _build_function_call_part_with_signature(
        name: str,
        args: Any,
        signature: Optional[Any] = None,
        tool_call_id: Optional[str] = None,
    ) -> types.Part:
        normalized_args = Pipe._normalize_function_args(args)
        try:
            fc = types.FunctionCall(name=name, args=normalized_args)
        except Exception:
            fc = types.FunctionCall()
            fc.name = name
            fc.args = normalized_args

        part = types.Part()
        part.function_call = fc
        if signature:
            part.thought_signature = signature
        if tool_call_id:
            try:
                part.function_call.id = tool_call_id
            except Exception:
                pass

        return part

    async def _run_tool_loop(
        self,
        client: genai.Client,
        model_id: str,
        contents: List[Dict[str, Any]],
        generation_config: types.GenerateContentConfig,
        native_tools: Dict[str, Any],
        __event_emitter__: Callable,
        max_iterations: int = 10,
    ) -> Tuple[types.GenerateContentResponse, int]:
        """
        Run a Gemini function-calling loop using native OpenWebUI tools.
        """
        tool_iterations = 0
        current_config = generation_config

        for iteration in range(max_iterations):
            response = await self._retry_with_backoff(
                lambda: client.aio.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=current_config,
                )
            )

            candidate = getattr(response, "candidates", [None])[0]
            if not candidate:
                return response, tool_iterations

            parts = getattr(getattr(candidate, "content", None), "parts", []) or []
            function_call_parts = [
                part for part in parts if getattr(part, "function_call", None)
            ]

            # PERBAIKAN: Jika tidak ada pemanggilan tool lagi, pastikan ada teks balasan
            if not function_call_parts:
                has_text = any(getattr(p, "text", None) for p in parts)

                if not has_text and tool_iterations > 0:
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text="Berdasarkan hasil eksekusi fungsi di atas, berikan jawaban atau rangkuman lengkap kepada pengguna."
                                )
                            ],
                        )
                    )
                    response = await self._retry_with_backoff(
                        lambda: client.aio.models.generate_content(
                            model=model_id,
                            contents=contents,
                            config=current_config,
                        )
                    )
                return response, tool_iterations

            tool_iterations += 1
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "tool_calls",
                        "description": f"Executing {len(function_call_parts)} tool call(s)...",
                        "done": False,
                    },
                }
            )

            function_response_parts = []
            model_function_call_parts = []

            for part in function_call_parts:
                fc = part.function_call
                tool_name = getattr(fc, "name", "")
                args = self._normalize_function_args(getattr(fc, "args", {}) or {})
                # FIX Gemini 3.x: pertahankan id unik dari setiap function call.
                tool_call_id = getattr(fc, "id", None)
                fr = await self._execute_native_tool_call(
                    tool_name,
                    args,
                    native_tools,
                    __event_emitter__,
                    tool_call_id=tool_call_id,
                )
                # FIX Gemini 3.x: FunctionResponse harus menyertakan id yang sama.
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fr["name"],
                        response=fr["response"],
                    )
                )

                sig = getattr(part, "thought_signature", None) or getattr(
                    part, "thoughtSignature", None
                )

                model_function_call_parts.append(
                    self._build_function_call_part_with_signature(
                        name=tool_name,
                        args=args,
                        signature=sig,
                        tool_call_id=tool_call_id,
                    )
                )

            contents.append(
                types.Content(role="model", parts=model_function_call_parts)
            )
            contents.append(types.Content(role="user", parts=function_response_parts))

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "tool_calls",
                        "description": f"Tool call(s) executed ({tool_iterations})",
                        "done": True,
                    },
                }
            )

        return response, tool_iterations

    async def _process_grounding_metadata(
        self,
        grounding_metadata_list: List[types.GroundingMetadata],
        text: str,
        __event_emitter__: Callable,
    ):
        grounding_chunks = []
        web_search_queries = []
        grounding_supports = []

        for metadata in grounding_metadata_list:
            if metadata.grounding_chunks:
                grounding_chunks.extend(metadata.grounding_chunks)
            if metadata.web_search_queries:
                web_search_queries.extend(metadata.web_search_queries)
            if metadata.grounding_supports:
                grounding_supports.extend(metadata.grounding_supports)

        if grounding_chunks:
            sources = self._format_grounding_chunks_as_sources(grounding_chunks)
            for source in sources:
                await __event_emitter__({"type": "source", "data": source})

        if web_search_queries:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "This response was grounded with Google Search",
                        "urls": [
                            f"https://www.google.com/search?q={query}"
                            for query in web_search_queries
                        ],
                    },
                }
            )

        replaced_text: Optional[str] = None
        if grounding_supports:
            ENCODING = "utf-8"
            text_bytes = text.encode(ENCODING)
            last_byte_index = 0
            cited_chunks = []

            for support in grounding_supports:
                cited_chunks.append(
                    text_bytes[last_byte_index : support.segment.end_index].decode(
                        ENCODING
                    )
                )
                footnotes = "".join(
                    [f"[{i + 1}]" for i in support.grounding_chunk_indices]
                )
                cited_chunks.append(f" {footnotes}")
                last_byte_index = support.segment.end_index

            if last_byte_index < len(text_bytes):
                cited_chunks.append(text_bytes[last_byte_index:].decode(ENCODING))

            replaced_text = "".join(cited_chunks)

        return replaced_text if replaced_text is not None else text

    async def _handle_streaming_response(
        self,
        response_iterator: Any,
        __event_emitter__: Callable,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        async def emit_chat_event(event_type: str, data: Dict[str, Any]) -> None:
            if not __event_emitter__:
                return
            try:
                await __event_emitter__({"type": event_type, "data": data})
            except Exception:
                pass

        await emit_chat_event("chat:start", {"role": "assistant"})

        grounding_metadata_list = []
        answer_chunks: list[str] = []
        thought_chunks: list[str] = []
        thinking_started_at: Optional[float] = None
        stream_usage_metadata = None

        try:
            async for chunk in response_iterator:
                if getattr(chunk, "usage_metadata", None):
                    stream_usage_metadata = chunk.usage_metadata

                if not chunk.candidates:
                    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                        block_reason = chunk.prompt_feedback.block_reason.name
                        message = f"[Blocked due to Prompt Safety: {block_reason}]"
                        await emit_chat_event(
                            "chat:finish",
                            {
                                "role": "assistant",
                                "content": message,
                                "done": True,
                                "error": True,
                            },
                        )
                        yield message
                    else:
                        message = "[Blocked by safety settings]"
                        await emit_chat_event(
                            "chat:finish",
                            {
                                "role": "assistant",
                                "content": message,
                                "done": True,
                                "error": True,
                            },
                        )
                        yield message
                    return

                if chunk.candidates[0].grounding_metadata:
                    grounding_metadata_list.append(
                        chunk.candidates[0].grounding_metadata
                    )

                parts = []
                try:
                    parts = chunk.candidates[0].content.parts or []
                except Exception:
                    if hasattr(chunk, "text") and chunk.text:
                        answer_chunks.append(chunk.text)
                        await __event_emitter__(
                            {
                                "type": "chat:message:delta",
                                "data": {
                                    "role": "assistant",
                                    "content": chunk.text,
                                },
                            }
                        )
                    continue

                for part in parts:
                    try:
                        if getattr(part, "function_call", None):
                            fc_args = self._normalize_function_args(
                                getattr(part.function_call, "args", None)
                            )
                            tool_call = {
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(fc_args),
                                },
                            }
                            yield {"choices": [{"delta": {"tool_calls": [tool_call]}}]}
                            continue

                        if getattr(part, "thought", False) and getattr(
                            part, "text", None
                        ):
                            if thinking_started_at is None:
                                thinking_started_at = time.time()
                            thought_chunks.append(part.text)
                            preview = part.text.replace("\n", " ").strip()
                            MAX_PREVIEW = 120
                            if len(preview) > MAX_PREVIEW:
                                preview = preview[:MAX_PREVIEW].rstrip() + "…"
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "action": "thinking",
                                        "description": f"Thinking… {preview}",
                                        "done": False,
                                        "hidden": False,
                                    },
                                }
                            )

                        elif getattr(part, "text", None):
                            answer_chunks.append(part.text)
                            await __event_emitter__(
                                {
                                    "type": "chat:message:delta",
                                    "data": {
                                        "role": "assistant",
                                        "content": part.text,
                                    },
                                }
                            )
                    except Exception:
                        continue

            final_answer_text = "".join(answer_chunks)
            if grounding_metadata_list and __event_emitter__:
                cited = await self._process_grounding_metadata(
                    grounding_metadata_list,
                    final_answer_text,
                    __event_emitter__,
                )
                final_answer_text = cited or final_answer_text

            final_content = final_answer_text
            details_block: Optional[str] = None

            if thought_chunks:
                duration_s = int(
                    max(0, time.time() - (thinking_started_at or time.time()))
                )
                thought_content = "".join(thought_chunks).strip()
                quoted_lines = []
                for line in thought_content.split("\n"):
                    quoted_lines.append(f"> {line}")
                quoted_content = "\n".join(quoted_lines)

                details_block = f"""<details>
<summary>Thought ({duration_s}s)</summary>

{quoted_content}

</details>""".strip()
                final_content = f"{details_block}{final_answer_text}"

            if not final_content:
                final_content = ""

            await emit_chat_event(
                "replace", {"role": "assistant", "content": final_content}
            )
            await emit_chat_event(
                "chat:message",
                {"role": "assistant", "content": final_content, "done": True},
            )

            if thought_chunks:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"action": "thinking", "done": True, "hidden": True},
                    }
                )

            usage = self._build_usage_dict(stream_usage_metadata)
            if usage:
                yield {"usage": usage}

            await emit_chat_event(
                "chat:finish",
                {"role": "assistant", "content": final_content, "done": True},
            )

            yield final_content

        except (asyncio.CancelledError, Exception) as e:
            error_msg = str(e).lower()
            if "connection closed" in error_msg or "cancelled" in error_msg:
                if answer_chunks:
                    final_content = "".join(answer_chunks)
                    await emit_chat_event(
                        "chat:finish",
                        {"role": "assistant", "content": final_content, "done": True},
                    )
                    yield final_content
                    return
                message = "Koneksi ke Gemini terputus di tengah jalan. Silakan coba gunakan versi API v1beta atau matikan mode streaming."
            elif "chunk too big" in error_msg or "chunk size" in error_msg:
                message = "Error: Image too large for processing. Please try with a smaller image (max 15 MB recommended) or reduce image quality."
            elif "quota" in error_msg or "rate limit" in error_msg:
                message = "Error: API quota exceeded. Please try again later."
            else:
                message = f"Error during streaming: {e}"
            await emit_chat_event(
                "chat:finish",
                {
                    "role": "assistant",
                    "content": message,
                    "done": True,
                    "error": True,
                },
            )
            yield message

    @staticmethod
    def _build_usage_dict(usage_metadata: Any) -> Optional[Dict[str, int]]:
        if not usage_metadata:
            return None
        usage: Dict[str, int] = {}
        if getattr(usage_metadata, "prompt_token_count", None) is not None:
            usage["prompt_tokens"] = usage_metadata.prompt_token_count
        if getattr(usage_metadata, "candidates_token_count", None) is not None:
            usage["completion_tokens"] = usage_metadata.candidates_token_count
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
            return usage
        return None

    def _get_safety_block_message(self, response: Any) -> Optional[str]:
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"[Blocked due to Prompt Safety: {response.prompt_feedback.block_reason.name}]"

        if not response.candidates:
            return "[Blocked by safety settings or no candidates generated]"

        candidate = response.candidates[0]
        if candidate.finish_reason == types.FinishReason.SAFETY:
            blocking_rating = next(
                (r for r in candidate.safety_ratings if r.blocked), None
            )
            reason = f" ({blocking_rating.category.name})" if blocking_rating else ""
            return f"[Blocked by safety settings{reason}]"
        elif candidate.finish_reason == types.FinishReason.PROHIBITED_CONTENT:
            return "[Content blocked due to prohibited content policy violation]"

        return None

    async def _generate_video(
        self,
        body: Dict[str, Any],
        model_id: str,
        __event_emitter__: Callable,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        async def emit_status(description: str, done: bool) -> None:
            if not __event_emitter__:
                return
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "action": "video_generation",
                            "description": description,
                            "done": done,
                        },
                    }
                )
            except Exception:
                pass

        messages = body.get("messages", [])
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_msg:
            return "Error: No user message found for video generation"

        prompt, images = await self._extract_images_from_message(last_user_msg)
        if not prompt:
            return "Error: No prompt provided for video generation"

        reference_image = None
        if images:
            first_img = images[0]
            try:
                img_data = first_img.get("inline_data", {})
                raw_data = img_data.get("data", "")
                img_bytes = base64.b64decode(raw_data)
                reference_image = types.Image(
                    image_bytes=img_bytes,
                    mime_type=img_data.get("mime_type", "image/png"),
                )
            except Exception:
                pass

        config = self._build_video_generation_config(body, __user__, model_id=model_id)
        await emit_status(f"Starting video generation with {model_id}...", False)

        client = self._get_client()
        try:
            generate_kwargs: Dict[str, Any] = {
                "model": model_id,
                "prompt": prompt,
                "config": config,
            }
            if reference_image:
                generate_kwargs["image"] = reference_image
            operation = await client.aio.models.generate_videos(**generate_kwargs)
        except Exception as e:
            await emit_status(f"Video generation failed: {e}", True)
            return f"Error starting video generation: {e}"

        poll_interval = max(self.valves.VIDEO_POLL_INTERVAL, 5)
        poll_timeout = max(self.valves.VIDEO_POLL_TIMEOUT, 0)
        elapsed = 0
        while not operation.done:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if poll_timeout > 0 and elapsed >= poll_timeout:
                error_msg = f"Video generation timed out after {elapsed}s (limit: {poll_timeout}s)"
                await emit_status(error_msg, True)
                return f"Error: {error_msg}"
            try:
                operation = await client.aio.operations.get(operation)
            except Exception:
                pass
            await emit_status(f"Generating video... ({elapsed}s elapsed)", False)

        if operation.error:
            error_msg = str(operation.error)
            await emit_status(f"Video generation failed: {error_msg}", True)
            return f"Video generation failed: {error_msg}"

        generated_video_files: List[Dict[str, Any]] = []
        generated_video_links: List[str] = []
        upload_failure_count = 0
        attachment_skipped_count = 0
        response = operation.response
        if not response or not response.generated_videos:
            return "Error: No videos were generated"

        for idx, gen_video in enumerate(response.generated_videos):
            video = gen_video.video
            if not video:
                continue

            video_bytes = None
            if getattr(video, "video_bytes", None):
                video_bytes = video.video_bytes

            if not video_bytes:
                try:
                    await asyncio.to_thread(client.files.download, file=video)
                    video_bytes = getattr(video, "video_bytes", None)
                except Exception:
                    pass

            if not video_bytes:
                tmp_path = None
                try:
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".mp4", delete=False
                    ) as tmp:
                        tmp_path = tmp.name
                    await asyncio.to_thread(video.save, tmp_path)
                    async with aiofiles.open(tmp_path, "rb") as f:
                        video_bytes = await f.read()
                except Exception:
                    pass
                finally:
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            if not video_bytes:
                continue

            mime_type = getattr(video, "mime_type", "video/mp4") or "video/mp4"

            file_entry = None
            video_url = None
            attachment_attempted = False
            if __request__ and __user__:
                attachment_attempted = True
                file_entry, video_url = await self._upload_video_with_status(
                    video_bytes,
                    mime_type,
                    __request__,
                    __user__,
                    __event_emitter__,
                    __metadata__,
                )
            else:
                video_data_b64 = base64.b64encode(video_bytes).decode("utf-8")
                video_url = f"data:{mime_type};base64,{video_data_b64}"

            if file_entry:
                generated_video_files.append(file_entry)
                if video_url:
                    generated_video_links.append(
                        f"[\U0001f3ac Generated Video {idx + 1}]({video_url})"
                    )
                continue

            if attachment_attempted:
                upload_failure_count += 1
            else:
                attachment_skipped_count += 1

            if attachment_attempted and video_url and not video_url.startswith("data:"):
                generated_video_links.append(
                    f"[\U0001f3ac Generated Video {idx + 1}]({video_url})"
                )
            elif attachment_attempted:
                generated_video_links.append(
                    f"Generated video {idx + 1}, but it could not be attached to the chat."
                )
            else:
                generated_video_links.append(f"Generated video {idx + 1}.")

        await emit_status(f"Video generation complete ({elapsed}s)", True)

        files_emitted = await self._emit_generated_video_files(
            generated_video_files, __event_emitter__
        )

        content_parts: List[str] = []
        if generated_video_files and files_emitted:
            video_count = len(generated_video_files)
            content_parts.append(
                "Generated video attached."
                if video_count == 1
                else f"Generated {video_count} videos attached."
            )
        else:
            content_parts.extend(generated_video_links)

        if generated_video_files and not files_emitted:
            content_parts.extend(generated_video_links)

        if upload_failure_count:
            content_parts.append("Some videos could not be attached directly.")

        if attachment_skipped_count:
            content_parts.append(
                "Video attachments were skipped because chat upload context was unavailable."
            )

        content = (
            "\n\n".join(part for part in content_parts if part)
            if content_parts
            else "[No video content generated]"
        )

        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
        }

    async def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        max_retries = self.valves.RETRY_COUNT
        retry_count = 0
        last_exception = None

        while retry_count <= max_retries:
            try:
                return await func(*args, **kwargs)
            except ServerError as e:
                retry_count += 1
                last_exception = e
                if retry_count <= max_retries:
                    wait_time = min(2**retry_count + (0.1 * retry_count), 10)
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception:
                raise
        assert last_exception is not None
        raise last_exception

    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: dict[str, Any],
        __event_emitter__: Callable,
        __tools__: dict[str, Any] | None,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> Union[str, Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        if __user__:
            self.user = await Users.get_user_by_id(__user__["id"])
        else:
            self.user = None

        try:
            model_id = body.get("model", "")
            try:
                model_id = self._prepare_model_id(model_id)
            except ValueError as ve:
                return f"Model Error: {ve}"

            if self._check_video_generation_support(model_id):
                return await self._generate_video(
                    body,
                    model_id,
                    __event_emitter__,
                    __request__,
                    __user__,
                    __metadata__,
                )

            supports_image_generation = self._check_image_generation_support(model_id)

            stream = body.get("stream", False)
            if not self.valves.STREAMING_ENABLED:
                stream = False
            messages = body.get("messages", [])

            if supports_image_generation:
                try:
                    contents, system_instruction = (
                        await self._build_image_generation_contents(
                            messages, __event_emitter__
                        )
                    )
                except ValueError as ve:
                    return f"Error: {ve}"
            else:
                contents, system_instruction = self._prepare_content(messages)
                if not contents:
                    return "Error: No valid message content found"

            generation_config = self._configure_generation(
                body,
                system_instruction,
                __metadata__,
                __tools__,
                __user__,
                supports_image_generation,
                model_id,
            )

            client = self._get_client()

            native_tools = self._get_native_tools_from_config(generation_config)
            has_native_tools = (
                bool(native_tools) and self.valves.NATIVE_TOOL_CALLING_ENABLED
            )
            if has_native_tools and stream and not supports_image_generation:
                stream = False
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "action": "tool_mode",
                            "description": "Native tools detected; switching to non-streaming mode for server-side execution.",
                            "done": True,
                        },
                    }
                )

            if stream:
                if supports_image_generation:
                    stream = False
                elif has_native_tools:
                    stream = False
                else:
                    try:

                        async def get_streaming_response():
                            return await client.aio.models.generate_content_stream(
                                model=model_id,
                                contents=contents,
                                config=generation_config,
                            )

                        response_iterator = await self._retry_with_backoff(
                            get_streaming_response
                        )
                        return self._handle_streaming_response(
                            response_iterator, __event_emitter__, __request__, __user__
                        )
                    except Exception as e:
                        return f"Error during streaming: {e}"

            if not stream or supports_image_generation:
                try:

                    async def get_response():
                        return await client.aio.models.generate_content(
                            model=model_id,
                            contents=contents,
                            config=generation_config,
                        )

                    start_ts = time.time()
                    if supports_image_generation:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing image request...",
                                    "done": False,
                                },
                            }
                        )

                    if has_native_tools and not supports_image_generation:
                        response, _ = await self._run_tool_loop(
                            client,
                            model_id,
                            contents,
                            generation_config,
                            native_tools,
                            __event_emitter__,
                        )
                    else:
                        response = await self._retry_with_backoff(get_response)

                    if supports_image_generation:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing complete",
                                    "done": True,
                                },
                            }
                        )

                    safety_message = self._get_safety_block_message(response)
                    if safety_message:
                        return safety_message

                    candidate = response.candidates[0]
                    parts = getattr(getattr(candidate, "content", None), "parts", [])
                    if not parts:
                        return "[No content generated or unexpected response structure]"

                    answer_segments: list[str] = []
                    thought_segments: list[str] = []
                    generated_images: list[str] = []
                    generated_image_files: List[Dict[str, Any]] = []
                    seen_generated_image_hashes: set[str] = set()
                    tool_calls = []

                    for part in parts:
                        if getattr(part, "function_call", None):
                            fc_args = self._normalize_function_args(
                                getattr(part.function_call, "args", None)
                            )
                            tool_calls.append(
                                {
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": json.dumps(fc_args),
                                    },
                                }
                            )
                        elif getattr(part, "thought", False) and getattr(
                            part, "text", None
                        ):
                            thought_segments.append(part.text)
                        elif getattr(part, "text", None):
                            answer_segments.append(part.text)
                        elif (
                            getattr(part, "inline_data", None)
                            and __request__
                            and __user__
                        ):
                            mime_type = part.inline_data.mime_type
                            image_data = part.inline_data.data

                            image_hash = self._image_data_hash(image_data)
                            if image_hash in seen_generated_image_hashes:
                                continue
                            seen_generated_image_hashes.add(image_hash)

                            image_url = await self._upload_image_with_status(
                                image_data,
                                mime_type,
                                __request__,
                                __user__,
                                __event_emitter__,
                            )
                            if image_url.startswith("data:"):
                                generated_images.append(
                                    f"![Generated Image]({image_url})"
                                )
                            else:
                                generated_image_files.append(
                                    self._build_generated_image_file(
                                        content_url=image_url,
                                        mime_type=mime_type,
                                    )
                                )
                        elif getattr(part, "inline_data", None):
                            mime_type = part.inline_data.mime_type
                            image_data = part.inline_data.data
                            image_hash = self._image_data_hash(image_data)
                            if image_hash in seen_generated_image_hashes:
                                continue
                            seen_generated_image_hashes.add(image_hash)

                            if isinstance(image_data, bytes):
                                image_data_b64 = base64.b64encode(image_data).decode(
                                    "utf-8"
                                )
                            else:
                                image_data_b64 = str(image_data)
                            data_url = f"data:{mime_type};base64,{image_data_b64}"
                            generated_images.append(f"![Generated Image]({data_url})")

                    final_answer = "".join(answer_segments)

                    grounding_metadata_list = []
                    if getattr(candidate, "grounding_metadata", None):
                        grounding_metadata_list.append(candidate.grounding_metadata)
                    if grounding_metadata_list:
                        cited = await self._process_grounding_metadata(
                            grounding_metadata_list,
                            final_answer,
                            __event_emitter__,
                        )
                        final_answer = cited or final_answer

                    full_response = ""
                    if thought_segments:
                        duration_s = int(max(0, time.time() - start_ts))
                        thought_content = "".join(thought_segments).strip()
                        quoted_lines = []
                        for line in thought_content.split("\n"):
                            quoted_lines.append(f"> {line}")
                        quoted_content = "\n".join(quoted_lines)
                        details_block = f"""<details>
<summary>Thought ({duration_s}s)</summary>

{quoted_content}

</details>""".strip()
                        full_response += details_block

                    full_response += final_answer

                    files_emitted = await self._emit_generated_image_files(
                        generated_image_files, __event_emitter__
                    )

                    if generated_image_files and not files_emitted:
                        generated_images.extend(
                            f"![Generated Image]({image_file['url']})"
                            for image_file in generated_image_files
                        )

                    if (
                        generated_image_files
                        and files_emitted
                        and not final_answer.strip()
                    ):
                        if full_response:
                            full_response += "\n\n"
                        full_response += "Generated image."

                    if generated_images:
                        if full_response:
                            full_response += "\n\n"
                        full_response += "\n\n".join(generated_images)

                    content = full_response.strip()

                    if not content:
                        if has_native_tools:
                            content = "Proses eksekusi fungsi telah selesai."
                        else:
                            content = (
                                "[Tidak ada respon teks yang dihasilkan oleh model]"
                            )

                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "chat:message",
                                "data": {
                                    "role": "assistant",
                                    "content": content,
                                    "done": True,
                                },
                            }
                        )

                    usage = self._build_usage_dict(
                        getattr(response, "usage_metadata", None)
                    )

                    message_data = {"role": "assistant", "content": content}

                    if tool_calls:
                        message_data["tool_calls"] = tool_calls

                    result = {
                        "choices": [{"message": message_data}],
                    }
                    if usage:
                        result["usage"] = usage

                    return content

                except Exception as e:
                    return f"Error generating content: {e}"

        except (ClientError, ServerError, APIError) as api_error:
            error_type = type(api_error).__name__
            return f"{error_type}: {api_error}"
        except ValueError as ve:
            return f"Configuration error: {ve}"
        except Exception as e:
            return f"An error occurred while processing your request: {e}"

