"""
title: Google Gemini Image Pipeline
author: owndev, olivier-lacroix (REST edition by dfirmans)
version: 2.2.0
required_open_webui_version: 0.11.4
description: Image generation and editing for Gemini image models via the REST API (no extra packages needed - uses only Open WebUI built-ins). Text-to-image and image-to-image with aspect ratio, resolution, history context, and automatic upload to Open WebUI. API-key mode only. Supports the full Nano Banana family (2.1, 2, 2 Lite, Pro) plus Nano Banana.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastapi import Request
from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:
    SRC_LOG_LEVELS = {}


def _http_client():
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(f"httpx is required but unavailable: {e}")
    try:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0), follow_redirects=True
        )
    except Exception:
        return httpx.AsyncClient(follow_redirects=True)


def _api_error(action: str, resp: Any) -> RuntimeError:
    try:
        data = resp.json()
        err = data.get("error", {}) if isinstance(data, dict) else {}
        msg = err.get("message") or resp.text
        code = err.get("code") or resp.status_code
        status = err.get("status") or ""
        return RuntimeError(f"Gemini API error during {action} ({code} {status}): {msg}")
    except Exception:
        return RuntimeError(f"Gemini API error during {action}: HTTP {resp.status_code}")


ASPECT_RATIO_OPTIONS = [
    "default", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]
RESOLUTION_OPTIONS = ["default", "1K", "2K", "4K"]

IMAGE_GENERATION_MODELS = [
    "gemini-2.5-flash-image",        # Nano Banana
    "gemini-3.1-flash-image",        # Nano Banana 2
    "gemini-3.1-flash-lite-image",   # Nano Banana 2 Lite
    "gemini-3-pro-image",            # Nano Banana Pro
    "gemini-nano-banana-2.1",        # Nano Banana 2.1 (latest)
]


class Pipe:
    """Image generation / editing pipeline for Gemini image models (pure REST)."""

    class Valves(BaseModel):
        BASE_URL: str = Field(
            default=os.getenv(
                "GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/"
            ),
            description="Base URL for the Google Generative AI REST API.",
        )
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI.",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_VERSION: str = Field(
            default=os.getenv("GOOGLE_API_VERSION", "v1beta"),
            description="API version (e.g. v1beta, v1).",
        )
        IMAGE_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_ASPECT_RATIO", "default"),
            description="Aspect ratio for image generation (Gemini 3 image models).",
            json_schema_extra={"enum": ASPECT_RATIO_OPTIONS},
        )
        IMAGE_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_RESOLUTION", "default"),
            description="Resolution for image generation: 1K, 2K, 4K (Gemini 3 image models).",
            json_schema_extra={"enum": RESOLUTION_OPTIONS},
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("GOOGLE_USE_PERMISSIVE_SAFETY", "false").lower() == "true",
            description="Use permissive safety settings.",
        )
        MODEL_CACHE_TTL: int = Field(
            default=int(os.getenv("GOOGLE_MODEL_CACHE_TTL", "600")),
            description="Seconds to cache the model list.",
        )
        RETRY_COUNT: int = Field(
            default=int(os.getenv("GOOGLE_RETRY_COUNT", "2")),
            description="Retries for temporary API failures (5xx/429).",
        )
        DEFAULT_SYSTEM_PROMPT: str = Field(
            default=os.getenv("GOOGLE_DEFAULT_SYSTEM_PROMPT", ""),
            description="Default system prompt prepended to the image prompt.",
        )
        MODEL_ADDITIONAL: str = Field(
            default=os.getenv("GOOGLE_MODEL_ADDITIONAL", ""),
            description="Comma-separated image model IDs to add manually.",
        )
        MODEL_WHITELIST: str = Field(
            default=os.getenv("GOOGLE_MODEL_WHITELIST", ""),
            description="Comma-separated model IDs to show. Empty = all image models.",
        )
        IMAGE_MAX_SIZE_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_MAX_SIZE_MB", "15.0")),
            description="Max image size in MB before compression.",
        )
        IMAGE_MAX_DIMENSION: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_MAX_DIMENSION", "2048")),
            description="Max width/height in pixels before resizing.",
        )
        IMAGE_COMPRESSION_QUALITY: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_COMPRESSION_QUALITY", "85")),
            description="JPEG quality for compression (1-100).",
        )
        IMAGE_ENABLE_OPTIMIZATION: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ENABLE_OPTIMIZATION", "true").lower() == "true",
            description="Enable image optimization (needs Pillow, otherwise skipped).",
        )
        IMAGE_PNG_THRESHOLD_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_PNG_THRESHOLD_MB", "0.5")),
            description="PNG files above this size (MB) are converted to JPEG.",
        )
        IMAGE_HISTORY_MAX_REFERENCES: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_HISTORY_MAX_REFERENCES", "5")),
            description="Max total images (history + current) sent in one call.",
        )
        IMAGE_ADD_LABELS: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ADD_LABELS", "true").lower() == "true",
            description="Prefix each image with a label like [Image 1] so the model can reference it.",
        )
        IMAGE_DEDUP_HISTORY: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_DEDUP_HISTORY", "true").lower() == "true",
            description="Deduplicate identical images (by hash) in history context.",
        )
        IMAGE_HISTORY_FIRST: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_HISTORY_FIRST", "true").lower() == "true",
            description="Place history images before current-message images.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Google Gemini Image"
        self.log = logging.getLogger("google_gemini_image.pipe")
        try:
            self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        except Exception:
            self.log.setLevel(logging.INFO)
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0

    # ---------------- REST plumbing ---------------- #
    def _validate(self) -> str:
        key = (self.valves.GOOGLE_API_KEY or "").strip()
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Open the function Valves and paste your Gemini API key."
            )
        return key

    def _url(self, *parts: str) -> str:
        base = (self.valves.BASE_URL or "https://generativelanguage.googleapis.com/").rstrip("/")
        return f"{base}/{self.valves.API_VERSION}/" + "/".join(parts)

    def _headers(self, key: str) -> Dict[str, str]:
        return {"x-goog-api-key": key, "Content-Type": "application/json"}

    async def _post(self, url: str, payload: Dict[str, Any], action: str) -> Dict[str, Any]:
        key = self._validate()
        last: Exception | None = None
        for attempt in range(self.valves.RETRY_COUNT + 1):
            try:
                async with _http_client() as client:
                    resp = await client.post(url, headers=self._headers(key), json=payload)
                if resp.status_code in (429,) or resp.status_code >= 500:
                    last = _api_error(action, resp)
                    await asyncio.sleep(min(2**attempt + 0.1 * attempt, 10))
                    continue
                if resp.status_code >= 400:
                    raise _api_error(action, resp)
                data = resp.json()
                return data if isinstance(data, dict) else {}
            except RuntimeError:
                raise
            except Exception as e:
                last = e
                await asyncio.sleep(min(2**attempt + 0.1 * attempt, 10))
        raise last if last else RuntimeError(f"Request failed: {action}")

    # ---------------- Models ---------------- #
    @staticmethod
    def _is_image_model(model_id: str) -> bool:
        mid = model_id.lower()
        for pattern in IMAGE_GENERATION_MODELS:
            if mid == pattern or pattern in mid:
                return True
        # Any Gemini model whose id mentions "image" is an image model
        # (e.g. gemini-3-pro-image, gemini-3.1-flash-lite-image),
        # with or without -preview.
        if mid.startswith("gemini-") and "image" in mid:
            return True
        # The Nano Banana family (e.g. gemini-nano-banana-2.1) are image models
        # even though "image" is not part of the id. This also catches future
        # versions automatically when listed by the models.list API.
        return mid.startswith("gemini-nano-banana")

    @staticmethod
    def _is_gemini_3_image_model(model_id: str) -> bool:
        if not Pipe._is_image_model(model_id):
            return False
        mid = model_id.lower()
        # Gemini 3.x image models (gemini-3-pro-image, gemini-3.1-flash-image,
        # gemini-3.1-flash-lite-image) support imageConfig (aspectRatio/imageSize).
        if mid.startswith("gemini-3-") or mid.startswith("gemini-3."):
            return True
        # Nano Banana 2.1+ (gemini-nano-banana-2.1) also supports aspectRatio and
        # imageSize (1K / 2K / 4K) via imageConfig.
        return mid.startswith("gemini-nano-banana")

    def strip_prefix(self, model_name: str) -> str:
        stripped = model_name.rsplit("/", 1)[-1]
        if "." in stripped and not stripped.startswith("gemini-"):
            stripped = stripped.split(".", 1)[1]
        return stripped

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        now = time.time()
        if (
            not force_refresh
            and self._model_cache is not None
            and (now - self._model_cache_time) < self.valves.MODEL_CACHE_TTL
        ):
            return self._model_cache
        seen: Dict[str, str] = {mid: mid + " 🎨" for mid in IMAGE_GENERATION_MODELS}
        try:
            key = (self.valves.GOOGLE_API_KEY or "").strip()
            if key:
                import urllib.request

                req = urllib.request.Request(
                    self._url("models"), headers={"x-goog-api-key": key}
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read().decode("utf-8", "replace"))
                for m in data.get("models", []) if isinstance(data, dict) else []:
                    if not isinstance(m, dict):
                        continue
                    mid = self.strip_prefix(str(m.get("name", "")))
                    if self._is_image_model(mid):
                        seen[mid] = str(m.get("displayName") or mid) + " 🎨"
        except Exception as e:
            self.log.warning(f"models.list failed, using static list: {e}")
        for extra in re.findall(r"[^,\s]+", self.valves.MODEL_ADDITIONAL or ""):
            extra = self.strip_prefix(extra)
            if self._is_image_model(extra):
                seen.setdefault(extra, extra + " 🎨")
        items = [{"id": k, "name": v} for k, v in sorted(seen.items())]
        whitelist = set(re.findall(r"[^,\s]+", self.valves.MODEL_WHITELIST or ""))
        if whitelist:
            items = [m for m in items if m["id"] in whitelist]
        self._model_cache = items
        self._model_cache_time = now
        return items

    def pipes(self) -> List[Dict[str, str]]:
        try:
            self.name = "Google Gemini Image: "
            return self.get_google_models()
        except ValueError as e:
            return [{"id": "error", "name": str(e)}]
        except Exception as e:
            self.log.exception(f"pipes listing failed: {e}")
            return [{"id": "error", "name": f"Unexpected error: {e}"}]

    def _prepare_model_id(self, model_id: str) -> str:
        original = model_id
        model_id = self.strip_prefix(model_id)
        if not self._is_image_model(model_id):
            found = next(
                (m["id"] for m in self.get_google_models() if m["name"] == original),
                None,
            )
            if found:
                return found
            raise ValueError(
                f"Model '{original}' is not an image generation model. Use the Text or Video pipeline for it."
            )
        return model_id

    # ---------------- Input images ---------------- #
    def _optimize_image(self, image_data: str) -> str:
        if not self.valves.IMAGE_ENABLE_OPTIMIZATION:
            return image_data
        try:
            from PIL import Image as _Image
        except ImportError:
            return image_data
        try:
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
            else:
                encoded, mime = image_data, "image/png"
            raw = base64.b64decode(encoded)
            size_mb = len(raw) / (1024 * 1024)
            needs_work = size_mb > self.valves.IMAGE_MAX_SIZE_MB or (
                mime == "image/png" and size_mb > self.valves.IMAGE_PNG_THRESHOLD_MB
            )
            with _Image.open(io.BytesIO(raw)) as img:
                w, h = img.size
                if max(w, h) > self.valves.IMAGE_MAX_DIMENSION:
                    needs_work = True
                if not needs_work:
                    return image_data
                if img.mode in ("RGBA", "P", "LA"):
                    img = img.convert("RGB")
                scale = min(1.0, self.valves.IMAGE_MAX_DIMENSION / max(w, h))
                if scale < 1.0:
                    img = img.resize((int(w * scale), int(h * scale)), _Image.LANCZOS)
                buf = io.BytesIO()
                img.save("JPEG", buf, quality=self.valves.IMAGE_COMPRESSION_QUALITY, optimize=True)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception as e:
            self.log.warning(f"Image optimization skipped: {e}")
            return image_data

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
                file_path = Path(await asyncio.to_thread(Storage.get_file, file_obj.path))
                if file_path.is_file():
                    raw = await asyncio.to_thread(file_path.read_bytes)
                    mime = (file_obj.meta or {}).get("content_type", "image/png")
                    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        except Exception as e:
            self.log.warning(f"Could not fetch file {file_url}: {e}")
        return None

    async def _extract_images(
        self, message: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        content = message.get("content", "")
        texts: List[str] = []
        images: List[Dict[str, Any]] = []

        def _add(data_url: str):
            try:
                opt = self._optimize_image(data_url)
                header, b64 = opt.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0]
                images.append({"inlineData": {"mimeType": mime, "data": b64}})
            except Exception as e:
                self.log.warning(f"Skipping image: {e}")

        md_re = re.compile(r"!\[[^\]]*\]\((data:image[^)]+|/files/[^)]+|/api/v1/files/[^)]+)\)")

        async def _resolve(url: str):
            if url.startswith("data:"):
                _add(url)
            else:
                b64 = await self._fetch_file_as_base64(url)
                if b64:
                    _add(b64)

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    texts.append(txt)
                    for m in md_re.finditer(txt):
                        await _resolve(m.group(1))
                elif item.get("type") == "image_url":
                    await _resolve(item.get("image_url", {}).get("url", ""))
        elif isinstance(content, str):
            texts.append(content)
            for m in md_re.finditer(content):
                await _resolve(m.group(1))
        return " ".join(t.strip() for t in texts if t.strip()), images

    def _dedup(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.valves.IMAGE_DEDUP_HISTORY:
            return images
        seen, out = set(), []
        for part in images:
            try:
                h = hashlib.sha256(part["inlineData"]["data"].encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
            except Exception:
                pass
            out.append(part)
        return out

    # ---------------- Upload ---------------- #
    @staticmethod
    def _hash(data: Any) -> str:
        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

    @staticmethod
    def _file_entry(content_url: str, mime_type: str) -> Dict[str, Any]:
        return {
            "type": "image",
            "url": content_url,
            "content_type": mime_type,
            "name": "Generated Image",
            "meta": {"content_type": mime_type},
        }

    async def _upload_image(self, request, user, image_b64: str, mime: str) -> str:
        try:
            decoded = base64.b64decode(image_b64)
        except Exception:
            padded = image_b64 + "=" * (-len(image_b64) % 4)
            try:
                decoded = base64.b64decode(padded)
            except Exception as e:
                self.log.warning(f"decode failed: {e}")
                return f"data:{mime};base64,{image_b64}"
        try:
            from fastapi import BackgroundTasks, UploadFile
            from open_webui.internal.db import get_async_db_context
            from open_webui.routers.files import upload_file
            from starlette.datastructures import Headers
        except ImportError as e:
            self.log.warning(f"upload backend unavailable, data-URL fallback: {e}")
            return f"data:{mime};base64,{image_b64}"
        bio = io.BytesIO(decoded)
        bio.seek(0)
        ext = "png"
        if "jpeg" in mime or "jpg" in mime:
            ext = "jpg"
        elif "webp" in mime:
            ext = "webp"
        filename = f"gemini-generated-{uuid.uuid4().hex}.{ext}"
        try:
            async with get_async_db_context() as db:
                up = await upload_file(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    file=UploadFile(file=bio, filename=filename, headers=Headers({"content-type": mime})),
                    process=False,
                    user=user,
                    metadata={"mime_type": mime, "source": "gemini_image_generation"},
                    db=db,
                )
            return request.app.url_path_for("get_file_content_by_id", id=up.id)
        except Exception as e:
            self.log.warning(f"upload failed, data-URL fallback: {e}")
            return f"data:{mime};base64,{image_b64}"

    async def _upload_with_status(
        self, data: Any, mime: str, request, user_dict: dict, emit: Optional[Callable]
    ) -> str:
        async def _status(desc: str, done: bool):
            if emit:
                try:
                    await emit({"type": "status", "data": {"action": "image_upload", "description": desc, "done": done}})
                except Exception as e:
                    self.log.warning(f"status emit failed: {e}")

        try:
            await _status("Uploading generated image to your library...", False)
            try:
                from open_webui.models.users import Users
            except ImportError as e:
                self.log.warning(f"Users backend unavailable, inline fallback: {e}")
                raise
            user = await Users.get_user_by_id(user_dict["id"])
            b64 = base64.b64encode(data).decode() if isinstance(data, bytes) else str(data)
            url = await self._upload_image(request, user, b64, mime)
            await _status("Image uploaded successfully!", True)
            return url
        except Exception as e:
            self.log.warning(f"upload failed, inline fallback: {e}")
            b64 = base64.b64encode(data).decode() if isinstance(data, bytes) else str(data)
            await _status("Using inline image (upload failed)", True)
            return f"data:{mime};base64,{b64}"

    async def _emit_files(self, files: List[Dict[str, Any]], emit: Optional[Callable]) -> bool:
        if not files or not emit:
            return False
        try:
            await emit({"type": "files", "data": {"files": files}})
            return True
        except Exception as e:
            self.log.warning(f"files emit failed: {e}")
            return False

    # ---------------- Payload ---------------- #
    def _validate_aspect(self, value: str) -> Optional[str]:
        if not value or value == "default":
            return None
        if value in [o for o in ASPECT_RATIO_OPTIONS if o != "default"]:
            return value
        self.log.warning(f"Invalid aspect ratio '{value}', using default.")
        return None

    def _validate_resolution(self, value: str) -> Optional[str]:
        if not value or value == "default":
            return None
        if value in [o for o in RESOLUTION_OPTIONS if o != "default"]:
            return value
        self.log.warning(f"Invalid resolution '{value}', using default.")
        return None

    def _build_payload(self, body: Dict[str, Any], parts: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
        g: Dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
        if body.get("temperature") is not None:
            g["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            g["topP"] = body["top_p"]
        if self._is_gemini_3_image_model(model_id):
            ar = self._validate_aspect(body.get("aspect_ratio", self.valves.IMAGE_GENERATION_ASPECT_RATIO))
            rs = self._validate_resolution(body.get("resolution", self.valves.IMAGE_GENERATION_RESOLUTION))
            if ar or rs:
                cfg: Dict[str, Any] = {}
                if ar:
                    cfg["aspectRatio"] = ar
                if rs:
                    cfg["imageSize"] = rs
                g["imageConfig"] = cfg
        payload: Dict[str, Any] = {"contents": [{"role": "user", "parts": parts}], "generationConfig": g}
        if self.valves.USE_PERMISSIVE_SAFETY:
            payload["safetySettings"] = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                )
            ]
        return payload

    # ---------------- Main ---------------- #
    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable] = None,
        __tools__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> Union[str, Dict[str, Any]]:
        async def _status(desc: str, done: bool):
            if __event_emitter__:
                try:
                    await __event_emitter__({"type": "status", "data": {"action": "image_processing", "description": desc, "done": done}})
                except Exception as e:
                    self.log.warning(f"status emit failed: {e}")

        try:
            model_id = self._prepare_model_id(body.get("model", ""))
        except ValueError as ve:
            return f"Model Error: {ve}"
        try:
            self._validate()
        except ValueError as ve:
            return str(ve)
        messages = body.get("messages", [])
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not last_user:
            return "Error: No user message found"
        try:
            prompt, current = await self._extract_images(last_user)
            history: List[Dict[str, Any]] = []
            for msg in messages:
                if msg is last_user or msg.get("role") not in ("user", "assistant"):
                    continue
                _, parts = await self._extract_images(msg)
                history.extend(parts)
            history, current = self._dedup(history), self._dedup(current)
            limit = max(1, self.valves.IMAGE_HISTORY_MAX_REFERENCES)
            combined = (history + current) if self.valves.IMAGE_HISTORY_FIRST else (current + history)
            combined = combined[:limit]
            if not prompt and not combined:
                return "Error: No prompt or images provided"
            if not prompt:
                prompt = "Analyze and describe the provided images."
            default_sys = (self.valves.DEFAULT_SYSTEM_PROMPT or "").strip()
            if default_sys:
                prompt = f"{default_sys}\n\n{prompt}"
            parts: List[Dict[str, Any]] = [{"text": prompt}]
            if self.valves.IMAGE_ADD_LABELS:
                for i, part in enumerate(combined, start=1):
                    parts.append({"text": f"[Image {i}]"})
                    parts.append(part)
            else:
                parts.extend(combined)
        except Exception as e:
            self.log.exception(f"content prep failed: {e}")
            return f"Error preparing image request: {e}"
        try:
            payload = self._build_payload(body, parts, model_id)
        except Exception as e:
            return f"Configuration error: {e}"
        try:
            await _status("Processing image request...", False)
            resp = await self._post(self._url(f"models/{model_id}:generateContent"), payload, "image generation")
            await _status("Processing complete", True)
            fb = resp.get("promptFeedback") or {}
            if (fb.get("blockReason")):
                return f"[Blocked due to Prompt Safety: {fb['blockReason']}]"
            cands = resp.get("candidates") or []
            rparts = (cands[0].get("content") or {}).get("parts", []) if cands else []
            if not rparts:
                return "[No content generated or unexpected response structure]"
            texts: List[str] = []
            inline_imgs: List[str] = []
            files: List[Dict[str, Any]] = []
            seen_hashes = set()
            for part in rparts:
                if not isinstance(part, dict):
                    continue
                if part.get("text"):
                    texts.append(part["text"])
                elif isinstance(part.get("inlineData"), dict):
                    mime = part["inlineData"].get("mimeType", "image/png")
                    data = part["inlineData"].get("data", "")
                    h = self._hash(data)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    if __request__ and __user__:
                        url = await self._upload_with_status(data, mime, __request__, __user__, __event_emitter__)
                        if url.startswith("data:"):
                            inline_imgs.append(f"![Generated Image]({url})")
                        else:
                            files.append(self._file_entry(url, mime))
                    else:
                        inline_imgs.append(f"![Generated Image](data:{mime};base64,{data})")
            out = "".join(texts)
            emitted = await self._emit_files(files, __event_emitter__)
            if files and not emitted:
                inline_imgs.extend(f"![Generated Image]({f['url']})" for f in files)
            if files and emitted and not out.strip():
                out = (out + "\n\n" if out else "") + "Generated image."
            if inline_imgs:
                out = (out + "\n\n" if out else "") + "\n\n".join(inline_imgs)
            return out or "[No content generated]"
        except RuntimeError as e:
            await _status(f"Image generation failed: {e}", True)
            return str(e)
        except Exception as e:
            self.log.exception(f"image request failed: {e}")
            await _status(f"Image generation failed: {e}", True)
            return f"An error occurred while processing your request: {e}"
