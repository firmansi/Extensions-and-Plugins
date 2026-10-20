"""
title: Google Gemini Veo Video Pipeline
author: owndev, olivier-lacroix (REST edition by dfirmans)
version: 2.0.0
required_open_webui_version: 0.11.4
description: Video generation for Google Veo models via the REST API (no extra packages needed - uses only Open WebUI built-ins). Text-to-video and image-to-video with polling status updates and automatic upload to Open WebUI. API-key mode only.
"""

import asyncio
import base64
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


def _http_client(long_timeout: bool = False):
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(f"httpx is required but unavailable: {e}")
    try:
        timeout = httpx.Timeout(300.0 if long_timeout else 120.0, connect=30.0)
        return httpx.AsyncClient(timeout=timeout, follow_redirects=True)
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


VIDEO_ASPECT_RATIOS = ["default", "16:9", "9:16"]
VIDEO_RESOLUTIONS = ["default", "720p", "1080p", "4k"]
VIDEO_DURATIONS = ["default", "4", "5", "6", "8"]
VIDEO_PERSON_GENERATION = ["default", "allow_all", "allow_adult", "dont_allow"]


class Pipe:
    """Video generation pipeline for Google Veo models (pure REST)."""

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
        VIDEO_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_ASPECT_RATIO", "default"),
            description="Aspect ratio: 16:9 landscape or 9:16 portrait.",
            json_schema_extra={"enum": VIDEO_ASPECT_RATIOS},
        )
        VIDEO_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_RESOLUTION", "default"),
            description="Resolution: 720p, 1080p, or 4k (model dependent).",
            json_schema_extra={"enum": VIDEO_RESOLUTIONS},
        )
        VIDEO_GENERATION_DURATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_DURATION", "default"),
            description="Duration in seconds: 4, 5, 6, or 8 (model dependent).",
            json_schema_extra={"enum": VIDEO_DURATIONS},
        )
        VIDEO_GENERATION_NEGATIVE_PROMPT: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_NEGATIVE_PROMPT", ""),
            description="Default negative prompt (what to exclude).",
        )
        VIDEO_GENERATION_PERSON_GENERATION: str = Field(
            default=os.getenv("GOOGLE_VIDEO_GENERATION_PERSON_GENERATION", "default"),
            description="Person generation control.",
            json_schema_extra={"enum": VIDEO_PERSON_GENERATION},
        )
        VIDEO_POLL_INTERVAL: int = Field(
            default=int(os.getenv("GOOGLE_VIDEO_POLL_INTERVAL", "10")),
            description="Polling interval in seconds while waiting for the video.",
        )
        VIDEO_POLL_TIMEOUT: int = Field(
            default=int(os.getenv("GOOGLE_VIDEO_POLL_TIMEOUT", "600")),
            description="Max seconds to wait before timing out (0 = no limit).",
        )
        MODEL_CACHE_TTL: int = Field(
            default=int(os.getenv("GOOGLE_MODEL_CACHE_TTL", "600")),
            description="Seconds to cache the model list.",
        )
        MODEL_ADDITIONAL: str = Field(
            default=os.getenv("GOOGLE_MODEL_ADDITIONAL", ""),
            description="Comma-separated Veo model IDs to add manually.",
        )
        MODEL_WHITELIST: str = Field(
            default=os.getenv("GOOGLE_MODEL_WHITELIST", ""),
            description="Comma-separated model IDs to show. Empty = all veo-* models.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Google Veo Video: "
        self.log = logging.getLogger("google_veo_video.pipe")
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

    # ---------------- Models ---------------- #
    @staticmethod
    def _is_video_model(model_id: str) -> bool:
        return model_id.lower().startswith("veo-")

    def strip_prefix(self, model_name: str) -> str:
        stripped = model_name.rsplit("/", 1)[-1]
        if "." in stripped and not stripped.startswith("veo-"):
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
        seen: Dict[str, str] = {
            "veo-3.1-generate-preview": "veo-3.1-generate-preview 🎬",
            "veo-3.0-generate-preview": "veo-3.0-generate-preview 🎬",
            "veo-2.0-generate-001": "veo-2.0-generate-001 🎬",
        }
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
                    if self._is_video_model(mid):
                        seen[mid] = str(m.get("displayName") or mid) + " 🎬"
        except Exception as e:
            self.log.warning(f"models.list failed, using static list: {e}")
        for extra in re.findall(r"[^,\s]+", self.valves.MODEL_ADDITIONAL or ""):
            extra = self.strip_prefix(extra)
            if self._is_video_model(extra):
                seen.setdefault(extra, extra + " 🎬")
        items = [{"id": k, "name": v} for k, v in sorted(seen.items())]
        whitelist = set(re.findall(r"[^,\s]+", self.valves.MODEL_WHITELIST or ""))
        if whitelist:
            items = [m for m in items if m["id"] in whitelist]
        self._model_cache = items
        self._model_cache_time = now
        return items

    def pipes(self) -> List[Dict[str, str]]:
        try:
            self.name = "Google Veo Video: "
            return self.get_google_models()
        except ValueError as e:
            return [{"id": "error", "name": str(e)}]
        except Exception as e:
            self.log.exception(f"pipes listing failed: {e}")
            return [{"id": "error", "name": f"Unexpected error: {e}"}]

    def _prepare_model_id(self, model_id: str) -> str:
        original = model_id
        model_id = self.strip_prefix(model_id)
        if not self._is_video_model(model_id):
            found = next(
                (m["id"] for m in self.get_google_models() if m["name"] == original),
                None,
            )
            if found:
                return found
            raise ValueError(
                f"Model '{original}' is not a Veo video model. Use the Text or Image pipeline for it."
            )
        return model_id

    # ---------------- Capabilities / params ---------------- #
    @staticmethod
    def _capabilities(model_id: str) -> Dict[str, Any]:
        mid = model_id.lower()
        if "veo-3.1" in mid:
            return {"supports_resolution": True, "valid_resolutions": ["720p", "1080p", "4k"], "valid_durations": [4, 6, 8]}
        if "veo-3" in mid:
            return {"supports_resolution": True, "valid_resolutions": ["720p", "1080p"], "valid_durations": [8]}
        if "veo-2" in mid:
            return {"supports_resolution": False, "valid_resolutions": [], "valid_durations": [5, 6, 8]}
        return {"supports_resolution": False, "valid_resolutions": [], "valid_durations": [8]}

    def _build_parameters(self, body: Dict[str, Any], model_id: str, minimal: bool = False) -> Dict[str, Any]:
        caps = self._capabilities(model_id)
        params: Dict[str, Any] = {}
        ar = body.get("aspect_ratio", self.valves.VIDEO_GENERATION_ASPECT_RATIO)
        if ar in ("16:9", "9:16"):
            params["aspectRatio"] = ar
        if not minimal:
            res = body.get("resolution", self.valves.VIDEO_GENERATION_RESOLUTION)
            res = res.strip().lower() if isinstance(res, str) else None
            if res and caps["supports_resolution"] and res in caps["valid_resolutions"]:
                params["resolution"] = res
            try:
                dur = int(body.get("duration", self.valves.VIDEO_GENERATION_DURATION))
            except (TypeError, ValueError):
                dur = None
            if dur in caps["valid_durations"]:
                params["durationSeconds"] = dur
            neg = body.get("negative_prompt", self.valves.VIDEO_GENERATION_NEGATIVE_PROMPT) or None
            if neg:
                params["negativePrompt"] = neg
            person = body.get("person_generation", self.valves.VIDEO_GENERATION_PERSON_GENERATION)
            if person in ("allow_all", "allow_adult", "dont_allow"):
                params["personGeneration"] = person
        self.log.debug(f"Video parameters for {model_id}: {params}")
        return params

    # ---------------- Input ---------------- #
    @staticmethod
    def _split_content(content: Any) -> Tuple[str, List[str]]:
        texts: List[str] = []
        data_urls: List[str] = []
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    data_urls.append(url)
        return " ".join(t.strip() for t in texts if isinstance(t, str) and t.strip()), data_urls

    # ---------------- Upload ---------------- #
    @staticmethod
    def _file_entry(file_id: str, url: str, filename: str, mime: str, size: int) -> Dict[str, Any]:
        return {
            "id": file_id,
            "type": "file",
            "url": url,
            "name": filename,
            "filename": filename,
            "size": size,
            "content_type": mime,
            "meta": {"content_type": mime, "size": size},
        }

    async def _upload_video(
        self, request, user, data: bytes, mime: str,
        chat_id: Optional[str] = None, message_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            from fastapi import BackgroundTasks, UploadFile
            from open_webui.internal.db import get_async_db_context
            from open_webui.models.chats import Chats
            from open_webui.routers.files import upload_file
            from starlette.datastructures import Headers
        except ImportError as e:
            raise RuntimeError(f"upload backend unavailable: {e}")
        bio = io.BytesIO(data)
        bio.seek(0)
        ext = "webm" if "webm" in mime else "mp4"
        filename = f"veo-generated-{uuid.uuid4().hex}.{ext}"
        async with get_async_db_context() as db:
            up = await upload_file(
                request=request,
                background_tasks=BackgroundTasks(),
                file=UploadFile(file=bio, filename=filename, headers=Headers({"content-type": mime})),
                process=False,
                user=user,
                metadata={"mime_type": mime, "source": "veo_video_generation"},
                db=db,
            )
            if chat_id and message_id:
                try:
                    await Chats.insert_chat_files(
                        chat_id=chat_id, message_id=message_id,
                        file_ids=[up.id], user_id=user.id, db=db,
                    )
                except Exception as e:
                    self.log.warning(f"chat file link failed: {e}")
        url = str(request.app.url_path_for("get_file_content_by_id", id=up.id))
        return url, self._file_entry(up.id, url, filename, mime, len(data))

    async def _emit_files(self, files: List[Dict[str, Any]], emit: Optional[Callable]) -> bool:
        if not files or not emit:
            return False
        try:
            await emit({"type": "files", "data": {"files": files}})
            return True
        except Exception as e:
            self.log.warning(f"files emit failed: {e}")
            return False

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
                    await __event_emitter__({"type": "status", "data": {"action": "video_generation", "description": desc, "done": done}})
                except Exception as e:
                    self.log.warning(f"status emit failed: {e}")

        __metadata__ = __metadata__ or {}
        try:
            model_id = self._prepare_model_id(body.get("model", ""))
        except ValueError as ve:
            return f"Model Error: {ve}"
        try:
            key = self._validate()
        except ValueError as ve:
            return str(ve)
        messages = body.get("messages", [])
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not last_user:
            return "Error: No user message found for video generation"
        prompt, data_urls = self._split_content(last_user.get("content", ""))
        if not prompt:
            return "Error: No prompt provided for video generation"
        instance: Dict[str, Any] = {"prompt": prompt}
        if data_urls:
            try:
                header, b64 = data_urls[0].split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                base64.b64decode(b64)
                instance["image"] = {"bytesBase64Encoded": b64, "mimeType": mime}
                self.log.debug("Using attached image for image-to-video")
            except Exception as e:
                self.log.warning(f"Reference image conversion failed, text-to-video only: {e}")
        headers = self._headers(key)
        await _status(f"Starting video generation with {model_id}...", False)
        operation: Dict[str, Any] = {}
        try:
            async with _http_client(long_timeout=True) as client:
                for minimal in (False, True):
                    params = self._build_parameters(body, model_id, minimal=minimal)
                    payload: Dict[str, Any] = {"instances": [instance], "parameters": params}
                    resp = await client.post(
                        self._url(f"models/{model_id}:predictLongRunning"),
                        headers=headers, json=payload,
                    )
                    if resp.status_code >= 400:
                        err = _api_error("starting video generation", resp)
                        if not minimal and resp.status_code == 400:
                            self.log.warning(f"Full params rejected, retrying minimal: {err}")
                            continue
                        raise err
                    operation = resp.json()
                    break
        except RuntimeError as e:
            self.log.exception(f"video request failed: {e}")
            await _status(f"Video generation failed: {e}", True)
            return str(e)
        except Exception as e:
            self.log.exception(f"video request failed: {e}")
            await _status(f"Video generation failed: {e}", True)
            return f"Error starting video generation: {e}"
        op_name = operation.get("name")
        if not op_name:
            return f"Error: API did not return an operation (response: {operation})"
        poll_interval = max(self.valves.VIDEO_POLL_INTERVAL, 5)
        poll_timeout = max(self.valves.VIDEO_POLL_TIMEOUT, 0)
        elapsed = 0
        op: Dict[str, Any] = {"done": False}
        try:
            async with _http_client(long_timeout=True) as client:
                while not op.get("done"):
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                    if poll_timeout > 0 and elapsed >= poll_timeout:
                        msg = f"Video generation timed out after {elapsed}s (limit: {poll_timeout}s)"
                        await _status(msg, True)
                        return f"Error: {msg}"
                    try:
                        resp = await client.get(self._url(op_name), headers=headers)
                        if resp.status_code >= 400:
                            self.log.warning(f"polling HTTP {resp.status_code}, retrying")
                        else:
                            op = resp.json()
                    except Exception as e:
                        self.log.warning(f"polling error, retrying: {e}")
                    await _status(f"Generating video... ({elapsed}s elapsed)", False)
        except Exception as e:
            self.log.exception(f"polling failed: {e}")
            await _status(f"Video generation failed: {e}", True)
            return f"Error while polling video operation: {e}"
        if op.get("error"):
            msg = json.dumps(op["error"])[:500]
            await _status(f"Video generation failed: {msg}", True)
            return f"Video generation failed: {msg}"
        try:
            gvr = (op.get("response") or {}).get("generateVideoResponse") or {}
            samples = gvr.get("generatedSamples")
            if isinstance(samples, dict):
                samples = [samples]
            if not samples:
                return "Error: No videos were generated"
            uris = [s.get("video", {}).get("uri") for s in samples if isinstance(s, dict)]
            uris = [u for u in uris if u]
            if not uris:
                return "Error: No video download URI in operation response"
        except Exception as e:
            return f"Error reading operation response: {e}"
        files: List[Dict[str, Any]] = []
        links: List[str] = []
        failed, skipped = 0, 0
        try:
            async with _http_client(long_timeout=True) as client:
                for idx, uri in enumerate(uris):
                    try:
                        dl = await client.get(uri, headers=headers)
                        if dl.status_code >= 400:
                            raise _api_error(f"downloading video {idx + 1}", dl)
                        video_bytes = dl.content
                    except Exception as e:
                        self.log.warning(f"video {idx} download failed: {e}")
                        links.append(f"Generated video {idx + 1}, but download failed: {e}")
                        continue
                    mime = "video/mp4"
                    entry, url = None, None
                    attempted = False
                    if __request__ and __user__:
                        attempted = True
                        try:
                            await _status("Uploading generated video to your library...", False)
                            try:
                                from open_webui.models.users import Users
                            except ImportError as e:
                                raise RuntimeError(f"Users backend unavailable: {e}")
                            user = await Users.get_user_by_id(__user__["id"])
                            url, entry = await self._upload_video(
                                __request__, user, video_bytes, mime,
                                (__metadata__ or {}).get("chat_id"), (__metadata__ or {}).get("message_id"),
                            )
                            await _status("Video uploaded successfully!", True)
                        except Exception as e:
                            self.log.warning(f"upload failed, data-URL fallback: {e}")
                            url = f"data:{mime};base64,{base64.b64encode(video_bytes).decode()}"
                            await _status("Using inline video (upload failed)", True)
                    else:
                        url = f"data:{mime};base64,{base64.b64encode(video_bytes).decode()}"
                    if entry:
                        files.append(entry)
                        if url:
                            links.append(f"[🎬 Generated Video {idx + 1}]({url})")
                        continue
                    if attempted:
                        failed += 1
                    else:
                        skipped += 1
                    links.append(f"[🎬 Generated Video {idx + 1}]({url})" if url and not url.startswith("data:") else f"Generated video {idx + 1}.")
        except Exception as e:
            self.log.exception(f"video handling failed: {e}")
            return f"Error processing generated video: {e}"
        await _status(f"Video generation complete ({elapsed}s)", True)
        emitted = await self._emit_files(files, __event_emitter__)
        parts: List[str] = []
        if files and emitted:
            n = len(files)
            parts.append("Generated video attached." if n == 1 else f"Generated {n} videos attached.")
        else:
            parts.extend(links)
        if files and not emitted:
            parts.extend(links)
        if failed:
            parts.append("Some videos could not be attached directly.")
        if skipped:
            parts.append("Video attachments were skipped because chat upload context was unavailable.")
        content = "\n\n".join(p for p in parts if p) or "[No video content generated]"
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}
