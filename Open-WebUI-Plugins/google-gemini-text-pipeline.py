"""
title: Google Gemini Text Pipeline
author: owndev, olivier-lacroix (REST edition by dfirmans)
version: 2.1.3
required_open_webui_version: 0.11.4
description: Text/chat pipeline for Google Gemini models via the REST API (no extra packages needed - uses only Open WebUI built-ins). Streaming, thinking budgets/levels, Google Search grounding, native Open WebUI tool calling (agentic loop), and usage tracking. API-key mode only; Vertex AI mode is not supported in this edition.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from fastapi import Request
from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:
    SRC_LOG_LEVELS = {}

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False


def _http_client(streaming: bool = False):
    """Build an httpx client (httpx ships with Open WebUI)."""
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(f"httpx is required but unavailable: {e}")
    if streaming:
        return httpx.AsyncClient(timeout=None, follow_redirects=True)
    try:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=30.0), follow_redirects=True
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


def _g(obj: Any, *keys: str, default: Any = None) -> Any:
    """Get first present key (tolerates camelCase/snake_case variants)."""
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _schema_for_annotation(ann: Any) -> Dict[str, Any]:
    """Map a Python type hint to a Gemini function-declaration schema.

    Gemini requires ``items`` on every ARRAY schema, so arrays always
    carry an item type (inferred from List[X] when available).
    """
    import typing as _typing

    origin = _typing.get_origin(ann)
    args = _typing.get_args(ann)
    if origin is _typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _schema_for_annotation(non_none[0])
        return {"type": "string"}
    if origin in (list, tuple, set, frozenset):
        item = _schema_for_annotation(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}
    aname = getattr(ann, "__name__", ann if isinstance(ann, str) else "")
    if aname in ("str", "string"):
        return {"type": "string"}
    if aname in ("int", "integer"):
        return {"type": "integer"}
    if aname in ("float", "number"):
        return {"type": "number"}
    if aname in ("bool", "boolean"):
        return {"type": "boolean"}
    if aname == "list":
        return {"type": "array", "items": {"type": "string"}}
    if aname == "dict":
        return {"type": "object"}
    return {"type": "string"}


class Pipe:
    """Text/chat pipeline for Google Gemini models (pure REST)."""

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
        STREAMING_ENABLED: bool = Field(
            default=os.getenv("GOOGLE_STREAMING_ENABLED", "true").lower() == "true",
            description="Enable streaming responses.",
        )
        INCLUDE_THOUGHTS: bool = Field(
            default=os.getenv("GOOGLE_INCLUDE_THOUGHTS", "true").lower() == "true",
            description="Show Gemini thinking output.",
        )
        STRIP_THINKING_FROM_HISTORY: bool = Field(
            default=os.getenv("GOOGLE_STRIP_THINKING_FROM_HISTORY", "true").lower()
            == "true",
            description="Remove previously rendered thinking summaries from history before replaying to the API.",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("GOOGLE_THINKING_BUDGET", "-1")),
            description="Thinking budget for Gemini 2.5 models (0=off, -1=dynamic, 1-32768=fixed).",
        )
        THINKING_LEVEL: str = Field(
            default=os.getenv("GOOGLE_THINKING_LEVEL", ""),
            description="Thinking level for Gemini 3 models (minimal/low/medium/high). Empty = model default.",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("GOOGLE_USE_PERMISSIVE_SAFETY", "false").lower()
            == "true",
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
            description="Default system prompt prepended to any user-defined system prompt.",
        )
        MODEL_ADDITIONAL: str = Field(
            default=os.getenv("GOOGLE_MODEL_ADDITIONAL", ""),
            description="Comma-separated model IDs to add manually.",
        )
        MODEL_WHITELIST: str = Field(
            default=os.getenv("GOOGLE_MODEL_WHITELIST", ""),
            description="Comma-separated model IDs to show. Empty = all gemini-* text models.",
        )
        IMAGE_MAX_SIZE_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_MAX_SIZE_MB", "15.0")),
            description="Max input-image size in MB before compression.",
        )
        IMAGE_MAX_DIMENSION: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_MAX_DIMENSION", "2048")),
            description="Max input-image width/height in pixels before resizing.",
        )
        IMAGE_COMPRESSION_QUALITY: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_COMPRESSION_QUALITY", "85")),
            description="JPEG quality for input-image compression (1-100).",
        )

    _THINKING_DETAILS_RE = re.compile(
        r"<details[^>]*>\s*<summary>\s*Thought \(\d+s\)\s*</summary>"
        r".*?</details>\s*",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Google Gemini Text: "
        self.log = logging.getLogger("google_gemini_text.pipe")
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

    async def _get(self, url: str, action: str) -> Dict[str, Any]:
        key = self._validate()
        async with _http_client() as client:
            resp = await client.get(url, headers=self._headers(key))
        if resp.status_code >= 400:
            raise _api_error(action, resp)
        data = resp.json()
        return data if isinstance(data, dict) else {}

    # ---------------- Models ---------------- #
    @staticmethod
    def _is_text_model(model_id: str) -> bool:
        mid = model_id.lower()
        if not mid.startswith("gemini-"):
            return False
        return "image" not in mid

    def strip_prefix(self, model_name: str) -> str:
        stripped = model_name.rsplit("/", 1)[-1]
        if "." in stripped and not stripped.startswith("gemini-"):
            stripped = stripped.split(".", 1)[1]
        return stripped

    def _static_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
            {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite"},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
            {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash-Lite"},
            {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro Preview"},
            {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash Preview"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
        ]

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        now = time.time()
        if (
            not force_refresh
            and self._model_cache is not None
            and (now - self._model_cache_time) < self.valves.MODEL_CACHE_TTL
        ):
            return self._model_cache
        items = list(self._static_models())
        try:
            key = (self.valves.GOOGLE_API_KEY or "").strip()
            if key:
                import urllib.request

                req = urllib.request.Request(
                    self._url("models"), headers={"x-goog-api-key": key}
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read().decode("utf-8", "replace"))
                seen: Dict[str, str] = {}
                for m in data.get("models", []) if isinstance(data, dict) else []:
                    mid = self.strip_prefix(str(_g(m, "name", default="")))
                    methods = _g(m, "supportedGenerationMethods", "supported_generation_methods", default=[]) or []
                    if self._is_text_model(mid) and ("generateContent" in methods or not methods):
                        seen[mid] = _g(m, "displayName", "display_name", default=mid)
                if seen:
                    items = [{"id": k, "name": v} for k, v in sorted(seen.items())]
        except Exception as e:
            self.log.warning(f"models.list failed, using static list: {e}")
        for extra in re.findall(r"[^,\s]+", self.valves.MODEL_ADDITIONAL or ""):
            extra = self.strip_prefix(extra)
            if self._is_text_model(extra) and all(m["id"] != extra for m in items):
                items.append({"id": extra, "name": extra})
        whitelist = set(re.findall(r"[^,\s]+", self.valves.MODEL_WHITELIST or ""))
        if whitelist:
            items = [m for m in items if m["id"] in whitelist]
        self._model_cache = items
        self._model_cache_time = now
        return items

    def pipes(self) -> List[Dict[str, str]]:
        try:
            self.name = "Google Gemini Text: "
            return self.get_google_models()
        except ValueError as e:
            return [{"id": "error", "name": str(e)}]
        except Exception as e:
            self.log.exception(f"pipes listing failed: {e}")
            return [{"id": "error", "name": f"Unexpected error: {e}"}]

    def _prepare_model_id(self, model_id: str) -> str:
        original = model_id
        model_id = self.strip_prefix(model_id)
        if not self._is_text_model(model_id):
            found = next(
                (m["id"] for m in self.get_google_models() if m["name"] == original),
                None,
            )
            if found:
                return found
            raise ValueError(
                f"Model '{original}' is not a text model. Use the Image or Video pipeline for it."
            )
        return model_id

    # ---------------- Content ---------------- #
    def _combine_system_prompts(self, user_system: Optional[str]) -> Optional[str]:
        default = (self.valves.DEFAULT_SYSTEM_PROMPT or "").strip()
        user = (user_system or "").strip()
        if default and user:
            return f"{default}\n\n{user}"
        return default or user or None

    def _strip_thinking(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._THINKING_DETAILS_RE.sub("", content)
        if isinstance(content, list):
            return [
                (
                    {**item, "text": self._THINKING_DETAILS_RE.sub("", item["text"])}
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    else item
                )
                for item in content
            ]
        return content

    def _optimize_input_image(self, data_url: str) -> str:
        if not PIL_AVAILABLE:
            return data_url
        try:
            header, encoded = data_url.split(",", 1)
            raw = base64.b64decode(encoded)
            if len(raw) / (1024 * 1024) <= self.valves.IMAGE_MAX_SIZE_MB:
                with Image.open(io.BytesIO(raw)) as img:
                    if max(img.size) <= self.valves.IMAGE_MAX_DIMENSION:
                        return data_url
            with Image.open(io.BytesIO(raw)) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                w, h = img.size
                scale = min(1.0, self.valves.IMAGE_MAX_DIMENSION / max(w, h))
                if scale < 1.0:
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save("JPEG", buf, quality=self.valves.IMAGE_COMPRESSION_QUALITY, optimize=True)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception as e:
            self.log.warning(f"Image optimization skipped: {e}")
            return data_url

    def _process_multimodal(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        for item in items:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    try:
                        opt = self._optimize_input_image(url)
                        header, encoded = opt.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                        parts.append({"inlineData": {"mimeType": mime, "data": encoded}})
                    except Exception as e:
                        self.log.warning(f"Skipping image: {e}")
                        parts.append({"text": "[Image could not be processed]"})
                else:
                    parts.append({"text": f"[Image URL not processed: {url}]"})
        return parts

    def _prepare_content(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        system = self._combine_system_prompts(
            next((m["content"] for m in messages if m.get("role") == "system"), None)
        )
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            api_role = "model" if role == "assistant" else "user"
            content = msg.get("content", "")
            if api_role == "model" and self.valves.STRIP_THINKING_FROM_HISTORY:
                content = self._strip_thinking(content)
            if isinstance(content, list):
                parts = self._process_multimodal(content)
            elif isinstance(content, str):
                parts = [{"text": content}]
            else:
                continue
            if api_role == "model" and self.valves.STRIP_THINKING_FROM_HISTORY:
                parts = [p for p in parts if not (isinstance(p.get("text"), str) and not p["text"].strip())]
            if parts:
                contents.append({"role": api_role, "parts": parts})
        return contents, system

    # ---------------- Payload ---------------- #
    def _is_gemini_3(self, model_id: str) -> bool:
        m = model_id.lower()
        return m.startswith("gemini-3-") or m.startswith("gemini-3.")

    def _supports_thinking(self, model_id: str) -> bool:
        return model_id.lower().startswith(("gemini-2.5", "gemini-3"))

    def _build_payload(
        self,
        body: Dict[str, Any],
        contents: List[Dict[str, Any]],
        system: Optional[str],
        metadata: Dict[str, Any],
        tools_dict: Optional[Dict[str, Any]],
        model_id: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        g: Dict[str, Any] = {}
        if body.get("temperature") is not None:
            g["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            g["topP"] = body["top_p"]
        if body.get("top_k") is not None:
            g["topK"] = body["top_k"]
        if body.get("max_tokens") is not None:
            g["maxOutputTokens"] = body["max_tokens"]
        if body.get("stop"):
            g["stopSequences"] = body["stop"]
        if self._supports_thinking(model_id):
            t: Dict[str, Any] = {
                "includeThoughts": bool(body.get("include_thoughts", True))
                and self.valves.INCLUDE_THOUGHTS
            }
            if self._is_gemini_3(model_id):
                level = (body.get("reasoning_effort") or self.valves.THINKING_LEVEL or "").strip().lower()
                mapping = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
                if level in mapping:
                    t["thinkingLevel"] = mapping[level]
            else:
                try:
                    budget = int(body.get("thinking_budget", self.valves.THINKING_BUDGET))
                except (TypeError, ValueError):
                    budget = -1
                t["thinkingBudget"] = -1 if budget == -1 else min(max(budget, 0), 32768)
            g["thinkingConfig"] = t
        if g:
            payload["generationConfig"] = g
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
        features = (metadata or {}).get("features", {})
        if features.get("google_search_tool", False):
            payload["tools"] = [{"googleSearch": {}}, {"urlContext": {}}]
        decls: List[Dict[str, Any]] = []
        if tools_dict:
            for tname, tdef in tools_dict.items():
                if tname.startswith("_") or self._is_image_tool(tname):
                    continue
                fn = tdef.get("callable") if isinstance(tdef, dict) else tdef
                if not callable(fn):
                    continue
                decl = self._tool_declaration(tname, fn)
                if decl:
                    decls.append(decl)
                    self.log.debug(f"Exposing tool '{tname}' to Gemini")
        if decls:
            payload.setdefault("tools", []).append({"functionDeclarations": decls})
        return payload

    # ---------------- Response parsing ---------------- #
    def _safety_message(self, resp: Dict[str, Any]) -> Optional[str]:
        fb = resp.get("promptFeedback") or {}
        reason = _g(fb, "blockReason", "block_reason")
        if reason:
            return f"[Blocked due to Prompt Safety: {reason}]"
        cands = resp.get("candidates") or []
        if not cands:
            return "[Blocked by safety settings or no candidates generated]"
        fr = _g(cands[0], "finishReason", "finish_reason")
        if fr == "SAFETY":
            ratings = _g(cands[0], "safetyRatings", "safety_ratings", default=[]) or []
            blocked = next((r for r in ratings if _g(r, "blocked")), None)
            extra = f" ({_g(blocked, 'category')})" if blocked else ""
            return f"[Blocked by safety settings{extra}]"
        if fr == "PROHIBITED_CONTENT":
            return "[Content blocked due to prohibited content policy violation]"
        return None

    @staticmethod
    def _usage(meta: Any) -> Optional[Dict[str, int]]:
        if not isinstance(meta, dict):
            return None
        out: Dict[str, int] = {}
        if _g(meta, "promptTokenCount", "prompt_token_count") is not None:
            out["prompt_tokens"] = _g(meta, "promptTokenCount", "prompt_token_count")
        if _g(meta, "candidatesTokenCount", "candidates_token_count") is not None:
            out["completion_tokens"] = _g(meta, "candidatesTokenCount", "candidates_token_count")
        if out:
            out["total_tokens"] = out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
            return out
        return None

    @staticmethod
    def _format_sources(chunks: list) -> list:
        out = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            rc = chunk.get("retrievedContext") or chunk.get("retrieved_context")
            web = chunk.get("web")
            if isinstance(rc, dict):
                title = _g(rc, "title") or "Document"
                out.append(
                    {
                        "source": {"name": title, "type": "vertex_ai_search", "uri": _g(rc, "uri")},
                        "document": [_g(rc, "chunkText", "chunk_text") or ""],
                        "metadata": [{"source": title}],
                    }
                )
            elif isinstance(web, dict):
                title = _g(web, "title") or "Source"
                out.append(
                    {
                        "source": {"name": title, "type": "web_search_results", "url": _g(web, "uri")},
                        "document": ["Click the link to view the content."],
                        "metadata": [{"source": title}],
                    }
                )
        return out

    async def _process_grounding(self, metas: List[Any], text: str, emit: Optional[Callable]) -> str:
        chunks, queries, supports = [], [], []
        for md in metas:
            if not isinstance(md, dict):
                continue
            chunks.extend(_g(md, "groundingChunks", "grounding_chunks", default=[]) or [])
            queries.extend(_g(md, "webSearchQueries", "web_search_queries", default=[]) or [])
            supports.extend(_g(md, "groundingSupports", "grounding_supports", default=[]) or [])
        if chunks and emit:
            for src in self._format_sources(chunks):
                try:
                    await emit({"type": "source", "data": src})
                except Exception as e:
                    self.log.warning(f"source emit failed: {e}")
        if queries and emit:
            try:
                await emit(
                    {
                        "type": "status",
                        "data": {
                            "action": "web_search",
                            "description": "This response was grounded with Google Search",
                            "urls": [f"https://www.google.com/search?q={q}" for q in queries],
                        },
                    }
                )
            except Exception as e:
                self.log.warning(f"status emit failed: {e}")
        if supports:
            try:
                raw = text.encode("utf-8")
                pos, parts = 0, []
                for sup in supports:
                    seg = _g(sup, "segment", default={}) or {}
                    end = _g(seg, "endIndex", "end_index", default=pos)
                    idx = _g(sup, "groundingChunkIndices", "grounding_chunk_indices", default=[]) or []
                    parts.append(raw[pos:end].decode("utf-8"))
                    parts.append(" " + "".join(f"[{i + 1}]" for i in idx))
                    pos = end
                if pos < len(raw):
                    parts.append(raw[pos:].decode("utf-8"))
                return "".join(parts)
            except Exception as e:
                self.log.warning(f"citation injection failed: {e}")
        return text

    @staticmethod
    def _split_parts(candidate: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        answers, thoughts = [], []
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("thought") and part.get("text"):
                thoughts.append(part["text"])
            elif part.get("text"):
                answers.append(part["text"])
        return answers, thoughts

    @staticmethod
    def _split_calls(candidate: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], Optional[str]]]:
        """Extract function calls plus their thought signatures.

        Gemini 3 validates that every replayed functionCall carries the
        thoughtSignature the model originally issued, else HTTP 400.
        The signature may sit at part level or inside the functionCall
        object, so both locations are checked.
        """
        calls: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            fc = part.get("functionCall")
            if isinstance(fc, dict) and fc.get("name"):
                args = fc.get("args")
                sig = (
                    part.get("thoughtSignature")
                    or part.get("thought_signature")
                    or fc.get("thoughtSignature")
                    or fc.get("thought_signature")
                )
                calls.append((fc["name"], args if isinstance(args, dict) else {},
                              sig if isinstance(sig, str) else None))
        return calls

    @staticmethod
    def _replay_model_parts(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the model's parts verbatim for history replay.

        Signatures and every other field are preserved exactly as issued;
        only the output-only ``thought`` flag is stripped.
        """
        content = candidate.get("content") or {}
        out: List[Dict[str, Any]] = []
        for part in content.get("parts") or []:
            if isinstance(part, dict):
                out.append({k: v for k, v in part.items() if k != "thought"})
            else:
                out.append(part)
        return out

    @staticmethod
    def _is_image_tool(tool_name: str) -> bool:
        return tool_name in ("generate_image", "edit_image")

    def _resolve_tool(self, tools_dict: Dict[str, Any], name: str) -> Optional[Callable]:
        tool_def = (tools_dict or {}).get(name)
        if isinstance(tool_def, dict) and callable(tool_def.get("callable")):
            return tool_def["callable"]
        if callable(tool_def):
            return tool_def
        return None

    @staticmethod
    def _tool_declaration(name: str, fn: Callable) -> Optional[Dict[str, Any]]:
        """Build a Gemini functionDeclaration from a Python callable."""
        import inspect as _inspect

        try:
            sig = _inspect.signature(fn)
        except (TypeError, ValueError):
            return None
        props: Dict[str, Any] = {}
        required: List[str] = []
        for pname, p in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if p.kind in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD):
                continue
            schema = _schema_for_annotation(
                p.annotation if p.annotation is not _inspect.Parameter.empty else str
            )
            schema["description"] = pname
            props[pname] = schema
            if p.default is _inspect.Parameter.empty:
                required.append(pname)
        if not props:
            return None
        doc = (_inspect.getdoc(fn) or "").strip().split("\n")[0][:500]
        decl: Dict[str, Any] = {
            "name": name,
            "description": doc or f"Tool {name}",
            "parameters": {"type": "object", "properties": props},
        }
        if required:
            decl["parameters"]["required"] = required
        return decl

    # ---------------- Streaming ---------------- #
    async def _handle_stream(
        self, url: str, payload: Dict[str, Any], emit: Optional[Callable]
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        async def _emit(etype: str, data: Dict[str, Any]) -> None:
            if emit:
                try:
                    await emit({"type": etype, "data": data})
                except Exception as e:
                    self.log.warning(f"emit {etype} failed: {e}")

        await _emit("chat:start", {"role": "assistant"})
        answers: List[str] = []
        thoughts: List[str] = []
        started: Optional[float] = None
        metas: List[Any] = []
        usage_meta = None
        key = self._validate()
        last: Exception | None = None
        stream_ok = False
        for _ in range(self.valves.RETRY_COUNT + 1):
            try:
                async with _http_client(streaming=True) as client:
                    async with client.stream(
                        "POST", url, headers=self._headers(key), json=payload
                    ) as resp:
                        if resp.status_code == 429 or resp.status_code >= 500:
                            body = await resp.aread()
                            try:
                                err = json.loads(body.decode("utf-8", "replace")).get("error", {})
                                last = RuntimeError(f"Gemini API {err.get('code')}: {err.get('message')}")
                            except Exception:
                                last = RuntimeError(f"Gemini API HTTP {resp.status_code}")
                            await asyncio.sleep(2)
                            continue
                        if resp.status_code >= 400:
                            raise _api_error("streaming", resp)
                        stream_ok = True
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except Exception:
                                continue
                            if not isinstance(chunk, dict):
                                continue
                            if chunk.get("usageMetadata"):
                                usage_meta = chunk["usageMetadata"]
                            cands = chunk.get("candidates") or []
                            if not cands:
                                fb = chunk.get("promptFeedback") or {}
                                reason = _g(fb, "blockReason", "block_reason")
                                msg = f"[Blocked due to Prompt Safety: {reason}]" if reason else "[Blocked by safety settings]"
                                await _emit("chat:finish", {"role": "assistant", "content": msg, "done": True, "error": True})
                                yield msg
                                return
                            cand = cands[0]
                            if cand.get("groundingMetadata"):
                                metas.append(cand["groundingMetadata"])
                            a, t = self._split_parts(cand)
                            for seg in t:
                                if started is None:
                                    started = time.time()
                                thoughts.append(seg)
                                preview = seg.replace("\n", " ").strip()[:120]
                                await _emit("status", {"action": "thinking", "description": f"Thinking… {preview}", "done": False, "hidden": False})
                            for seg in a:
                                answers.append(seg)
                                await _emit("chat:message:delta", {"role": "assistant", "content": seg})
                break
            except Exception as e:
                last = e
                if stream_ok:
                    break
        if not stream_ok and not answers and not thoughts:
            msg = f"Error during streaming: {last}" if last else "Error during streaming."
            await _emit("chat:finish", {"role": "assistant", "content": msg, "done": True, "error": True})
            yield msg
            return
        try:
            final = "".join(answers)
            if metas and emit:
                final = await self._process_grounding(metas, final, emit) or final
            content = final
            if thoughts:
                dur = int(max(0, time.time() - (started or time.time())))
                quoted = "\n".join(f"> {l}" for l in "".join(thoughts).strip().split("\n"))
                content = f"<details>\n<summary>Thought ({dur}s)</summary>\n\n{quoted}\n\n</details>{final}"
            content = content or ""
            await _emit("replace", {"role": "assistant", "content": content})
            await _emit("chat:message", {"role": "assistant", "content": content, "done": True})
            await _emit("status", {"action": "thinking", "done": True, "hidden": True})
            usage = self._usage(usage_meta)
            if usage:
                yield {"usage": usage}
            await _emit("chat:finish", {"role": "assistant", "content": content, "done": True})
            yield content
        except Exception as e:
            self.log.exception(f"streaming error: {e}")
            msg = f"Error during streaming: {e}"
            await _emit("chat:finish", {"role": "assistant", "content": msg, "done": True, "error": True})
            yield msg

    # ---------------- Main ---------------- #
    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable] = None,
        __tools__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> Union[str, Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        __metadata__ = __metadata__ or {}
        try:
            model_id = self._prepare_model_id(body.get("model", ""))
        except ValueError as ve:
            return f"Model Error: {ve}"
        try:
            self._validate()
        except ValueError as ve:
            return str(ve)
        stream = bool(body.get("stream", False)) and self.valves.STREAMING_ENABLED
        if stream and __tools__:
            # Tool calls need multi-round execution; run the agentic loop non-streaming.
            self.log.debug("Tools enabled: forcing non-streaming agentic loop")
            stream = False
        messages = body.get("messages", [])
        contents, system = self._prepare_content(messages)
        if not contents:
            return "Error: No valid message content found"
        try:
            payload = self._build_payload(body, contents, system, __metadata__, __tools__, model_id)
        except Exception as e:
            return f"Configuration error: {e}"
        if stream:
            url = self._url(f"models/{model_id}:streamGenerateContent") + "?alt=sse"
            return self._handle_stream(url, payload, __event_emitter__)
        try:
            start = time.time()
            url = self._url(f"models/{model_id}:generateContent")
            work_contents = list(contents)
            all_answers: List[str] = []
            all_thoughts: List[str] = []
            ground_metas: List[Any] = []
            usage_meta = None
            final = ""
            for _ in range(8):  # agentic tool-calling loop guard
                payload["contents"] = work_contents
                resp = await self._post(url, payload, "text generation")
                blocked = self._safety_message(resp)
                if blocked:
                    return blocked
                cands = resp.get("candidates") or []
                if not cands:
                    return "[No content generated or unexpected response structure]"
                if resp.get("usageMetadata"):
                    usage_meta = resp["usageMetadata"]
                answers, thoughts = self._split_parts(cands[0])
                all_answers.extend(answers)
                all_thoughts.extend(thoughts)
                gm = cands[0].get("groundingMetadata")
                if gm:
                    ground_metas.append(gm)
                calls = self._split_calls(cands[0])
                if not calls or not __tools__:
                    final = "".join(all_answers)
                    break
                result_parts: List[Dict[str, Any]] = []
                for cname, cargs, csig in calls:
                    if __event_emitter__:
                        try:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "action": "tool_execution",
                                        "description": f"Running tool {cname}...",
                                        "done": False,
                                    },
                                }
                            )
                        except Exception as e:
                            self.log.warning(f"status emit failed: {e}")
                    fn = self._resolve_tool(__tools__, cname)
                    if fn is None:
                        result = f"Error: tool '{cname}' is not available"
                    else:
                        try:
                            result = fn(**cargs)
                            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                                result = await result
                        except Exception as e:
                            result = f"Error executing tool '{cname}': {e}"
                    self.log.debug(f"Tool '{cname}' returned {len(str(result))} chars")
                    result_parts.append(
                        {"functionResponse": {"name": cname, "response": {"result": str(result)}}}
                    )
                if __event_emitter__:
                    try:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"action": "tool_execution", "done": True},
                            }
                        )
                    except Exception as e:
                        self.log.warning(f"status emit failed: {e}")
                # Replay the model's parts verbatim so thought signatures
                # survive exactly as issued (Gemini 3 validates them).
                work_contents.append({"role": "model", "parts": self._replay_model_parts(cands[0])})
                work_contents.append({"role": "user", "parts": result_parts})
            else:
                final = "".join(all_answers)
            if not final and not all_thoughts:
                return "[No content generated or unexpected response structure]"
            if ground_metas and __event_emitter__:
                final = await self._process_grounding(ground_metas, final, __event_emitter__) or final
            content = ""
            if all_thoughts:
                dur = int(max(0, time.time() - start))
                quoted = "\n".join(f"> {l}" for l in "".join(all_thoughts).strip().split("\n"))
                content += f"<details>\n<summary>Thought ({dur}s)</summary>\n\n{quoted}\n\n</details>"
            content += final
            usage = self._usage(usage_meta)
            if usage and __event_emitter__:
                try:
                    await __event_emitter__({"type": "usage", "data": usage})
                except Exception as e:
                    self.log.warning(f"usage emit failed: {e}")
            return content or "[No content generated]"
        except RuntimeError as e:
            return str(e)
        except Exception as e:
            self.log.exception(f"request failed: {e}")
            return f"An error occurred while processing your request: {e}"
