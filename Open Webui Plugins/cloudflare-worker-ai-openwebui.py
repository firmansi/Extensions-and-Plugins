"""
title: CloudFlare AI Gateway Pipe
authors: Chris Bloem
version: 1.1.0
required_open_webui_version: 0.1.6
license: MIT
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        CF_ACCOUNT_ID: str = Field(default="")
        CF_API_KEY: str = Field(default="")
        CF_AI_Gateway: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "cloudflare"
        self.name = "cloudflare/"
        self.valves = self.Valves(
            **{
                "CF_ACCOUNT_ID": os.getenv("CF_ACCOUNT_ID", ""),
                "CF_API_KEY": os.getenv("CF_API_KEY", ""),
                "CF_AI_Gateway": os.getenv("CF_AI_Gateway", ""),
            }
        )

        # Approximate context windows (prompt + response) per model.
        # These are chosen to match Cloudflare model limits reasonably well.
        self.MODEL_CONTEXT_WINDOWS = {
            "@cf/openai/gpt-oss-120b": 128000,
            "@cf/openai/gpt-oss-20b": 128000,
            "@cf/meta/llama-3.1-8b-instruct": 128000,
            "@cf/meta/llama-2-7b-chat-fp16": 4096,
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast": 24000,
            "@cf/meta/llama-3.2-11b-vision-instruct": 128000,
            "@hf/thebloke/deepseek-coder-6.7b-instruct-awq": 4096,
        }

    def get_cloudflare_models(self):
        return [
            {"id": "@cf/openai/gpt-oss-120b", "name": "gpt-oss-120b"},
            {"id": "@cf/openai/gpt-oss-20b", "name": "gpt-oss-20b"},
            {"id": "@cf/meta/llama-3.1-8b-instruct", "name": "llama3.1:8b"},
            {"id": "@cf/meta/llama-2-7b-chat-fp16", "name": "llama2:7b"},
            {"id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "name": "llama3.3:70b"},
            {"id": "@cf/meta/llama-3.2-11b-vision-instruct", "name": "llama3.2:11b"},
            {
                "id": "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
                "name": "deepseek:6.7b",
            },
        ]

    def pipes(self) -> List[dict]:
        return self.get_cloudflare_models()

    def _messages_to_input_text(self, system_message, messages: List[dict]) -> str:
        parts = []
        if system_message:
            parts.append(f"system: {system_message}")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _get_context_window(self, model_id: str) -> int:
        """
        Return the configured context window for the model.
        Fallback to 16384 if unknown.
        """
        return self.MODEL_CONTEXT_WINDOWS.get(model_id, 16384)

    def _approx_token_count(self, text: str) -> int:
        """
        Very rough heuristic: ~4 chars per token.
        Good enough to avoid blowing past hard limits.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _compute_max_output_tokens(
        self,
        prompt_tokens: int,
        context_window: int,
        requested_max: Optional[int],
        default_cap: int = 4096,
    ) -> int:
        """
        Ensure requested max tokens don't exceed the model's context window:

            prompt_tokens + max_output_tokens <= context_window

        If user didn't specify, pick a sane default and clamp.
        """
        if context_window <= 0:
            context_window = 16384

        if requested_max is not None:
            try:
                max_tokens = int(requested_max)
            except (TypeError, ValueError):
                max_tokens = default_cap
        else:
            # Default: up to default_cap, but never more than 1/4 of window
            max_tokens = min(default_cap, max(256, context_window // 4))

        if max_tokens < 1:
            max_tokens = 1

        # Clamp so we don't exceed context window when combined with prompt
        if prompt_tokens + max_tokens > context_window:
            max_tokens = max(1, context_window - prompt_tokens)

        return max_tokens

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])

        model_raw = body["model"]

        if "cloudfare_workerai." in model_raw:
            model_id = model_raw.split("cloudfare_workerai.")[-1]
        elif "cloudflare_workerai." in model_raw:
            model_id = model_raw.split("cloudflare_workerai.")[-1]
        else:
            model_id = model_raw

        is_gpt_oss = model_id.startswith("@cf/openai/gpt-oss")

        context_window = self._get_context_window(model_id)

        url = (
            f"https://gateway.ai.cloudflare.com/v1/"
            f"{self.valves.CF_ACCOUNT_ID}/{self.valves.CF_AI_Gateway}/workers-ai/{model_id}"
        )

        headers = {
            "Authorization": f"Bearer {self.valves.CF_API_KEY}",
            "Content-Type": "application/json",
        }

        if is_gpt_oss:
            # GPT-OSS uses Responses-style input, not chat completions.
            # We still accept Open WebUI's messages and collapse them into a single text prompt.
            input_text = self._messages_to_input_text(system_message, messages)
            prompt_tokens = self._approx_token_count(input_text)

            max_output_tokens = self._compute_max_output_tokens(
                prompt_tokens=prompt_tokens,
                context_window=context_window,
                requested_max=body.get("max_tokens"),
                default_cap=8192,
            )

            payload = {
                "input": input_text,
                "max_output_tokens": max_output_tokens,
            }

            # For GPT-OSS, "instructions" is the system prompt field.
            if system_message:
                payload["instructions"] = str(system_message)

            body_stream = False

        else:
            msg_text = self._messages_to_input_text(system_message, messages)
            prompt_tokens = self._approx_token_count(msg_text)

            max_tokens = self._compute_max_output_tokens(
                prompt_tokens=prompt_tokens,
                context_window=context_window,
                requested_max=body.get("max_tokens"),
                default_cap=16384,
            )

            payload = {
                "messages": messages,
                "stream": body.get("stream", False),
                "max_tokens": max_tokens,
                "temperature": body.get("temperature", 0.2),
                "top_p": body.get("top_p", 1),
                "top_k": body.get("top_k", 1),
                "repetition_penalty": body.get("repetition_penalty", 1),
                "frequency_penalty": body.get("frequency_penalty", 0),
                "presence_penalty": body.get("presence_penalty", 0),
            }

            if system_message:
                payload["messages"].insert(
                    0, {"role": "system", "content": str(system_message)}
                )

            body_stream = body.get("stream", False)

        if "seed" in body:
            payload["seed"] = body["seed"]
        if "raw" in body:
            payload["raw"] = body["raw"]

        try:
            if body_stream and not is_gpt_oss:
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(
        self, url: str, headers: dict, payload: dict
    ) -> Generator[str, None, None]:
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for chunk in response.iter_lines():
                    if not chunk:
                        continue

                    line = chunk.decode("utf-8")

                    if not line.startswith("data: "):
                        continue

                    if line.strip() == "data: [DONE]":
                        break

                    json_str = line[6:]
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(data, dict) and "response" in data:
                        piece = data["response"]
                        if piece is None:
                            continue
                        yield str(piece)
                    else:
                        # unknown shape – yield raw JSON chunk as text
                        yield json_str
        except Exception as e:
            yield f"Error: {str(e)}"

    def non_stream_response(self, url: str, headers: dict, payload: dict) -> str:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        inner = data
        if isinstance(data, dict) and isinstance(data.get("result"), dict):
            inner = data["result"]

        if isinstance(inner, dict) and "response" in inner:
            val = inner["response"]
            return "" if val is None else str(val)

        if isinstance(inner, dict) and "output" in inner:
            try:
                outputs = inner.get("output", [])
                texts: list[str] = []

                for output in outputs:
                    if not isinstance(output, dict):
                        continue
                    content = output.get("content", [])
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") in ("output_text", "text"):
                            txt = part.get("text", "")
                            if txt:
                                texts.append(txt)

                if texts:
                    return "".join(texts)
            except Exception:
                pass

        try:
            return json.dumps(inner)
        except TypeError:
            try:
                return json.dumps(data)
            except TypeError:
                return str(inner)
