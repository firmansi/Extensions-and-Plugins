"""
title: CloudFlare AI Gateway Pipe
authors: Chris Bloem
version: 1.2.0
required_open_webui_version: 0.1.6
license: MIT
"""

import os
import requests
import json
import logging
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message

# Configure logging
log = logging.getLogger(__name__)


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
        # Updated for latest Cloudflare model lineup (May 2026)
        self.MODEL_CONTEXT_WINDOWS = {
            # OpenAI GPT-OSS Models
            "@cf/openai/gpt-oss-120b": 128000,
            "@cf/openai/gpt-oss-20b": 128000,
            # Meta Llama Models
            "@cf/meta/llama-3.1-8b-instruct": 128000,
            "@cf/meta/llama-3.2-1b": 8192,
            "@cf/meta/llama-2-7b-chat-fp16": 4096,
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast": 24000,
            "@cf/meta/llama-3.2-11b-vision-instruct": 128000,
            # Qwen Models
            "@cf/qwen/qwen2-57b-a14b": 32768,
            # Mistral Models
            "@cf/mistral/mistral-7b-instruct-v0.1": 8000,
            # DeepSeek Models
            "@hf/thebloke/deepseek-coder-6.7b-instruct-awq": 4096,
        }

    def get_cloudflare_models(self) -> List[dict]:
        """
        Return list of available Cloudflare AI models.
        
        :return: List of model dictionaries with id and name
        """
        return [
            {"id": "@cf/openai/gpt-oss-120b", "name": "gpt-oss-120b"},
            {"id": "@cf/openai/gpt-oss-20b", "name": "gpt-oss-20b"},
            {"id": "@cf/meta/llama-3.1-8b-instruct", "name": "llama3.1:8b"},
            {"id": "@cf/meta/llama-3.2-1b", "name": "llama3.2:1b"},
            {"id": "@cf/meta/llama-2-7b-chat-fp16", "name": "llama2:7b"},
            {"id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "name": "llama3.3:70b"},
            {"id": "@cf/meta/llama-3.2-11b-vision-instruct", "name": "llama3.2:11b"},
            {"id": "@cf/qwen/qwen2-57b-a14b", "name": "qwen2:57b"},
            {"id": "@cf/mistral/mistral-7b-instruct-v0.1", "name": "mistral:7b"},
            {
                "id": "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
                "name": "deepseek:6.7b",
            },
        ]

    def pipes(self) -> List[dict]:
        """Return available pipes/models."""
        return self.get_cloudflare_models()

    def _messages_to_input_text(
        self, system_message: Optional[dict], messages: List[dict]
    ) -> str:
        """
        Convert message list to plain text format.
        
        :param system_message: Optional system message dict
        :param messages: List of message dictionaries
        :return: Formatted text with role prefixes
        """
        parts = []
        if system_message:
            content = system_message.get("content", "")
            parts.append(f"system: {content}")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            # Handle both string and list content formats
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content = " ".join(text_parts)
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _get_context_window(self, model_id: str) -> int:
        """
        Return the configured context window for the model.
        
        :param model_id: Model identifier
        :return: Context window size in tokens
        """
        return self.MODEL_CONTEXT_WINDOWS.get(model_id, 16384)

    def _approx_token_count(self, text: str) -> int:
        """
        Estimate token count using improved heuristic.
        
        Uses ~4 characters per token (reasonable average).
        Fallback to at least 1 token for empty strings.
        
        :param text: Text to count
        :return: Approximate token count
        """
        if not text:
            return 0
        # Improved estimation: count words and special chars
        # Average: ~1 word = ~1.3 tokens, ~1 space = ~0.3 tokens
        word_count = len(text.split())
        char_count = len(text)
        # More accurate: chars/4 + word_count*0.3 for overhead
        estimated = max(1, (char_count // 4) + (word_count // 3))
        return estimated

    def _compute_max_output_tokens(
        self,
        prompt_tokens: int,
        context_window: int,
        requested_max: Optional[int],
        default_cap: int = 4096,
    ) -> int:
        """
        Compute safe max output tokens considering context window.
        
        Ensures: prompt_tokens + max_output_tokens <= context_window
        
        :param prompt_tokens: Estimated tokens in prompt
        :param context_window: Model's total context window
        :param requested_max: User-requested max tokens (optional)
        :param default_cap: Default maximum if not requested
        :return: Safe maximum output tokens
        """
        if context_window <= 0:
            context_window = 16384

        if requested_max is not None:
            try:
                max_tokens = int(requested_max)
            except (TypeError, ValueError):
                log.warning(f"Invalid max_tokens value: {requested_max}, using default")
                max_tokens = default_cap
        else:
            # Default: up to default_cap, but never more than 1/4 of window
            max_tokens = min(default_cap, max(256, context_window // 4))

        if max_tokens < 1:
            max_tokens = 1

        # Clamp so we don't exceed context window when combined with prompt
        if prompt_tokens + max_tokens > context_window:
            max_tokens = max(1, context_window - prompt_tokens)
            log.warning(
                f"Max tokens clamped: prompt={prompt_tokens}, "
                f"window={context_window}, final_max={max_tokens}"
            )

        return max_tokens

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        """
        Main pipe function for handling chat requests.
        
        :param body: Request body containing messages, model, and parameters
        :return: Response string or generator for streaming
        """
        try:
            # Extract system and user messages
            if "messages" not in body:
                log.error("Missing 'messages' field in request body")
                return "Error: Missing 'messages' field in request"

            system_message, messages = pop_system_message(body["messages")]

            # Extract and normalize model ID
            model_raw = body.get("model", "")
            if not model_raw:
                log.error("Missing 'model' field in request body")
                return "Error: Missing 'model' field in request"

            # Handle model ID prefixes (support both typo and correct spellings)
            if "cloudfare_workerai." in model_raw:
                log.warning("Using deprecated 'cloudfare_workerai' prefix")
                model_id = model_raw.split("cloudfare_workerai.")[-1]
            elif "cloudflare_workerai." in model_raw:
                model_id = model_raw.split("cloudflare_workerai.")[-1]
            else:
                model_id = model_raw

            log.info(f"Processing request for model: {model_id}")

            # Determine model type
            is_gpt_oss = model_id.startswith("@cf/openai/gpt-oss")
            context_window = self._get_context_window(model_id)

            # Build request URL
            url = (
                f"https://gateway.ai.cloudflare.com/v1/"
                f"{self.valves.CF_ACCOUNT_ID}/{self.valves.CF_AI_Gateway}/workers-ai/{model_id}"
            )

            headers = {
                "Authorization": f"Bearer {self.valves.CF_API_KEY}",
                "Content-Type": "application/json",
            }

            # Build payload based on model type
            if is_gpt_oss:
                payload = self._build_gpt_oss_payload(
                    system_message, messages, body, context_window
                )
                body_stream = False
            else:
                payload = self._build_chat_payload(
                    system_message, messages, body, context_window
                )
                body_stream = body.get("stream", False)

            # Add optional fields
            if "seed" in body:
                payload["seed"] = body["seed"]
            if "raw" in body:
                payload["raw"] = body["raw"]

            # Execute request
            try:
                if body_stream and not is_gpt_oss:
                    return self.stream_response(url, headers, payload)
                else:
                    return self.non_stream_response(url, headers, payload)
            except requests.exceptions.ConnectionError as e:
                log.error(f"Connection error to Cloudflare API: {e}")
                return f"Error: Connection failed - {str(e)}"
            except requests.exceptions.Timeout as e:
                log.error(f"Request timeout to Cloudflare API: {e}")
                return f"Error: Request timeout - {str(e)}"
            except requests.exceptions.RequestException as e:
                log.error(f"Request failed: {e}")
                return f"Error: Request failed - {str(e)}"

        except KeyError as e:
            log.error(f"Missing required field: {e}")
            return f"Error: Missing required field: {e}"
        except Exception as e:
            log.exception(f"Unexpected error in pipe method: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _build_gpt_oss_payload(
        self,
        system_message: Optional[dict],
        messages: List[dict],
        body: dict,
        context_window: int,
    ) -> dict:
        """
        Build payload for GPT-OSS models (Responses API format).
        
        :param system_message: System message
        :param messages: User messages
        :param body: Original request body
        :param context_window: Model context window
        :return: Formatted payload for GPT-OSS
        """
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

        # For GPT-OSS, "instructions" is the system prompt field
        if system_message:
            payload["instructions"] = str(system_message.get("content", ""))

        return payload

    def _build_chat_payload(
        self,
        system_message: Optional[dict],
        messages: List[dict],
        body: dict,
        context_window: int,
    ) -> dict:
        """
        Build payload for standard chat completion models.
        
        :param system_message: System message
        :param messages: User messages
        :param body: Original request body
        :param context_window: Model context window
        :return: Formatted payload for chat completion
        """
        msg_text = self._messages_to_input_text(system_message, messages)
        prompt_tokens = self._approx_token_count(msg_text)

        max_tokens = self._compute_max_output_tokens(
            prompt_tokens=prompt_tokens,
            context_window=context_window,
            requested_max=body.get("max_tokens"),
            default_cap=16384,
        )

        # Start with base message payload
        payload_messages = messages.copy()

        # Insert system message if present
        if system_message:
            payload_messages.insert(0, system_message)

        payload = {
            "messages": payload_messages,
            "stream": body.get("stream", False),
            "max_tokens": max_tokens,
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 1),
            "top_k": body.get("top_k", 1),
            "repetition_penalty": body.get("repetition_penalty", 1),
            "frequency_penalty": body.get("frequency_penalty", 0),
            "presence_penalty": body.get("presence_penalty", 0),
        }

        return payload

    def stream_response(
        self, url: str, headers: dict, payload: dict
    ) -> Generator[str, None, None]:
        """
        Stream response from Cloudflare API.
        
        :param url: API endpoint URL
        :param headers: Request headers
        :param payload: Request payload
        :yield: Text chunks from API response
        """
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=300
            ) as response:
                if response.status_code != 200:
                    error_msg = f"HTTP Error {response.status_code}: {response.text}"
                    log.error(error_msg)
                    yield f"Error: {error_msg}"
                    return

                for chunk in response.iter_lines():
                    if not chunk:
                        continue

                    try:
                        line = chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        log.warning("Failed to decode chunk, skipping")
                        continue

                    if not line.startswith("data: "):
                        continue

                    if line.strip() == "data: [DONE]":
                        break

                    json_str = line[6:]
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        log.debug(f"Failed to parse JSON: {json_str}")
                        continue

                    # Extract text from response
                    if isinstance(data, dict):
                        # Handle different response formats
                        if "response" in data:
                            piece = data.get("response")
                            if piece is not None:
                                yield str(piece)
                        elif "delta" in data and "content" in data["delta"]:
                            piece = data["delta"]["content"]
                            if piece is not None:
                                yield str(piece)
                        elif "choices" in data:
                            for choice in data.get("choices", []):
                                if "delta" in choice:
                                    content = choice["delta"].get("content")
                                    if content:
                                        yield str(content)
        except requests.exceptions.Timeout:
            log.error("Stream response timeout")
            yield "Error: Request timeout during streaming"
        except Exception as e:
            log.exception(f"Error in stream_response: {e}")
            yield f"Error: {str(e)}"

    def non_stream_response(self, url: str, headers: dict, payload: dict) -> str:
        """
        Get non-streaming response from Cloudflare API.
        
        :param url: API endpoint URL
        :param headers: Request headers
        :param payload: Request payload
        :return: Response text
        """
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            log.debug(f"API Response: {data}")

            # Navigate response structure
            inner = data
            if isinstance(data, dict) and isinstance(data.get("result"), dict):
                inner = data["result"]

            # Try different response formats
            if isinstance(inner, dict) and "response" in inner:
                val = inner["response"]
                return "" if val is None else str(val)

            if isinstance(inner, dict) and "output" in inner:
                return self._extract_output_text(inner.get("output", []))

            # Handle choices array (OpenAI format)
            if isinstance(inner, dict) and "choices" in inner:
                for choice in inner.get("choices", []):
                    if "message" in choice:
                        msg = choice["message"]
                        content = msg.get("content", "")
                        if content:
                            return str(content)

            # Fallback: return JSON representation
            try:
                return json.dumps(inner)
            except TypeError:
                return str(inner)

        except requests.exceptions.Timeout:
            log.error("Non-stream response timeout")
            return "Error: Request timeout"
        except requests.exceptions.HTTPError as e:
            log.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except requests.exceptions.RequestException as e:
            log.error(f"Request error: {e}")
            return f"Error: Request failed - {str(e)}"
        except json.JSONDecodeError as e:
            log.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response - {str(e)}"
        except Exception as e:
            log.exception(f"Unexpected error in non_stream_response: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _extract_output_text(self, outputs: List[dict]) -> str:
        """
        Extract text from Responses API output format.
        
        :param outputs: List of output items from API
        :return: Extracted text content
        """
        texts: List[str] = []

        try:
            for output in outputs:
                if not isinstance(output, dict):
                    continue

                content = output.get("content", [])
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    # Handle output_text and text types
                    if part.get("type") in ("output_text", "text"):
                        txt = part.get("text", "")
                        if txt:
                            texts.append(txt)

            return "".join(texts) if texts else ""

        except Exception as e:
            log.warning(f"Error extracting output text: {e}")
            return ""