"""
title: Google Gemini Pipeline Test
author: test
version: 0.0.1
required_open_webui_version: 0.11.4
description: Minimal test pipeline to verify OWUI can load the Pipe class structure.
"""

import os
import json
import logging
from typing import List, Dict, Any, Union, Optional, AsyncIterator, Callable
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Google Gemini Test"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "gemini-test", "name": "Gemini Test Model"}]

    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: dict[str, Any],
        __event_emitter__: Callable,
        __tools__: dict[str, Any] | None = None,
        __request__=None,
        __user__: Optional[dict] = None,
    ) -> Union[str, Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        messages = body.get("messages", [])
        last_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    last_msg = content
                elif isinstance(content, list):
                    last_msg = " ".join(
                        item.get("text", "") for item in content if item.get("type") == "text"
                    )
                break

        tools_info = f"Tools available: {len(__tools__) if __tools__ else 0}"
        response = f"[Test Pipeline] Received: '{last_msg[:100]}'. {tools_info}. Pipeline loaded successfully!"

        return response
