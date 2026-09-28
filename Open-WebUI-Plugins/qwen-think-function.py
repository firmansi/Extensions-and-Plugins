"""
title: Enable Thinking
author: delbertbeta
author_url: https://github.com/delbertbeta
funding_url: https://github.com/open-webui
version: 0.1.0
description: Enable Thinking
"""

"""
Support toggle { "enable_thinking": true } to the request body to enable thinking.
thinking_budget setting is supported.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Callable, Awaitable


class Filter:
    class Valves(BaseModel):
        priority: int = 2

        thinking_budget: int = Field(default=5000, description="max reasoning length")
        pass  # No user-configurable valves as per the requested structure

    def __init__(self):

        self.valves = self.Valves()

        self.toggle: bool = True

        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBjbGFzcz0ic2l6ZS02Ij4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAxOHYtNS4yNW0wIDBhNi4wMSA2LjAxIDAgMCAwIDEuNS0uMTg5bS0xLjUuMTg5YTYuMDEgNi4wMSAwIDAgMS0xLjUtLjE4OW0zLjc1IDcuNDc4YTEyLjA2IDEyLjA2IDAgMCAxLTQuNSAwbTMuNzUgMi4zODNhMTQuNDA2IDE0LjQwNiAwIDAgMS0zIDBNMTQuMjUgMTh2LS4xOTJjMC0uOTgzLjY1OC0xLjgyMyAxLjUwOC0yLjMxNmE3LjUgNy41IDAgMSAwLTcuNTE3IDBjLjg1LjQ5MyAxLjUwOSAxLjMzMyAxLjUwOSAyLjMxNlYxOCIgLz4KPC9zdmc+Cg=="""

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
    ) -> Dict[str, Any]:

        if self.toggle:
            body["enable_thinking"] = True
            if self.valves.thinking_budget > 0:
                body["thinking_budget"] = self.valves.thinking_budget

        return body

    async def outlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return body

