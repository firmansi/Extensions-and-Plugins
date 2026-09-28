"""
title: Google Gemini Pipeline Import Test
author: test
version: 0.0.2
required_open_webui_version: 0.11.4
description: Test pipeline to verify all imports are available in the OWUI environment.
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


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Google Gemini Import Test"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "gemini-import-test", "name": "Gemini Import Test"}]

    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: dict[str, Any],
        __event_emitter__: Callable,
        __tools__: dict[str, Any] | None = None,
        __request__=None,
        __user__: Optional[dict] = None,
    ) -> Union[str, Dict[str, Any], AsyncIterator[Union[str, Dict[str, Any]]]]:
        tools_count = len(__tools__) if __tools__ else 0
        tool_names = list(__tools__.keys()) if __tools__ else []
        return f"[Import Test] All imports OK. Tools: {tools_count}. Names: {tool_names[:5]}"
