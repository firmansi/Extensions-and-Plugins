"""
title: Import Test Batch 2 - OWUI Internals
author: test
version: 0.0.1
description: Test OpenWebUI internal imports only.
"""

import os
from typing import List, Dict, Any, Union, Optional, AsyncIterator, Callable
from pydantic import BaseModel, Field
from open_webui.env import SRC_LOG_LEVELS
from open_webui.internal.db import get_async_db_context
from fastapi import Request, UploadFile, BackgroundTasks
from open_webui.routers.files import upload_file
from open_webui.models.chats import Chats
from open_webui.models.users import UserModel, Users
from starlette.datastructures import Headers


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="", description="API key")

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Import Test B2"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "test-b2", "name": "Test Batch 2"}]

    async def pipe(self, body: Dict[str, Any], __metadata__: dict, __event_emitter__: Callable, __tools__: dict | None = None, __request__=None, __user__: dict | None = None) -> str:
        return "[Batch 2] OWUI internals imports OK"
