"""
title: Import Test Batch 3 - Extra Packages
author: test
version: 0.0.1
description: Test extra package imports (aiofiles, PIL, cryptography, pydantic_core).
"""

import os
from typing import List, Dict, Any, Union, Optional, AsyncIterator, Callable
from pydantic import BaseModel, Field
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
import aiofiles
from PIL import Image
from cryptography.fernet import Fernet, InvalidToken


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="", description="API key")

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Import Test B3"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "test-b3", "name": "Test Batch 3"}]

    async def pipe(self, body: Dict[str, Any], __metadata__: dict, __event_emitter__: Callable, __tools__: dict | None = None, __request__=None, __user__: dict | None = None) -> str:
        return "[Batch 3] Extra packages imports OK"
