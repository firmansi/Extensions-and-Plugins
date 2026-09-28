"""
title: Import Test Batch 1 - Core + Google GenAI
author: test
version: 0.0.1
description: Test core imports + google genai only.
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
from typing import List, Union, Optional, Dict, Any, Tuple, AsyncIterator, Callable
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="", description="API key")

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Import Test B1"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "test-b1", "name": "Test Batch 1"}]

    async def pipe(self, body: Dict[str, Any], __metadata__: dict, __event_emitter__: Callable, __tools__: dict | None = None, __request__=None, __user__: dict | None = None) -> str:
        return "[Batch 1] Core + Google GenAI imports OK"
