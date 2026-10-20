"""
title: Diagnostic Pipe
author: dfirmans
version: 1.0.0
required_open_webui_version: 0.6.0
description: Minimal pipe to test whether function creation works at all. No requirements, stdlib plus pydantic only.
"""

from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        GREETING: str = Field(default="Hello")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": "test", "name": "Diagnostic Test"}]

    async def pipe(self, body: dict):
        return f"{self.valves.GREETING}, diagnostic OK. Model: {body.get('model')}"
