"""LLM client abstraction for Graphmemory."""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from loguru import logger
from openai import OpenAI


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        messages: List[Dict],
        json_mode: bool = False,
        stop: Optional[List[str]] = None,
    ) -> str: ...


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        max_retries: int = 5,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key or None,
            base_url=base_url or None,
        )

    def complete(
        self,
        messages: List[Dict],
        json_mode: bool = False,
        stop: Optional[List[str]] = None,
    ) -> str:
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=self.temperature,
                    stop=stop or None,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                logger.error(f"LLM attempt {attempt + 1}/{self.max_retries}: {exc}")
                if attempt < self.max_retries - 1:
                    wait = 2**attempt + random.random()
                    logger.info(f"Retrying in {wait:.1f}s…")
                    time.sleep(wait)

        logger.error("Max retries reached, returning empty string.")
        return ""
