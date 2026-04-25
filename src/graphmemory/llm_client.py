"""LLM client abstraction for Graphmemory."""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
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
        reasoning_effort: str = "",
        disable_thinking: bool = False,
        use_extra_body_thinking: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort or None
        self.disable_thinking = disable_thinking
        self.use_extra_body_thinking = use_extra_body_thinking
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
        payload_messages = self._prepare_messages(messages)

        # DashScope Qwen3: disable thinking via extra_body instead of /no_think prefix
        extra_body: Optional[Dict] = None
        if self.disable_thinking and self.use_extra_body_thinking:
            extra_body = {"enable_thinking": False}

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=payload_messages,
                    response_format=response_format,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    stop=stop or None,
                    extra_body=extra_body,
                )
                content = resp.choices[0].message.content or ""
                return self._normalize_content(content)
            except Exception as exc:
                logger.error(f"LLM attempt {attempt + 1}/{self.max_retries}: {exc}")
                if attempt < self.max_retries - 1:
                    wait = 2**attempt + random.random()
                    logger.info(f"Retrying in {wait:.1f}s…")
                    time.sleep(wait)

        logger.error("Max retries reached, returning empty string.")
        return ""

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        payload = deepcopy(messages)
        if not self.disable_thinking:
            return payload

        marker = "/no_think"
        if payload and payload[0].get("role") == "system":
            content = payload[0].get("content", "")
            if isinstance(content, str) and marker not in content:
                payload[0]["content"] = f"{marker}\n{content}"
            return payload

        return [{"role": "system", "content": marker}] + payload

    def _normalize_content(self, content: str) -> str:
        if self.disable_thinking and "</think>" in content:
            _, tail = content.rsplit("</think>", 1)
            cleaned = tail.strip()
            if cleaned:
                return cleaned
        return content
