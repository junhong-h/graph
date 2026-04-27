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
        top_p: float | None = None,
        seed: int | None = None,
        max_retries: int = 5,
        reasoning_effort: str = "",
        disable_thinking: bool = False,
        use_extra_body_thinking: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
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
        if json_mode:
            payload_messages = self._ensure_json_instruction(payload_messages)

        # DashScope Qwen3: disable thinking via extra_body instead of /no_think prefix
        extra_body: Optional[Dict] = None
        if self.disable_thinking and self.use_extra_body_thinking:
            extra_body = {"enable_thinking": False}

        unsupported_seed = False

        attempt = 0
        while attempt < self.max_retries:
            try:
                request: Dict = {
                    "model": self.model,
                    "messages": payload_messages,
                    "response_format": response_format,
                    "temperature": self.temperature,
                    "reasoning_effort": self.reasoning_effort,
                    "stop": stop or None,
                }
                if self.top_p is not None:
                    request["top_p"] = self.top_p
                if self.seed is not None and not unsupported_seed:
                    request["seed"] = self.seed
                if extra_body is not None:
                    request["extra_body"] = extra_body

                resp = self.client.chat.completions.create(**request)
                content = resp.choices[0].message.content or ""
                return self._normalize_content(content)
            except Exception as exc:
                if self.seed is not None and not unsupported_seed and "seed" in str(exc).lower():
                    unsupported_seed = True
                    logger.warning("LLM provider rejected seed; retrying without seed.")
                    continue
                logger.error(f"LLM attempt {attempt + 1}/{self.max_retries}: {exc}")
                if attempt < self.max_retries - 1:
                    wait = self._retry_wait_seconds(exc, attempt)
                    logger.info(f"Retrying in {wait:.1f}s…")
                    time.sleep(wait)
                attempt += 1

        logger.error("Max retries reached, returning empty string.")
        return ""

    def _retry_wait_seconds(self, exc: Exception, attempt: int) -> float:
        message = str(exc).lower()
        if "429" in message or "token-limit" in message or "insufficient_quota" in message:
            return min(90.0, 15.0 * (attempt + 1) + random.random() * 3.0)
        return 2**attempt + random.random()

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

    def _ensure_json_instruction(self, messages: List[Dict]) -> List[Dict]:
        """DashScope JSON Mode requires the prompt to contain the word JSON."""
        if any("json" in str(m.get("content", "")).lower() for m in messages):
            return messages
        payload = deepcopy(messages)
        instruction = "Return valid JSON only."
        if payload and payload[0].get("role") == "system":
            content = payload[0].get("content", "")
            if isinstance(content, str):
                payload[0]["content"] = f"{content}\n\n{instruction}"
                return payload
        return [{"role": "system", "content": instruction}] + payload

    def _normalize_content(self, content: str) -> str:
        if self.disable_thinking and "</think>" in content:
            _, tail = content.rsplit("</think>", 1)
            cleaned = tail.strip()
            if cleaned:
                return cleaned
        return content
