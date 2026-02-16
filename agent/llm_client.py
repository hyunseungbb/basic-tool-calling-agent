"""LLM 클라이언트 추상 인터페이스 및 Anthropic 구현체."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import anthropic

from agent.config import settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLM 클라이언트 추상 인터페이스.

    벤더 교체를 용이하게 하기 위해 추상화한다.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """텍스트 생성.

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트

        Returns:
            생성된 텍스트
        """

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """JSON 형식으로 텍스트 생성.

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트

        Returns:
            파싱된 JSON 딕셔너리
        """


class AnthropicClient(LLMClient):
    """Anthropic Claude 기반 LLM 클라이언트."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Claude를 사용하여 텍스트를 생성한다."""
        logger.debug("LLM generate 호출: model=%s", self._model)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text
        logger.debug("LLM 응답 길이: %d chars", len(text))
        return text

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Claude를 사용하여 JSON 응답을 생성한다.

        응답에서 JSON 블록을 추출하여 파싱한다.
        """
        raw = self.generate(system_prompt, user_prompt)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """텍스트에서 JSON 객체를 추출한다.

        ```json ... ``` 블록 또는 { ... } 형태를 모두 처리한다.
        """
        # ```json ... ``` 블록 추출 시도
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            json_str = text[start:end].strip()
        else:
            # 중괄호로 시작/끝나는 부분 추출
            first_brace = text.index("{")
            last_brace = text.rindex("}") + 1
            json_str = text[first_brace:last_brace]

        return json.loads(json_str)


def get_llm_client() -> LLMClient:
    """기본 LLM 클라이언트 인스턴스를 반환한다."""
    return AnthropicClient()

