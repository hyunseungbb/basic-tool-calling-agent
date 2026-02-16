"""임베딩 생성 모듈.

sentence-transformers 기반 EmbeddingProvider를 제공한다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from agent.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """임베딩 생성 추상 인터페이스."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩 벡터로 변환한다."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 임베딩 벡터 리스트로 변환한다."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """임베딩 벡터 차원."""


class SentenceTransformerProvider(EmbeddingProvider):
    """sentence-transformers 기반 임베딩."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._model = None  # lazy loading

    def _load_model(self):
        """모델을 지연 로딩한다."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("임베딩 모델 로딩: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("임베딩 모델 로딩 완료 (dim=%d)", self.dimension)

    def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩한다."""
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩을 수행한다."""
        self._load_model()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        """임베딩 차원을 반환한다."""
        return settings.embedding_dim


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """싱글턴 EmbeddingProvider 인스턴스를 반환한다."""
    global _provider
    if _provider is None:
        _provider = SentenceTransformerProvider()
    return _provider

