from typing import Any

from ._registry import _PREPROCESS_REGISTRY


class Embedder:
    """
    임베딩 모델(text-embedding-3-small)을 호출하고,
    임베딩 수행 전 전처리(Narrativized, JSON 변환 등)를 담당합니다.
    Registry Pattern을 사용하여 elif 체인 없이 전략 클래스를 동적으로 확장합니다.
    """

    def __init__(self, openai_service: Any | None = None):
        self.openai_service = openai_service

    def preprocess(
        self,
        row: dict,
        method: str = "kv_pairs",
        exclude_fields: list[str] | None = None,
    ) -> str | list[str]:
        """
        임베딩을 수행하기 전, 데이터를 특정 방법론에 따라 전처리합니다.
        Registry에서 전략 클래스를 조회하여 인스턴스화 후 process()를 호출합니다.
        """
        if method not in _PREPROCESS_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing method: '{method}'. "
                f"Available: {list(_PREPROCESS_REGISTRY.keys())}"
            )
        strategy = _PREPROCESS_REGISTRY[method](openai_service=self.openai_service)
        return strategy.process(row, exclude_fields=exclude_fields)

    def embed(self, text: str) -> list[float]:
        """
        OpenAI (text-embedding-3-small)을 통해 단일 텍스트를 벡터로 변환합니다.
        """
        if not self.openai_service:
            raise ValueError("openai_service is not initialized.")
        return self.openai_service.get_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        OpenAI (text-embedding-3-small)을 통해 다중 텍스트를 한 번에 벡터로 변환합니다.
        데이터 적재 시간을 획기적으로 줄여줍니다.
        """
        if not self.openai_service:
            raise ValueError("openai_service is not initialized.")
        return self.openai_service.get_embeddings(texts)
