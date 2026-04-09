from typing import Any, Protocol

import numpy as np
from omegaconf import DictConfig

# ==========================================
# FAISS Index Protocol
# ==========================================


class _FaissIndex(Protocol):
    """FAISS IndexFlatIP의 numpy 편의 API(class_wrappers 런타임 패치 버전)를 나타내는 구조적 타입."""

    ntotal: int

    def add(self, x: np.ndarray) -> None: ...

    def search(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def reconstruct(self, key: int) -> np.ndarray: ...


# ==========================================
# Retrieval Strategy Registry
# ==========================================

RETRIEVAL_STRATEGIES: dict[str, type["BaseRetrievalStrategy"]] = {}


def register_strategy(name: str):
    """Retrieval Strategy 등록을 위한 데코레이터"""

    def decorator(cls: type["BaseRetrievalStrategy"]) -> type["BaseRetrievalStrategy"]:
        RETRIEVAL_STRATEGIES[name] = cls
        return cls

    return decorator


class BaseRetrievalStrategy:
    """Retrieval 전략 생성을 위한 베이스 클래스"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.top_k = config.get("top_k", 5)

    def post_add_documents(self, documents: list[dict[str, Any]]) -> None:
        """add_documents() 완료 후 호출되는 hook. 기본 구현은 no-op."""

    def set_query_tokens(self, tokens: list[str]) -> None:
        """search() 직전에 query 토큰을 주입하는 hook. 기본 구현은 no-op."""

    def set_query_text(self, text: str) -> None:
        """search() 직전에 원본 query 텍스트를 주입하는 hook. 기본 구현은 no-op.
        polarity/template 감지 등 원문이 필요한 전략에서 오버라이드합니다."""

    def search(
        self,
        index: _FaissIndex,
        documents: list[dict[str, Any]],
        query_np: np.ndarray,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("서브타입에서 search를 구현해야 합니다.")
