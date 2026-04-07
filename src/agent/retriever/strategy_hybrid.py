from typing import Any

import numpy as np

from ._registry import BaseRetrievalStrategy, _FaissIndex, register_strategy


@register_strategy("hybrid")
class HybridRetrievalStrategy(BaseRetrievalStrategy):
    """4차 개선: Dense(FAISS) + Sparse(BM25/Keyword) 조합 검색
    현재는 Placeholder 역할.
    """

    def search(
        self,
        index: _FaissIndex,
        documents: list[dict[str, Any]],
        query_np: np.ndarray,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "HybridRetrievalStrategy는 아직 완전 구현되지 않았습니다. Sparse 지원 처리가 필요합니다."
        )
