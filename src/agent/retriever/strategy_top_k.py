from typing import Any

import numpy as np

from ._registry import BaseRetrievalStrategy, _FaissIndex, register_strategy


@register_strategy("top_k")
class TopKRetrievalStrategy(BaseRetrievalStrategy):
    """1차 Baseline: 가장 기본적인 FAISS Top-K 검색"""

    def search(
        self,
        index: _FaissIndex,
        documents: list[dict[str, Any]],
        query_np: np.ndarray,
    ) -> list[dict[str, Any]]:
        # faiss.search 반환값: (거리/유사도 배열, 해당하는 문서의 인덱스 번호 배열)
        distances, indices = index.search(query_np, self.top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx != -1 and idx < len(documents):
                doc_info = documents[idx].copy()
                doc_info["score"] = float(dist)
                results.append(doc_info)

        return results
