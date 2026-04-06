import faiss
import numpy as np
from typing import List, Dict, Any
from omegaconf import DictConfig

from ._registry import register_strategy, BaseRetrievalStrategy


@register_strategy("score_threshold")
class ScoreThresholdRetrievalStrategy(BaseRetrievalStrategy):
    """2차 개선: 검색된 문서들 중 score가 특정 임계값을 넘는 문서들만 반환"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.threshold = config.score_threshold.get("value", 0.0)
        self.metric = config.get("metric", "IP")  # "IP" or "L2"

    def search(
        self,
        index: faiss.Index,
        documents: List[Dict[str, Any]],
        query_np: np.ndarray,
    ) -> List[Dict[str, Any]]:
        # 먼저 top_k 탐색
        distances, indices = index.search(query_np, self.top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(documents):
                score = float(dist)

                # IP(내적/코사인유사도)는 클수록 좋음, L2는 작을수록 좋음
                if self.metric == "IP" and score >= self.threshold:
                    doc_info = documents[idx].copy()
                    doc_info["score"] = score
                    results.append(doc_info)
                elif self.metric == "L2" and score <= self.threshold:
                    doc_info = documents[idx].copy()
                    doc_info["score"] = score
                    results.append(doc_info)

        return results
