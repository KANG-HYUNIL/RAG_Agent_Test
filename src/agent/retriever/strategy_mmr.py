import logging
from typing import Any

import numpy as np
from omegaconf import DictConfig

from ._registry import BaseRetrievalStrategy, _FaissIndex, register_strategy

log = logging.getLogger(__name__)


@register_strategy("mmr")
class MMRRetrievalStrategy(BaseRetrievalStrategy):
    """3차 개선: Maximal Marginal Relevance 기반 분산 고려 검색"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.fetch_k = config.mmr.get("fetch_k", 20)
        self.lambda_mult = config.mmr.get("lambda_mult", 0.7)

    def _cosine_similarity(
        self, vec1: np.ndarray, doc_vectors: np.ndarray
    ) -> np.ndarray:
        # np.dot 기반 (query vs fetch_k vectors)
        return np.dot(doc_vectors, vec1)

    def search(
        self,
        index: _FaissIndex,
        documents: list[dict[str, Any]],
        query_np: np.ndarray,
    ) -> list[dict[str, Any]]:
        # 1. 탐색 후보군(fetch_k) 추출
        actual_fetch = min(self.fetch_k, index.ntotal)
        if actual_fetch == 0:
            return []

        distances, indices = index.search(query_np, actual_fetch)

        # 2. 결과 인덱스와 거리를 추출
        candidate_indices = [
            idx for idx in indices[0] if idx != -1 and idx < len(documents)
        ]
        if not candidate_indices:
            return []

        # 3. FAISS에서 문서 벡터 재추출 (IndexFlatIP 등 원본을 들고 있는 인덱스 유형만 reconstruct 지원됨)
        try:
            candidate_vectors = np.array(
                [index.reconstruct(int(i)) for i in candidate_indices],
                dtype=np.float32,
            )
        except Exception as e:
            log.warning(f"인덱스에서 벡터 재추출에 실패했습니다: {e}")
            return []

        # 4. MMR 코어 로직
        query_vec = query_np[0]
        query_sims = self._cosine_similarity(
            query_vec, candidate_vectors
        )  # shape: (len(cands),)

        selected_indices = []
        selected_sims = []

        # 첫 번째 문서 선택 (Query와 가장 유사도가 높은 것)
        best_idx = int(np.argmax(query_sims))
        selected_indices.append(candidate_indices[best_idx])
        selected_sims.append(query_sims[best_idx])

        unselected_idx_map = {
            i: candidate_indices[i]
            for i in range(len(candidate_indices))
            if i != best_idx
        }

        while len(selected_indices) < min(self.top_k, len(candidate_indices)):
            best_score = -np.inf
            idx_to_add = -1

            # 현재까지 선택된 문서 벡터들 추출
            selected_docs_vecs = np.array(
                [index.reconstruct(int(i)) for i in selected_indices],
                dtype=np.float32,
            )

            for i, _cand_idx in unselected_idx_map.items():
                cand_vec = candidate_vectors[i]

                # MMR 계산 공식
                sim_with_query = query_sims[i]
                sim_with_selected = self._cosine_similarity(
                    cand_vec, selected_docs_vecs
                )
                max_sim_with_selected = np.max(sim_with_selected)

                mmr_score = (
                    self.lambda_mult * sim_with_query
                    - (1 - self.lambda_mult) * max_sim_with_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    idx_to_add = i

            if idx_to_add != -1:
                selected_indices.append(candidate_indices[idx_to_add])
                selected_sims.append(query_sims[idx_to_add])
                del unselected_idx_map[idx_to_add]
            else:
                break

        # 5. 결과 반환 배열 구성
        results = []
        for idx_origin, sim in zip(selected_indices, selected_sims, strict=False):
            doc_info = documents[idx_origin].copy()
            doc_info["score"] = float(sim)
            results.append(doc_info)

        return results
