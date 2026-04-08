"""
strategy_hybrid.py

Dense(FAISS) + Sparse(BM25) Hybrid Retrieval 전략.

흐름:
  1. Retriever.add_documents() 완료 후 post_add_documents() hook으로 BM25 인덱스 eager 빌드.
  2. Retriever.search(query_text=...) 가 호출되면 set_query_tokens()로 BM25 query token 주입.
  3. search() 에서 FAISS top-dense_top_k + BM25 top-sparse_top_k를 RRF(k=60)로 합산.
  4. 중복 제거 후 상위 top_k 반환. score = RRF 합산값.

BM25 document text:
  "{polarity_tag} {question_noun_tokens} {choice_noun_tokens} {cat_token}"
  Kiwi로 NNG/NNP/SH/SL/SN/XR 추출 + 법률 boilerplate stopwords 제거.
  polarity_tag("부정방향"/"긍정방향"/"중립")를 텍스트 앞에 삽입하여
  쿼리-문서 간 polarity alignment 보강.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from omegaconf import DictConfig

from agent.utils.korean_tokenizer import detect_polarity, tokenize_korean

from ._registry import BaseRetrievalStrategy, _FaissIndex, register_strategy

log = logging.getLogger(__name__)

_RRF_K = 60  # Reciprocal Rank Fusion 표준 상수


def _build_doc_bm25_tokens(doc: dict[str, Any]) -> list[str]:
    """
    단일 학습 문서(doc with content_dict)를 BM25 토큰 리스트로 변환합니다.

    - question에서 polarity 감지 → 첫 토큰으로 삽입
    - question + A/B/C/D 전체를 Kiwi NNG/NNP 필터링으로 토크나이징
    - category 태그 추가
    """
    content = doc.get("content_dict", {})
    question = str(content.get("question", ""))

    polarity_tag = detect_polarity(question)

    combined = " ".join(
        t
        for t in [
            question,
            str(content.get("A", "")),
            str(content.get("B", "")),
            str(content.get("C", "")),
            str(content.get("D", "")),
        ]
        if t
    )
    content_tokens = tokenize_korean(combined)

    category = str(content.get("Category", ""))
    cat_token = "형사법" if "Criminal" in category else "민사법"

    return [polarity_tag, *content_tokens, cat_token]


@register_strategy("hybrid")
class HybridRetrievalStrategy(BaseRetrievalStrategy):
    """Dense(FAISS) + Sparse(BM25) Hybrid 검색 전략.

    설정 파라미터 (configs/retrieval/hybrid.yaml):
        top_k              : 최종 반환 문서 수 (final_k)
        hybrid.dense_top_k : FAISS 후보 수 (기본 10)
        hybrid.sparse_top_k: BM25 후보 수 (기본 10)
        hybrid.fusion      : 현재 "rrf"만 지원
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        hybrid_cfg = config.get("hybrid", {})
        self.dense_top_k: int = int(hybrid_cfg.get("dense_top_k", 10))
        self.sparse_top_k: int = int(hybrid_cfg.get("sparse_top_k", 10))
        self._bm25: Any = None
        self._bm25_doc_list: list[dict[str, Any]] = []
        self._query_tokens: list[str] = []

    # ── BM25 인덱스 ─────────────────────────────────────────────────────────

    def post_add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Retriever.add_documents() 완료 후 BM25 인덱스를 eager 빌드합니다."""
        self._build_bm25_index(documents)

    def _build_bm25_index(self, documents: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        log.info(f"[HybridRetrieval] BM25 인덱스 빌드 시작 (문서 수: {len(documents)})")
        tokenized = [_build_doc_bm25_tokens(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_doc_list = list(documents)
        log.info("[HybridRetrieval] BM25 인덱스 빌드 완료.")

    # ── Query 토큰 주입 ──────────────────────────────────────────────────────

    def set_query_tokens(self, tokens: list[str]) -> None:
        """BM25 검색 query 토큰을 주입합니다. Retriever.search() 내부에서 자동 호출됩니다."""
        self._query_tokens = tokens

    # ── Hybrid Search ────────────────────────────────────────────────────────

    def search(
        self,
        index: _FaissIndex,
        documents: list[dict[str, Any]],
        query_np: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Dense + Sparse Hybrid 검색.

        BM25 미초기화 시 dense-only fallback (엣지 케이스 대비).
        BM25 query 토큰이 없으면 dense-only로 동작.
        """
        if self._bm25 is None:
            log.warning("[HybridRetrieval] BM25 미초기화. Dense-only fallback.")
            self._build_bm25_index(documents)

        # ── Dense retrieval ──────────────────────────────────────────────────
        actual_dense_k = min(self.dense_top_k, max(index.ntotal, 1))
        distances, indices = index.search(query_np, actual_dense_k)
        dense_ranked = [
            int(idx) for idx in indices[0] if idx != -1 and idx < len(documents)
        ]

        # ── Sparse (BM25) retrieval ──────────────────────────────────────────
        sparse_ranked: list[int] = []
        if self._query_tokens and self._bm25 is not None:
            scores: np.ndarray = self._bm25.get_scores(self._query_tokens)
            top_indices = np.argsort(scores)[::-1][: self.sparse_top_k]
            sparse_ranked = [int(i) for i in top_indices if scores[i] > 0.0]

        # ── RRF 합산 ────────────────────────────────────────────────────────
        rrf_scores: dict[int, float] = {}
        for rank, doc_idx in enumerate(dense_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (
                _RRF_K + rank + 1
            )
        for rank, doc_idx in enumerate(sparse_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (
                _RRF_K + rank + 1
            )

        # ── 정렬 후 top_k 반환 ──────────────────────────────────────────────
        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
        final_k = min(self.top_k, len(sorted_indices))

        results: list[dict[str, Any]] = []
        for doc_idx in sorted_indices[:final_k]:
            if doc_idx < len(documents):
                doc_info = documents[doc_idx].copy()
                doc_info["score"] = rrf_scores[doc_idx]
                results.append(doc_info)

        return results
