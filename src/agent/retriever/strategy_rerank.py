"""
strategy_rerank.py

Dense retrieval 후단 보정(Post-Retrieval Reranking) 전략 — 6차 실험용.

흐름:
  1. FAISS로 retrieve_k개 후보 탐색 후 score_threshold 필터 적용
  2. 후보에 대해 조정 스코어 계산:
       adjusted = faiss_score
                + bm25_aux_weight  × norm_bm25_score  (선택)
                - polarity_penalty  (쿼리-문서 polarity mismatch 시) (선택)
                - template_penalty  (OX/연결형 템플릿 + 핵심어 overlap 낮음 시) (선택)
  3. adjusted 내림차순 정렬 후 final_k 반환

설계 원칙:
  - LLM 추가 호출 없음 / 추가 embedding 호출 없음 / 외부 reranker 없음
  - index-time(post_add_documents)에서 BM25 / polarity / template 정보 1회 캐시
  - query-time 비용: regex O(1) + BM25 get_scores O(vocab) + set 교집합 O(retrieve_k)
  - Retriever.set_query_text() hook 활용 → 원문 polarity 감지 정확성 보장

설정 파라미터 (configs/retrieval/rerank_*.yaml):
  method:               rerank
  top_k:                <final_k>       # benchmark.py 호환용 (= final_k)
  metric:               "IP"
  normalize_query:      true
  score_threshold:
    value:              0.35
  rerank:
    retrieve_k:         10              # FAISS 후보 수
    final_k:            5               # 최종 반환 수
    polarity_penalty:   0.0             # 0.0 → OFF
    bm25_aux_weight:    0.0             # 0.0 → OFF (BM25 인덱스 빌드 생략)
    template_penalty:   0.0             # 0.0 → OFF
    template_overlap_threshold: 0.3    # 자카드 < 이 값이면 template_penalty 적용
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from omegaconf import DictConfig

from agent.utils.korean_tokenizer import detect_polarity, extract_core_tokens, tokenize_korean

from ._registry import BaseRetrievalStrategy, _FaissIndex, register_strategy

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 템플릿 감지 (OX연결형 / 바르게연결형)
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_OX = re.compile(r"옳은.{0,15}옳지\s*않은|○.{0,10}×")
_TEMPLATE_LINK = re.compile(r"바르게\s*연결한|올바르게\s*연결")


def _detect_template(question: str) -> str | None:
    """OX연결형이면 'OX', 연결형이면 'LINK', 해당 없으면 None."""
    if _TEMPLATE_OX.search(question):
        return "OX"
    if _TEMPLATE_LINK.search(question):
        return "LINK"
    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    """자카드 유사도. 둘 다 공집합이면 0.0 반환."""
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BM25 문서 토큰 빌더 (hybrid 전략과 동일 방식)
# ─────────────────────────────────────────────────────────────────────────────


def _build_doc_bm25_tokens(doc: dict[str, Any]) -> list[str]:
    """단일 학습 문서를 BM25 토큰 리스트로 변환 (polarity tag + content tokens + cat tag)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# PostRetrievalRerankStrategy
# ─────────────────────────────────────────────────────────────────────────────


@register_strategy("rerank")
class PostRetrievalRerankStrategy(BaseRetrievalStrategy):
    """Dense retrieval 후단 보정 전략 (6차 실험).

    retrieve_k 후보를 dense로 뽑은 뒤 polarity / BM25 auxiliary / template penalty를
    조합하여 adjusted score로 재정렬 후 final_k를 반환합니다.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)  # self.top_k = config.top_k (benchmark.py 호환)

        # score_threshold
        st_cfg = config.get("score_threshold", {})
        self.threshold: float = float(st_cfg.get("value", 0.35))
        self.metric: str = str(config.get("metric", "IP"))

        # rerank 파라미터
        rk_cfg = config.get("rerank", {})
        self.retrieve_k: int = int(rk_cfg.get("retrieve_k", 10))
        self.final_k: int = int(rk_cfg.get("final_k", self.top_k))
        self.polarity_penalty: float = float(rk_cfg.get("polarity_penalty", 0.0))
        self.bm25_aux_weight: float = float(rk_cfg.get("bm25_aux_weight", 0.0))
        self.template_penalty: float = float(rk_cfg.get("template_penalty", 0.0))
        self.template_overlap_threshold: float = float(
            rk_cfg.get("template_overlap_threshold", 0.3)
        )

        # 런타임 캐시 (post_add_documents에서 채워짐)
        self._doc_polarities: list[str] = []
        self._doc_templates: list[str | None] = []
        self._doc_core_token_sets: list[set[str]] = []

        # BM25 (bm25_aux_weight > 0일 때만 초기화)
        self._bm25: Any = None
        self._query_tokens: list[str] = []

        # query 원문 기반 메타 (set_query_text에서 채워짐)
        self._query_polarity: str = "중립"
        self._query_template: str | None = None
        self._query_core_tokens: set[str] = set()

        log.info(
            f"[RerankStrategy] retrieve_k={self.retrieve_k}, final_k={self.final_k}, "
            f"polarity_penalty={self.polarity_penalty}, "
            f"bm25_aux_weight={self.bm25_aux_weight}, "
            f"template_penalty={self.template_penalty}"
        )

    # ── Index-time hook ──────────────────────────────────────────────────────

    def post_add_documents(self, documents: list[dict[str, Any]]) -> None:
        """add_documents() 완료 후 1회 호출. 각 문서의 메타 정보를 캐시합니다."""
        log.info(f"[RerankStrategy] 문서 캐시 빌드 시작 (총 {len(documents)}개)...")

        self._doc_polarities = []
        self._doc_templates = []
        self._doc_core_token_sets = []

        for doc in documents:
            content = doc.get("content_dict", {})
            question = str(content.get("question", ""))

            # polarity 캐시
            self._doc_polarities.append(detect_polarity(question))

            # template 캐시
            self._doc_templates.append(_detect_template(question))

            # core token 캐시 (template_penalty > 0 일 때만 실제 추출)
            if self.template_penalty > 0.0:
                self._doc_core_token_sets.append(set(extract_core_tokens(question)))
            else:
                self._doc_core_token_sets.append(set())

        # BM25 인덱스 (bm25_aux_weight > 0 일 때만)
        if self.bm25_aux_weight > 0.0:
            self._build_bm25_index(documents)

        log.info("[RerankStrategy] 문서 캐시 빌드 완료.")

    def _build_bm25_index(self, documents: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        log.info(f"[RerankStrategy] BM25 인덱스 빌드 중 (문서 수: {len(documents)})...")
        tokenized = [_build_doc_bm25_tokens(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        log.info("[RerankStrategy] BM25 인덱스 빌드 완료.")

    # ── Query-time hooks (Retriever.search()에서 자동 호출) ──────────────────

    def set_query_tokens(self, tokens: list[str]) -> None:
        """BM25 auxiliary scoring 용 query 토큰 주입."""
        self._query_tokens = tokens

    def set_query_text(self, text: str) -> None:
        """원본 query 텍스트 주입 — polarity / template 감지에 사용."""
        self._query_polarity = detect_polarity(text)
        self._query_template = _detect_template(text)
        # template_penalty > 0 일 때만 core tokens 추출 (비용 절감)
        if self.template_penalty > 0.0:
            self._query_core_tokens = set(extract_core_tokens(text))
        else:
            self._query_core_tokens = set()

    # ── 핵심: Post-Retrieval Reranking ──────────────────────────────────────

    def search(
        self,
        index: _FaissIndex,
        documents: list[dict[str, Any]],
        query_np: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Dense retrieve_k 후보 → 후단 보정 → final_k 반환."""

        # ── Step 1: FAISS dense retrieve_k 조회 ─────────────────────────────
        actual_k = min(self.retrieve_k, max(index.ntotal, 1))
        distances, indices = index.search(query_np, actual_k)

        candidates: list[tuple[int, float]] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx == -1 or idx >= len(documents):
                continue
            score = float(dist)
            if self.metric == "IP" and score < self.threshold:
                continue
            candidates.append((int(idx), score))

        if not candidates:
            return []

        # ── Step 2: BM25 auxiliary scores (후보 subset에 대해서만 활용) ──────
        bm25_normed: dict[int, float] = {}
        if self.bm25_aux_weight > 0.0 and self._bm25 is not None and self._query_tokens:
            all_scores: np.ndarray = self._bm25.get_scores(self._query_tokens)
            cand_indices = [idx for idx, _ in candidates]
            raw = np.array(
                [max(0.0, float(all_scores[i])) for i in cand_indices], dtype=np.float32
            )
            rng = float(raw.max() - raw.min())
            normed = (raw - raw.min()) / rng if rng > 1e-9 else np.zeros_like(raw)
            for i, doc_idx in enumerate(cand_indices):
                bm25_normed[doc_idx] = float(normed[i])

        # ── Step 3: Adjusted score 계산 ─────────────────────────────────────
        adjusted: list[tuple[int, float]] = []

        for doc_idx, faiss_score in candidates:
            adj = faiss_score

            # BM25 auxiliary 가산
            if self.bm25_aux_weight > 0.0:
                adj += self.bm25_aux_weight * bm25_normed.get(doc_idx, 0.0)

            # Polarity penalty
            if self.polarity_penalty > 0.0:
                doc_pol = (
                    self._doc_polarities[doc_idx]
                    if doc_idx < len(self._doc_polarities)
                    else "중립"
                )
                # 양쪽 모두 중립이 아닌데 방향이 다르면 감점
                if (
                    self._query_polarity != "중립"
                    and doc_pol != "중립"
                    and self._query_polarity != doc_pol
                ):
                    adj -= self.polarity_penalty

            # Template penalty
            if self.template_penalty > 0.0 and doc_idx < len(self._doc_templates):
                doc_tmpl = self._doc_templates[doc_idx]
                if doc_tmpl is not None:
                    # 후보가 OX/LINK 템플릿인 경우 핵심어 overlap 검사
                    doc_core = self._doc_core_token_sets[doc_idx]
                    overlap = _jaccard(self._query_core_tokens, doc_core)
                    if overlap < self.template_overlap_threshold:
                        adj -= self.template_penalty

            adjusted.append((doc_idx, adj))

        # ── Step 4: 재정렬 후 final_k 반환 ──────────────────────────────────
        adjusted.sort(key=lambda x: x[1], reverse=True)
        effective_final_k = min(self.final_k, len(adjusted))

        results: list[dict[str, Any]] = []
        for doc_idx, adj_score in adjusted[:effective_final_k]:
            doc_info = documents[doc_idx].copy()
            doc_info["score"] = adj_score
            results.append(doc_info)

        return results
