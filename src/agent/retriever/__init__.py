"""
retriever 패키지

FAISS 기반 유사도 검색 및 Registry 패턴 기반 전략 관리.

파일 구조 (알파벳 정렬 기준):
  __init__.py                  — 패키지 진입점, 공개 API 노출
  _registry.py                 — registry dict · 데코레이터 · BaseRetrievalStrategy
  retriever.py                 — [ENTRY POINT] Retriever 클래스
  strategy_hybrid.py           — 전략: Hybrid (Dense + Sparse, Placeholder)
  strategy_mmr.py              — 전략: Maximal Marginal Relevance
  strategy_score_threshold.py  — 전략: Score Threshold
  strategy_top_k.py            — 전략: Top-K (Baseline)

외부 import 예시:
    from agent.retriever import Retriever
    from agent.retriever._registry import RETRIEVAL_STRATEGIES, BaseRetrievalStrategy, register_strategy
"""

# strategy_* 모듈 import → @register_strategy 데코레이터가 실행되어 RETRIEVAL_STRATEGIES에 자동 등록
from . import (
    strategy_hybrid,
    strategy_mmr,
    strategy_score_threshold,
    strategy_top_k,
)
from .retriever import Retriever

__all__ = [
    "Retriever",
    "strategy_hybrid",
    "strategy_mmr",
    "strategy_score_threshold",
    "strategy_top_k",
]
