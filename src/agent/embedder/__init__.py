"""
embedder 패키지

OpenAI text-embedding-3-small 호출 및 Registry 패턴 기반 전처리 전략 관리.

파일 구조 (알파벳 정렬 기준):
  __init__.py                          — 패키지 진입점, 공개 API 노출
  _registry.py                         — registry dict · 데코레이터 · BasePreprocessStrategy
  embedder.py                          — [ENTRY POINT] Embedder 클래스
  strategy_dual_representation.py      — 전략: Dual Representation
  strategy_field_weighted_kv.py        — 전략: Field-Weighted KV
  strategy_kv_pairs.py                 — 전략: KV Pairs
  strategy_narrativized_lite.py        — 전략: Narrativized-lite
  strategy_raw.py                      — 전략: Raw (Baseline)
  strategy_synthetic_query_expansion.py — 전략: Synthetic Query Expansion

외부 import 예시:
    from agent.embedder import Embedder
    from agent.embedder._registry import _PREPROCESS_REGISTRY, BasePreprocessStrategy, register_preprocess
"""
from .embedder import Embedder

# strategy_* 모듈 import → @register_preprocess 데코레이터가 실행되어 _PREPROCESS_REGISTRY에 자동 등록
from . import (  # noqa: F401
    strategy_raw,
    strategy_kv_pairs,
    strategy_narrativized_lite,
    strategy_field_weighted_kv,
    strategy_dual_representation,
    strategy_synthetic_query_expansion,
)

__all__ = ["Embedder"]
