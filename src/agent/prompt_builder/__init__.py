"""
prompt_builder 패키지

LLM 프롬프트 생성 및 Registry 패턴 기반 프롬프트 전략 관리.

파일 구조 (알파벳 정렬 기준):
  __init__.py                    — 패키지 진입점, 공개 API 노출
  _registry.py                   — PromptResult · registry dict · 데코레이터 · BasePromptStrategy
  prompt_builder.py              — [ENTRY POINT] PromptBuilder 클래스
  strategy_raw_stuffing.py       — 전략: 원문 그대로 stuffing (baseline)
  strategy_labeled_context.py    — 전략: 번호 라벨 부착, 원문 전달
  strategy_structured_context.py — 전략: key-value 필드 구조화 전달
  strategy_few_shot_envelope.py  — 전략: retrieved 예시를 Q&A 데모로 감싸 전달

설계 원칙:
  1. 출력 형식은 항상 "A/B/C/D 한 글자"로 고정. 전략이 이를 바꾸면 안 됨.
  2. context 배치 순서는 retrieval 레이어가 결정. 전략이 재정렬하면 안 됨.
  3. score·rank는 optional metadata. 없어도 전략이 정상 동작해야 함.
  4. 각 전략의 차이는 "context를 LLM에 어떤 구조로 전달하는가"에만 있음.

외부 import 예시:
    from agent.prompt_builder import PromptBuilder, PromptResult
    from agent.prompt_builder._registry import _PROMPT_REGISTRY, BasePromptStrategy, register_prompt
"""

# strategy_* 모듈 import → @register_prompt 데코레이터가 실행되어 _PROMPT_REGISTRY에 자동 등록
from . import (
    strategy_few_shot_envelope,
    strategy_labeled_context,
    strategy_raw_stuffing,
    strategy_structured_context,
)
from ._registry import PromptResult
from .prompt_builder import PromptBuilder

__all__ = [
    "PromptBuilder",
    "PromptResult",
    "strategy_few_shot_envelope",
    "strategy_labeled_context",
    "strategy_raw_stuffing",
    "strategy_structured_context",
]
