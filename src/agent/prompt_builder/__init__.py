"""
prompt_builder 패키지

LLM 프롬프트 생성 및 Registry 패턴 기반 프롬프트 전략 관리.

파일 구조 (알파벳 정렬 기준):
  __init__.py        — 패키지 진입점, 공개 API 노출
  _registry.py       — registry dict · 데코레이터 · BasePromptStrategy
  prompt_builder.py  — [ENTRY POINT] PromptBuilder 클래스
  strategy_mcq.py    — 전략: MCQ (객관식 A/B/C/D 프롬프트)

외부 import 예시:
    from agent.prompt_builder import PromptBuilder
    from agent.prompt_builder._registry import _PROMPT_REGISTRY, BasePromptStrategy, register_prompt
"""
from .prompt_builder import PromptBuilder

# strategy_* 모듈 import → @register_prompt 데코레이터가 실행되어 _PROMPT_REGISTRY에 자동 등록
from . import strategy_mcq  # noqa: F401

__all__ = ["PromptBuilder"]
