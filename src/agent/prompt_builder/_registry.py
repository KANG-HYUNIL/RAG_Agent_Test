from dataclasses import dataclass
from typing import Any

# ==========================================
# 공통 반환 타입
# ==========================================


@dataclass
class PromptResult:
    """
    프롬프트 전략이 반환하는 (system_prompt, user_prompt) 쌍.

    - system_prompt: 역할·출력 형식·전달 방식에 관한 지시 (전략별로 달라질 수 있음)
    - user_prompt:   query + context 블록 (전략의 핵심 차이가 여기서 드러남)
    """

    system_prompt: str
    user_prompt: str


# ==========================================
# 기본 system prompt (전략이 override하지 않으면 이 값 사용)
# ==========================================

_DEFAULT_SYSTEM_PROMPT = """당신은 대한민국 법률 객관식 문제를 푸는 정답 분류기입니다.
    제공된 참고자료를 최우선 근거로 사용하세요.
    참고자료에 없는 외부 지식이나 추측을 근거로 답을 바꾸지 마세요.
    문제와 참고자료를 비교하여 가장 적절한 선택지 하나를 고르세요.
    반드시 A, B, C, D 중 하나의 알파벳 한 글자만 출력하세요.
    다른 설명은 출력하지 마세요."""


# ==========================================
# 프롬프트 전략 레지스트리
# ==========================================

_PROMPT_REGISTRY: dict[str, type["BasePromptStrategy"]] = {}


def register_prompt(name: str):
    """지정된 이름으로 프롬프트 전략 클래스를 레지스트리에 등록하는 데코레이터."""

    def decorator(cls: type["BasePromptStrategy"]) -> type["BasePromptStrategy"]:
        _PROMPT_REGISTRY[name] = cls
        return cls

    return decorator


# ==========================================
# 베이스 클래스
# ==========================================


class BasePromptStrategy:
    """
    프롬프트 전략 베이스 클래스.

    공통 입력 인터페이스:
      question  (str)            — 문제 본문
      choices   (dict[str, str]) — {"A": ..., "B": ..., "C": ..., "D": ...}
      contexts  (list[dict])     — retrieval 결과.
                                   각 항목: {"content_dict": {...}, ...}
                                   score·rank 등은 optional metadata이며,
                                   없어도 정상 동작해야 합니다.
      **kwargs                   — 전략별 config 파라미터

    반환: PromptResult(system_prompt, user_prompt)
      - system_prompt: 역할·출력 형식 지시. 전략별 class 속성으로 override 가능.
      - user_prompt:   context 전달 구조의 차이가 집중되는 곳.

    설계 원칙:
      1. 출력 형식은 항상 "A/B/C/D 한 글자"로 고정 — 전략이 이를 바꿔서는 안 됨.
      2. contexts의 순서는 retrieval 레이어가 결정 — 전략이 score 기반으로 재정렬하면 안 됨.
      3. score·rank 등 retrieval metadata는 선택적으로만 사용 (없어도 동작).
    """

    # 전략 클래스 속성으로 override 가능
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT

    def build(
        self,
        question: str,
        choices: dict[str, str],
        contexts: list[dict],
        **kwargs: Any,
    ) -> PromptResult:
        raise NotImplementedError("서브타입에서 build를 구현해야 합니다.")
