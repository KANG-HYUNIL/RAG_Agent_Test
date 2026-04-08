"""
query_encoder.py

dev query 텍스트 구성 방법론.

corpus-side serialization(embedder)과 분리된 query-side 표현 레이어입니다.

지원 method:
  - question_only          : 질문 텍스트만 사용 (baseline)
  - question_plus_choices  : 질문 + 선택지 A~D를 결합 (4차 strongest baseline)
  - polarity_aware_qpc     : question_plus_choices + polarity 방향 태그 prefix
  - core_focus_query       : 질문의 핵심 법률 명사 + polarity 태그 (선택지 없음, Kiwi 사용)
"""

from __future__ import annotations

_VALID_METHODS = frozenset(
    {
        "question_only",
        "question_plus_choices",
        "polarity_aware_qpc",
        "core_focus_query",
    }
)


def build_query_text(
    method: str,
    question: str,
    choices: dict[str, str],
) -> str:
    """
    query 임베딩에 사용할 텍스트를 method에 따라 구성합니다.

    Args:
        method:   지원 method 중 하나 (위 _VALID_METHODS 참조)
        question: dev row의 질문 본문
        choices:  {"A": ..., "B": ..., "C": ..., "D": ...}

    Returns:
        임베딩할 query 텍스트 문자열

    Raises:
        ValueError: 지원하지 않는 method인 경우
    """
    if method == "question_only":
        return question

    if method == "question_plus_choices":
        choice_str = " ".join(f"{k}) {v}" for k, v in choices.items())
        return f"{question} {choice_str}"

    if method == "polarity_aware_qpc":
        return _build_polarity_aware_qpc(question, choices)

    if method == "core_focus_query":
        return _build_core_focus_query(question)

    raise ValueError(
        f"Unknown query_representation method: '{method}'. "
        f"Available: {sorted(_VALID_METHODS)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# polarity_aware_qpc
# ─────────────────────────────────────────────────────────────────────────────


def _build_polarity_aware_qpc(question: str, choices: dict[str, str]) -> str:
    """
    question_plus_choices에 polarity 태그 prefix를 추가합니다.

    구성: "[{polarity_tag}] {question} {choice_str}"

    polarity_tag:
      - "부정방향" : 옳지 않은, 아닌 것, 틀린 등
      - "긍정방향" : 옳은, 해당하는 등
      - "중립"    : 판정 불가 / OX 연결형

    regex 기반 감지 (LLM/embedding 추가 호출 없음).
    """
    from agent.utils.korean_tokenizer import detect_polarity

    polarity_tag = detect_polarity(question)
    choice_str = " ".join(f"{k}) {v}" for k, v in choices.items())
    return f"[{polarity_tag}] {question} {choice_str}"


# ─────────────────────────────────────────────────────────────────────────────
# core_focus_query
# ─────────────────────────────────────────────────────────────────────────────


def _build_core_focus_query(question: str) -> str:
    """
    질문에서 핵심 법률 명사 토큰만 추출하여 compact query를 구성합니다.

    구성: "[{polarity_tag}] {core_noun_tokens}"
    - 선택지 없음 (선택지 표면어 noise 제거가 목표)
    - Kiwi 형태소 분석으로 NNG/NNP/SH/SL/SN 추출
    - 법률 boilerplate stopwords 제거 ("설명", "내용", "경우" 등)
    - polarity 태그 prefix

    예시:
      "피고인 출석의 예외에 대한 설명으로 옳지 않은 것은?"
      → "[부정방향] 피고인 출석 예외"

      "불법행위로 인한 손해배상청구권의 소멸시효에 관한 설명으로 옳은 것은?"
      → "[긍정방향] 불법행위 손해배상청구권 소멸시효"
    """
    from agent.utils.korean_tokenizer import detect_polarity, extract_core_tokens

    polarity_tag = detect_polarity(question)
    core_tokens = extract_core_tokens(question)

    if core_tokens:
        return f"[{polarity_tag}] {' '.join(core_tokens)}"

    # fallback: Kiwi 실패 시 question 원문 그대로
    return f"[{polarity_tag}] {question}"
