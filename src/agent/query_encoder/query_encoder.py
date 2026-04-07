"""
query_encoder.py

dev query 텍스트 구성 방법론.

corpus-side serialization(embedder)과 분리된 query-side 표현 레이어입니다.
  - question_only    : 질문 텍스트만 사용 (baseline)
  - question_plus_choices : 질문 + 선택지 A~D를 결합하여 사용
"""

_VALID_METHODS = frozenset({"question_only", "question_plus_choices"})


def build_query_text(
    method: str,
    question: str,
    choices: dict[str, str],
) -> str:
    """
    query 임베딩에 사용할 텍스트를 method에 따라 구성합니다.

    Args:
        method:   "question_only" | "question_plus_choices"
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

    raise ValueError(
        f"Unknown query_representation method: '{method}'. "
        f"Available: {sorted(_VALID_METHODS)}"
    )
