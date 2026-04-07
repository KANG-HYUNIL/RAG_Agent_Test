from typing import Any

from ._registry import _PROMPT_REGISTRY, PromptResult


class PromptBuilder:
    """
    검색된 데이터를 바탕으로 LLM(gpt-4o-mini)에 전달할 최종 프롬프트를 구성합니다.
    Registry Pattern을 통해 다양한 프롬프트 전략을 동적으로 확장합니다.

    등록된 전략 목록:
      raw_stuffing       — 원문 그대로 stuffing (baseline)
      labeled_context    — 번호 라벨 부착, 원문 전달
      structured_context — key-value 필드 구조화 전달
      few_shot_envelope  — retrieved 예시를 Q&A 데모로 감싸 전달
    """

    def build_prompt(
        self,
        method: str,
        question: str,
        choices: dict[str, str],
        contexts: list[dict],
        **config_kwargs: Any,
    ) -> PromptResult:
        """
        Registry에 등록된 전략 이름(method)으로 프롬프트를 생성합니다.

        Args:
            method:        레지스트리 키 (예: "raw_stuffing", "few_shot_envelope")
            question:      문제 본문
            choices:       {"A": ..., "B": ..., "C": ..., "D": ...}
            contexts:      retriever가 반환한 참고 자료 리스트
                           각 항목: {"content_dict": {...}, ...}
                           score·rank는 optional metadata
            **config_kwargs: hydra config에서 내려온 전략별 파라미터

        Returns:
            PromptResult(system_prompt, user_prompt)

        Raises:
            KeyError: 등록되지 않은 method 이름이 주어진 경우
        """
        if method not in _PROMPT_REGISTRY:
            available = list(_PROMPT_REGISTRY.keys())
            msg = f"프롬프트 전략 '{method}'가 등록되지 않았습니다. 사용 가능: {available}"
            raise KeyError(msg)

        strategy = _PROMPT_REGISTRY[method]()
        return strategy.build(
            question=question,
            choices=choices,
            contexts=contexts,
            **config_kwargs,
        )
