from typing import Any

from ._registry import BasePromptStrategy, PromptResult, register_prompt


@register_prompt("raw_stuffing")
class RawStuffingPromptStrategy(BasePromptStrategy):
    """
    원문 그대로 stuffing 전략 (baseline).

    전달 구조:
      - retrieved contexts를 레이블 없이 순서대로 이어 붙입니다.
      - 구조·번호·요약 없이 원문 텍스트만 전달합니다.
      - context 배치 순서는 retrieval 레이어가 결정한 순서를 그대로 따릅니다.

    system prompt: 기본 역할 지시 (DEFAULT_SYSTEM_PROMPT)

    config 파라미터:
      exclude_fields (list[str], 기본 []): context 전달에서 제외할 필드 이름 목록
    """

    def build(
        self,
        question: str,
        choices: dict[str, str],
        contexts: list[dict],
        exclude_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> PromptResult:
        excluded: set[str] = set(exclude_fields or [])
        raw_parts: list[str] = []
        for ctx in contexts:
            content = ctx.get("content_dict", {})
            raw_parts.append(
                " | ".join(
                    f"{k}: {v}" for k, v in content.items() if v and k not in excluded
                )
            )

        context_block = (
            "\n\n".join(raw_parts) if raw_parts else "참고할 자료가 없습니다."
        )

        choice_block = (
            f"A) {choices.get('A', '')}\n"
            f"B) {choices.get('B', '')}\n"
            f"C) {choices.get('C', '')}\n"
            f"D) {choices.get('D', '')}"
        )

        user_prompt = (
            "다음 참고자료를 읽고, 이어지는 문제의 가장 적절한 정답 알파벳을 고르세요.\n\n"
            "--- 참고자료 ---\n"
            f"{context_block}\n\n"
            "--- 문제 ---\n"
            f"문제: {question}\n"
            f"보기:\n{choice_block}\n\n"
            "정답 알파벳 (A, B, C, D 중 하나만 출력):"
        )

        return PromptResult(system_prompt=self.system_prompt, user_prompt=user_prompt)
