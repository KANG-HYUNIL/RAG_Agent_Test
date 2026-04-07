from typing import Any

from ._registry import BasePromptStrategy, PromptResult, register_prompt


@register_prompt("labeled_context")
class LabeledContextPromptStrategy(BasePromptStrategy):
    """
    번호 라벨 부착 전달 전략.

    전달 구조:
      - retrieved contexts에 [참고자료 #N] 번호 라벨을 붙여 전달합니다.
      - 내용은 원문 그대로이며, 라벨만 추가됩니다.
      - raw_stuffing과의 차이: 각 항목의 경계가 명확하게 구분됩니다.

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
        context_parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            content = ctx.get("content_dict", {})
            text = " | ".join(
                f"{k}: {v}" for k, v in content.items() if v and k not in excluded
            )
            context_parts.append(f"[참고자료 #{i}]\n{text}")

        context_block = (
            "\n\n".join(context_parts) if context_parts else "참고할 자료가 없습니다."
        )

        choice_block = (
            f"A) {choices.get('A', '')}\n"
            f"B) {choices.get('B', '')}\n"
            f"C) {choices.get('C', '')}\n"
            f"D) {choices.get('D', '')}"
        )

        user_prompt = (
            "다음 번호가 매겨진 참고자료를 읽고, 이어지는 문제의 가장 적절한 정답 알파벳을 고르세요.\n\n"
            "--- 참고자료 ---\n"
            f"{context_block}\n\n"
            "--- 문제 ---\n"
            f"문제: {question}\n"
            f"보기:\n{choice_block}\n\n"
            "정답 알파벳 (A, B, C, D 중 하나만 출력):"
        )

        return PromptResult(system_prompt=self.system_prompt, user_prompt=user_prompt)
