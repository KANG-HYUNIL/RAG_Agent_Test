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
      - structured_context와의 차이: 필드를 재구조화하지 않고 원문을 유지합니다.
      - position_optimized와의 차이: 순서 재배치를 수행하지 않습니다.
                                     배치 순서는 retrieval 레이어가 결정합니다.

    system prompt: 기본 역할 지시 (DEFAULT_SYSTEM_PROMPT)
    """

    def build(
        self,
        question: str,
        choices: dict[str, str],
        contexts: list[dict],
        **kwargs: Any,
    ) -> PromptResult:
        context_parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            content = ctx.get("content_dict", {})
            text = " | ".join(f"{k}: {v}" for k, v in content.items() if v)
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
