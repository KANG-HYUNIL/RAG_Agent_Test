from typing import Any

from ._registry import register_prompt, BasePromptStrategy


@register_prompt("mcq")
class MCQPromptStrategy(BasePromptStrategy):
    """
    객관식(A/B/C/D) 문제 응답용 프롬프트 생성 전략.
    검색된 컨텍스트와 문제·보기를 조합하여 LLM에 전달할 최종 프롬프트를 구성합니다.
    """

    def build(self, question: str, choices: dict, contexts: list, **kwargs: Any) -> str:
        # 컨텍스트 문자열화
        context_str_parts = []
        for i, ctx in enumerate(contexts):
            content = ctx.get("content_dict", {})
            text = " | ".join(f"{k}: {v}" for k, v in content.items() if v)
            context_str_parts.append(f"[참고자료 {i+1}]\n{text}")

        context_block = "\n\n".join(context_str_parts)

        # 보기 문자열화
        choice_block = (
            f"A) {choices.get('A', '')}\n"
            f"B) {choices.get('B', '')}\n"
            f"C) {choices.get('C', '')}\n"
            f"D) {choices.get('D', '')}"
        )

        return (
            "다음 참고자료를 읽고, 이어지는 문제의 가장 적절한 정답 알파벳을 고르세요.\n\n"
            "--- 참고자료 ---\n"
            f"{context_block if context_block else '참고할 자료가 없습니다.'}\n\n"
            "--- 문제 ---\n"
            f"문제: {question}\n"
            f"보기:\n{choice_block}\n\n"
            "정답 알파벳 (A, B, C, D 중 하나만 출력):"
        )
