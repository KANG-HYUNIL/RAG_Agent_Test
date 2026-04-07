from typing import Any

from ._registry import BasePromptStrategy, PromptResult, register_prompt


@register_prompt("compress_summarize")
class CompressSummarizePromptStrategy(BasePromptStrategy):
    """
    압축 전달 전략.

    전달 구조:
      - 각 context를 max_chars_per_context 글자 수로 절삭한 뒤 전달합니다.
      - 번호 라벨을 붙여 절삭된 각 항목을 구분합니다.
      - raw_stuffing과의 차이: 전체 원문 대신 길이 제한된 압축본을 전달합니다.

    system prompt: 기본 역할 지시 (DEFAULT_SYSTEM_PROMPT)

    config 파라미터:
      max_chars_per_context (int, 기본 200): context당 최대 글자 수
    """

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "…"

    def build(
        self,
        question: str,
        choices: dict[str, str],
        contexts: list[dict],
        max_chars_per_context: int = 200,
        **kwargs: Any,
    ) -> PromptResult:
        compressed_parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            content = ctx.get("content_dict", {})
            raw_text = " | ".join(f"{k}: {v}" for k, v in content.items() if v)
            compressed_parts.append(f"[압축 자료 #{i}]\n{self._truncate(raw_text, max_chars_per_context)}")

        context_block = "\n\n".join(compressed_parts) if compressed_parts else "참고할 자료가 없습니다."

        choice_block = (
            f"A) {choices.get('A', '')}\n"
            f"B) {choices.get('B', '')}\n"
            f"C) {choices.get('C', '')}\n"
            f"D) {choices.get('D', '')}"
        )

        user_prompt = (
            "아래 압축된 참고자료를 바탕으로 문제를 풀어주세요.\n\n"
            "--- 압축 참고자료 ---\n"
            f"{context_block}\n\n"
            "--- 문제 ---\n"
            f"문제: {question}\n"
            f"보기:\n{choice_block}\n\n"
            "정답 알파벳 (A, B, C, D 중 하나만 출력):"
        )

        return PromptResult(system_prompt=self.system_prompt, user_prompt=user_prompt)
