from typing import Any

from ._registry import BasePromptStrategy, PromptResult, register_prompt


@register_prompt("structured_context")
class StructuredContextPromptStrategy(BasePromptStrategy):
    """
    필드 구조화 전달 전략.

    전달 구조:
      - 각 context의 content_dict 필드를 "- 필드명: 값" 형식으로 명시적 구조화합니다.
      - evidence_packet과 달리 retrieval score·rank에 의존하지 않습니다.
      - score는 optional metadata이며, 없어도 정상 동작합니다.
      - labeled_context와의 차이: 원문을 그대로 붙이지 않고 필드를 줄 단위로 분리합니다.

    system prompt: "제공된 필드 정보만 활용" 지시로 강화

    config 파라미터:
      exclude_fields (list[str], 기본 []): 전달에서 제외할 필드 이름 목록
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

        packet_parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            content = ctx.get("content_dict", {})
            lines: list[str] = [f"[참고자료 #{i}]"]
            for key, val in content.items():
                if key not in excluded and val:
                    lines.append(f"- {key}: {val}")
            packet_parts.append("\n".join(lines))

        context_block = "\n\n".join(packet_parts) if packet_parts else "[참고자료 없음]"

        choice_block = (
            f"A) {choices.get('A', '')}\n"
            f"B) {choices.get('B', '')}\n"
            f"C) {choices.get('C', '')}\n"
            f"D) {choices.get('D', '')}"
        )

        user_prompt = (
            "아래 참고자료의 필드 정보를 근거로 문제를 풀어주세요.\n\n"
            "--- 참고자료 ---\n"
            f"{context_block}\n\n"
            "--- 문제 ---\n"
            f"문제: {question}\n"
            f"보기:\n{choice_block}\n\n"
            "정답 알파벳 (A, B, C, D 중 하나만 출력):"
        )

        return PromptResult(system_prompt=self.system_prompt, user_prompt=user_prompt)
