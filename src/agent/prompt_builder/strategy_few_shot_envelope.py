from typing import Any

from ._registry import BasePromptStrategy, PromptResult, register_prompt

_ANSWER_MAP: dict[Any, str] = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
}


@register_prompt("few_shot_envelope")
class FewShotEnvelopePromptStrategy(BasePromptStrategy):
    """
    Few-shot envelope 전략.

    전달 구조:
      - retrieved contexts를 "질문-보기-정답" 형식의 Q&A 데모로 포맷합니다.
      - 데모를 통해 A/B/C/D 단독 출력 양식을 LLM에 시범 제시합니다.
      - raw_stuffing과의 차이: context를 참고자료가 아닌 입출력 예시로 처리합니다.

    system prompt: few-shot 데모 형식 준수 지시로 강화

    config 파라미터:
      num_examples (int, 기본 3): 사용할 예시 수 (contexts 개수 초과 시 전체 사용)
    """

    @staticmethod
    def _answer_letter(raw: Any) -> str:
        return _ANSWER_MAP.get(raw, str(raw))

    def build(
        self,
        question: str,
        choices: dict[str, str],
        contexts: list[dict],
        num_examples: int = 3,
        **kwargs: Any,
    ) -> PromptResult:
        examples = contexts[:num_examples]

        shot_parts: list[str] = []
        for i, ctx in enumerate(examples, start=1):
            content = ctx.get("content_dict", {})
            ex_question = content.get("question", content.get("Question", ""))
            ex_a = content.get("A", "")
            ex_b = content.get("B", "")
            ex_c = content.get("C", "")
            ex_d = content.get("D", "")
            ex_answer = self._answer_letter(
                content.get("answer", content.get("Answer", ""))
            )

            shot_parts.append(
                f"[예시 {i}]\n"
                f"문제: {ex_question}\n"
                f"보기:\nA) {ex_a}\nB) {ex_b}\nC) {ex_c}\nD) {ex_d}\n"
                f"정답: {ex_answer}"
            )

        shot_block = "\n\n".join(shot_parts) if shot_parts else "(예시 없음)"

        choice_block = (
            f"A) {choices.get('A', '')}\n"
            f"B) {choices.get('B', '')}\n"
            f"C) {choices.get('C', '')}\n"
            f"D) {choices.get('D', '')}"
        )

        user_prompt = (
            "아래 예시를 참고하여, 마지막 문제의 정답 알파벳을 맞춰주세요.\n\n"
            "--- 예시 ---\n"
            f"{shot_block}\n\n"
            "--- 풀어야 할 문제 ---\n"
            f"문제: {question}\n"
            f"보기:\n{choice_block}\n\n"
            "정답 알파벳 (A, B, C, D 중 하나만 출력):"
        )

        return PromptResult(system_prompt=self.system_prompt, user_prompt=user_prompt)
