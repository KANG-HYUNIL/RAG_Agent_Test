from ._registry import _PROMPT_REGISTRY


class PromptBuilder:
    """
    검색된 데이터를 바탕으로 LLM(gpt-4o-mini)에 전달할 최종 프롬프트를 구성합니다.
    Registry Pattern을 통해 다양한 프롬프트 전략(MCQ, 요약, 서술형 등)을 동적으로 확장합니다.
    """

    def __init__(self) -> None:
        self.system_prompt = (
            "당신은 대한민국 법률 전문가이자 정답 분류기입니다. "
            "주어진 컨텍스트(참고 자료)를 바탕으로 객관식 문제의 올바른 정답 알파벳을 정확하게 선택해야 합니다. "
            "반드시 A, B, C, D 중 하나의 알파벳만 답변으로 출력하십시오. 다른 설명은 절대 추가하지 마십시오."
        )

    def build_mcq_prompt(self, question: str, choices: dict, contexts: list) -> str:
        """
        컨텍스트와 질문, 보기(A, B, C, D)를 조합하여 MCQ 프롬프트를 생성합니다.

        :param question: 문제 내용
        :param choices: {"A": "...", "B": "...", "C": "...", "D": "..."} 형태의 보기
        :param contexts: 검색된 참고 자료 리스트
        :return: LLM에 전달할 사용자 프롬프트 문자열
        """
        strategy = _PROMPT_REGISTRY["mcq"]()
        return strategy.build(question=question, choices=choices, contexts=contexts)
