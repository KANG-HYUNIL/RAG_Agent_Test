from ._registry import _PREPROCESS_REGISTRY, BasePreprocessStrategy, register_preprocess


@register_preprocess("synthetic_query_expansion")
class SyntheticQueryExpansionPreprocessStrategy(BasePreprocessStrategy):
    """
    5. Synthetic Query Expansion
    해당 문서가 답할 법한 예상 질문을 LLM으로 생성해서 문서 내용 뒤에 붙임.
    openai_service가 없으면 kv_pairs 결과만 반환 (graceful fallback).
    """

    def process(self, row: dict) -> str:
        base_text = _PREPROCESS_REGISTRY["kv_pairs"](
            openai_service=self.openai_service
        ).process(row)
        # kv_pairs 전략은 항상 str을 반환하므로 str로 단언
        if not isinstance(base_text, str):
            base_text = " ".join(base_text)
        if not self.openai_service:
            return base_text

        prompt = (
            f"다음 법률 객관식 문제를 바탕으로, 사용자가 질문할 만한 유사한 단답형/서술형 질문 1개를 작성하세요.\n"
            f"문제 정보: {base_text}\n"
            f"예상 질문:"
        )

        hypothetical_query = self.openai_service.generate_text(prompt).strip()
        return f"{base_text}\n가상 질문(Synthetic Query): {hypothetical_query}"
