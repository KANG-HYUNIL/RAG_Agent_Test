"""
openai_service.py

OpenAI API 호출(텍스트 생성, 임베딩, 추론)을 전담하는 서비스 클래스입니다.
"""

import asyncio
from typing import Literal

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from config.config import Settings


class OpenAIService:
    """
    OpenAI API 호출을 전담하는 서비스 클래스.

    텍스트 생성(generate_text), 임베딩(get_embedding / get_embeddings),
    RAG 추론(infer)을 담당합니다.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._embedding_model = settings.openai_embedding_model

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """
        LLM(gpt-4o-mini 등)을 사용하여 텍스트를 생성합니다.

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (빈 문자열이면 생략)

        Returns:
            LLM이 생성한 텍스트

        Raises:
            ValueError: LLM이 빈 응답을 반환한 경우
        """
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI가 빈 응답을 반환했습니다.")
        return content

    def get_embedding(self, text: str) -> list[float]:
        """
        단일 텍스트의 임베딩 벡터를 반환합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            float 리스트 (dim=1536, text-embedding-3-small 기준)
        """
        clean = text.replace("\n", " ")
        response = self._client.embeddings.create(
            input=[clean],
            model=self._embedding_model,
        )
        return response.data[0].embedding

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        다중 텍스트(Batch)의 임베딩 벡터를 한 번의 API 호출로 반환합니다.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트 (입력 순서 보장)
        """
        clean_texts = [t.replace("\n", " ") for t in texts]
        response = self._client.embeddings.create(
            input=clean_texts,
            model=self._embedding_model,
        )
        return [data.embedding for data in response.data]

    async def infer(self, query: str) -> Literal["A", "B", "C", "D"]:
        """
        법률 객관식 문제를 받아 A/B/C/D 정답을 반환합니다.

        TODO: 전략 레지스트리 기반 RAG 파이프라인 연결 예정.
              (Embedder → Retriever → PromptBuilder → generate_text)
              전략 설정 방식(ENV 변수 기반) 결정 후 구현.
              현재는 LLM 직접 호출 방식으로 동작.

        Args:
            query: 법률 객관식 문제 텍스트

        Returns:
            A, B, C, D 중 하나

        Raises:
            ValueError: LLM이 유효하지 않은 답변을 반환한 경우
        """
        system_prompt = (
            "당신은 법률 객관식 문제를 분석하는 전문가입니다. "
            "반드시 A, B, C, D 중 하나만 대답하세요. 다른 말은 하지 마세요."
        )

        result = await asyncio.to_thread(self.generate_text, query, system_prompt)
        stripped = result.strip().upper()

        match stripped:
            case "A":
                return "A"
            case "B":
                return "B"
            case "C":
                return "C"
            case "D":
                return "D"
            case _:
                raise ValueError(f"LLM이 유효하지 않은 답변을 반환했습니다: {result!r}")
