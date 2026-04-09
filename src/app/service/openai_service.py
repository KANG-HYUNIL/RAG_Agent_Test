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

    텍스트 생성(generate_text) 및 임베딩(get_embedding / get_embeddings) 기능을 제공합니다.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._embedding_model = settings.openai_embedding_model

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """
        LLM(gpt-4o-mini 등)을 사용하여 텍스트를 생성합니다.
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
        """
        clean_texts = [t.replace("\n", " ") for t in texts]
        # 큰 배치의 경우 chunking이 필요할 수 있으나 현재 규모에서는 직접 호출
        response = self._client.embeddings.create(
            input=clean_texts,
            model=self._embedding_model,
        )
        return [data.embedding for data in response.data]
