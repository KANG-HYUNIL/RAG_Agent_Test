"""
openai_service.py

OpenAI API 클라이언트를 관리하고 추론(inference) 인터페이스를 제공하는 서비스 모듈입니다.

현재 상태: 골격(skeleton) — OpenAI 클라이언트 초기화 코드만 구현되어 있으며,
실제 RAG 파이프라인 추론 로직은 추후 구현 예정입니다.
"""

from typing import Literal

from openai import AsyncOpenAI

from config.config import Settings


class OpenAIService:
    """
    OpenAI API 연동 서비스 클래스.

    AsyncOpenAI 클라이언트를 싱글 인스턴스로 보유하며,
    임베딩 생성과 채팅 완성(chat completion) 기능을 제공합니다.

    Attributes:
        _client: AsyncOpenAI 비동기 클라이언트 인스턴스
        _settings: 애플리케이션 설정 인스턴스
    """

    def __init__(self, settings: Settings) -> None:
        """
        OpenAIService 초기화.

        Args:
            settings: 애플리케이션 전역 설정 (api_key, 모델명 등 포함)
        """
        # AsyncOpenAI 클라이언트를 초기화합니다. API 키는 설정에서 주입받습니다.
        self._client: AsyncOpenAI = AsyncOpenAI(api_key=settings.openai_api_key)
        self._settings: Settings = settings

    async def infer(self, query: str) -> Literal["A", "B", "C", "D"]:
        """
        법률 객관식 문제에 대한 정답 선지를 추론합니다.

        내부적으로 RAG 파이프라인(임베딩 → 유사 문서 검색 → 프롬프트 구성 → 생성)을
        실행하여 A/B/C/D 중 하나를 반환합니다.

        Args:
            query: 법률 객관식 문제 텍스트 (문제 + 선지 포함)

        Returns:
            정답 선지 문자열. 반드시 "A", "B", "C", "D" 중 하나.

        Raises:
            NotImplementedError: RAG 파이프라인 미구현 상태
        """
        # TODO: RAG 파이프라인 구현 후 아래 오류 제거
        # Step 1. query를 임베딩하여 벡터 생성
        # Step 2. 벡터 스토어에서 top-k 유사 문서(train.csv 샘플) 검색
        # Step 3. 검색된 few-shot 예제와 query를 조합하여 프롬프트 구성
        # Step 4. gpt-4o-mini에게 프롬프트 전달 → A/B/C/D 응답 수신 및 파싱
        raise NotImplementedError("RAG 파이프라인이 아직 구현되지 않았습니다.")

    async def create_embedding(self, text: str) -> list[float]:
        """
        텍스트의 임베딩 벡터를 생성합니다.

        Args:
            text: 임베딩할 입력 텍스트

        Returns:
            float 리스트 형태의 임베딩 벡터

        Raises:
            NotImplementedError: 임베딩 생성 로직 미구현 상태
        """
        # TODO: text-embedding-3-small 모델 호출 구현
        # response = await self._client.embeddings.create(
        #     model=self._settings.openai_embedding_model,
        #     input=text,
        # )
        # return response.data[0].embedding
        raise NotImplementedError("임베딩 생성 로직이 아직 구현되지 않았습니다.")

    async def chat_completion(self, prompt: str) -> str:
        """
        주어진 프롬프트에 대한 채팅 완성 응답을 반환합니다.

        Args:
            prompt: gpt-4o-mini에 전달할 완성된 프롬프트 텍스트

        Returns:
            모델 응답 텍스트 (원시 문자열)

        Raises:
            NotImplementedError: 채팅 완성 로직 미구현 상태
        """
        # TODO: gpt-4o-mini chat completion 호출 구현
        # response = await self._client.chat.completions.create(
        #     model=self._settings.openai_chat_model,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=1,
        #     temperature=0.0,
        # )
        # return response.choices[0].message.content or ""
        raise NotImplementedError("채팅 완성 로직이 아직 구현되지 않았습니다.")
