"""
legal_rag_service.py

FastAPI 애플리케이션의 서비스 레이어입니다.
LegalRAGAgent의 인스턴스를 관리하며, 컨트롤러와의 인터페이스를 담당합니다.
"""

from typing import Literal

from fastapi import Request

from agent.agent_core import LegalRAGAgent
from config.config import Settings


class LegalRAGService:
    """
    RAG 추론 요청을 처리하는 응용 서비스.
    에이전트 인스턴스를 래핑하여 단일 인터페이스를 제공합니다.
    """

    def __init__(self, agent: LegalRAGAgent) -> None:
        self._agent = agent

    async def ask(self, query: str) -> Literal["A", "B", "C", "D"]:
        """
        에이전트에게 질의를 전달하고 정답을 반환합니다.

        Args:
            query: 법률 문제 텍스트

        Returns:
            A, B, C, D 중 하나
        """
        return await self._agent.ask(query=query)


def get_legal_rag_service(request: Request) -> LegalRAGService:
    """
    FastAPI Depends()에서 사용할 의존성 팩토리 함수.
    app.state.agent 에 저장된 LegalRAGAgent 인스턴스를 주입받아
    서비스 인격(Service Wrapper)을 생성하여 반환합니다.
    """
    # app.state.agent는 server.py의 lifespan에서 초기화됩니다.
    agent: LegalRAGAgent = request.app.state.agent
    return LegalRAGService(agent=agent)
