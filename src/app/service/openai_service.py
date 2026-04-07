import os
from openai import OpenAI
from config.config import get_settings

class OpenAIService:
    """
    OpenAI API 호출을 전담하는 서비스 클래스.
    설정된 API Key와 Model을 사용하여 텍스트 생성 및 임베딩을 수행합니다.
    """
    def __init__(self):
        
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.openai_chat_model
        self.embedding_model = settings.openai_embedding_model

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """
        LLM(gpt-4o-mini 등)을 사용하여 텍스트를 생성합니다.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> list[float]:
        # TODO : 임베딩 벡터 변환 전 전처리 전략을 여기에도 넣을건지 고민 및 확정 필요
        
        """
        단일 텍스트의 임베딩 벡터를 반환합니다.
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model # config.yaml 기준: text-embedding-3-small
        )
        return response.data[0].embedding
        
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        다중 텍스트(Batch)의 임베딩 벡터를 한 번의 API 호출로 반환하여 속도를 높입니다.
        """
        clean_texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(
            input=clean_texts,
            model=self.embedding_model # text-embedding-3-small
        )
        # 반환된 리스트는 넣은 텍스트의 순서와 동일하게 유지됩니다.
        return [data.embedding for data in response.data]
