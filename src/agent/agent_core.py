"""
agent_core.py

RAG 파이프라인의 전체 오케스트레이션을 담당하는 객체입니다.
DataLoader, Retriever, PromptBuilder, OpenAIService 등을 통합하여
질의 응답(Unified interface)을 제공합니다.
"""

import asyncio
from typing import Literal

from hydra import compose, initialize
from omegaconf import DictConfig, ListConfig, OmegaConf

from agent.chunker import Chunker
from agent.data_loader import DataLoader
from agent.embedder import Embedder
from agent.prompt_builder import PromptBuilder
from agent.query_encoder import build_query_text
from agent.retriever import Retriever
from app.service.openai_service import OpenAIService
from config.config import Settings


class LegalRAGAgent:
    """
    법률 RAG 에이전트.
    초기화 시 설정된 전략(Hydra Config)에 따라 검색 및 프롬프트 생성을 수행합니다.
    """

    def __init__(self, settings: Settings) -> None:
        """
        에이전트 초기화 및 RAG 컴포넌트 조립 및 인덱싱(Phase 1).

        Args:
            settings: 애플리케이션 전역 설정 (RAG 전략 이름 포함)
        """
        self._settings = settings

        # 1. Hydra 설정 로드 (4대 전략 정밀 반영)
        with initialize(version_base=None, config_path="../../configs"):
            overrides = [
                f"retrieval={settings.rag_retrieval_strategy}",
                f"prompt={settings.rag_prompt_strategy}",
                f"query_representation={settings.rag_query_representation_strategy}",
                f"serialization={settings.rag_serialization_strategy}",
            ]
            if settings.rag_extra_overrides:
                # 쉼표로 구분된 추가 오버라이드 설정을 리스트에 확장
                overrides.extend(settings.rag_extra_overrides.split(","))

            self.cfg = compose(
                config_name="config",
                overrides=overrides,
            )

        # 2. 컴포넌트 인스턴스화
        self.data_loader = DataLoader()
        self.chunker = Chunker()
        self.openai_service = OpenAIService(settings=settings)
        self.embedder = Embedder(openai_service=self.openai_service)
        self.retriever = Retriever(config=self.cfg.retrieval, embedding_dim=1536)
        self.prompt_builder = PromptBuilder()

        # 3. 전략 파라미터 캐싱 (benchmark.py 로직)
        prompt_cfg_raw = OmegaConf.to_container(self.cfg.prompt, resolve=True)
        self.prompt_method = str(prompt_cfg_raw.pop("method"))
        prompt_cfg_raw.pop("description", None)
        self.prompt_kwargs = prompt_cfg_raw

        self.query_repr_method = str(self.cfg.query_representation.method)
        self.top_k = int(self.cfg.retrieval.get("top_k", 3))
        self.category_filter_enabled = bool(
            self.cfg.retrieval.get("category_filter", {}).get("enabled", False)
        )

        # 4. Phase 1: 인덱싱 (서버 기동 시 1회 수행)
        self._initialize_index()

    def _initialize_index(self) -> None:
        """benchmark.py의 Phase 1 로직을 수행하여 인덱스를 구축합니다."""
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        train_data_path = os.path.join(project_root, "data", "train.csv")
        
        # 데이터 로드
        train_rows = self.data_loader.load_csv(train_data_path)
        # 청킹
        chunks = self.chunker.chunk_data(train_rows)
        # 직렬화 및 임베딩 텍스트 추출
        serial_cfg_raw = OmegaConf.to_container(self.cfg.serialization, resolve=True)
        serial_exclude = list(serial_cfg_raw.get("exclude_fields", ["answer", "Human Accuracy"]))
        
        expand_chunks = []
        texts_to_embed = []
        for chunk in chunks:
            preprocessed = self.embedder.preprocess(
                chunk["content_dict"],
                method=self.cfg.serialization.method,
                exclude_fields=serial_exclude,
            )
            if isinstance(preprocessed, list):
                for text in preprocessed:
                    expand_chunks.append(chunk.copy())
                    texts_to_embed.append(text)
            else:
                expand_chunks.append(chunk)
                texts_to_embed.append(preprocessed)
                
        # 배치 임베딩 생성 (500개씩)
        batch_size = 500
        all_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_text = texts_to_embed[i : i + batch_size]
            batch_embeds = self.embedder.embed_batch(batch_text)
            all_embeddings.extend(batch_embeds)
            
        # 리트리버 적재
        self.retriever.add_documents(expand_chunks, all_embeddings)

    async def ask(self, query: str) -> Literal["A", "B", "C", "D"]:
        """
        자연어 형태의 질의를 받아 최종 정답(A/B/C/D)을 반환합니다.
        물음표(?)를 기준으로 질문과 선지를 분리하며, 벤치마크의 Phase 2 로직을 따릅니다.

        Args:
            query: 질문과 선지가 포함된 자연어 문자열

        Returns:
            "A", "B", "C", "D" 중 하나
        """
        # 1. 쿼리 파싱 (질문, 선지, 카테고리 추출)
        category, question, choices = self._parse_query_text(query)

        # 디버깅로그 (Docker logs에서 확인 가능)
        print(f"[debug] Category: {category}, Question: {question}")

        try:
            # 2. 쿼리 텍스트 구성 및 임베딩
            embedded_query_text = build_query_text(
                method=self.query_repr_method,
                question=question,
                choices=choices,
            )
            
            # 임베딩 생성
            query_vector = await asyncio.to_thread(self.embedder.embed, embedded_query_text)

            # 3. 관련 컨텍스트 검색 
            # 파싱된 카테고리가 있으면 필터 적용
            metadata_filter = {"Category": category} if category and category != "Unknown" else None
            
            retrieved_nodes = self.retriever.search(
                query_vector,
                top_k=self.top_k,
                metadata_filter=metadata_filter,
                query_text=embedded_query_text,
            )

            # 4. 프롬프트 생성 (benchmark.py : 296-303)
            prompt_result = self.prompt_builder.build_prompt(
                method=self.prompt_method,
                question=question,
                choices=choices,
                contexts=retrieved_nodes,
                **self.prompt_kwargs,
            )

            # 5. LLM 답변 생성 (benchmark.py : 305-309)
            generated_answer = await asyncio.to_thread(
                self.openai_service.generate_text,
                prompt=prompt_result.user_prompt,
                system_prompt=prompt_result.system_prompt,
            )

            # 6. 정답 추출 (benchmark.py : 312)
            pred = self._parse_prediction_label(generated_answer)
            
            # [debug] LLM 응답 및 추출 결과 출력
            print(f"[debug] Generated: {generated_answer.strip()[:100]}... -> Parsed: {pred}")

            # A, B, C, D가 아닌 경우 기본값 A
            if pred not in ["A", "B", "C", "D"]:
                return "A"
                
            return pred # type: ignore

        except Exception as e:
            print(f"Error during RAG inference: {e}")
            return "A"

    def _parse_query_text(self, query: str) -> tuple[str, str, dict[str, str]]:
        """
        입력 문자열에서 [Category], 질문, 선택지를 추출합니다.
        """
        import re
        
        # 1. 카테고리 추출 ([Category] 형태)
        category = "Unknown"
        cat_match = re.search(r"^\[(.*?)\]", query)
        if cat_match:
            category = cat_match.group(1).strip()
            # 카테고리 태그 제거
            query = query[cat_match.end():].strip()
        
        # 2. 질문 분리 (첫 번째 물음표 기준)
        parts = query.split("?", 1)
        question = parts[0] + "?" if len(parts) > 1 else parts[0]
        remaining = parts[1] if len(parts) > 1 else ""
        
        choices = {"A": "", "B": "", "C": "", "D": ""}
        
        # 3. 선지 추출 (1., 2., 3., 4. 또는 A:, B:, C:, D: 또는 A. B. ...)
        pattern = re.compile(
            r"(?:[1-4A-D][\.\)\s:])\s*(.*?)(?=\s*(?:[1-4A-D][\.\)\s:])|$)", 
            re.DOTALL
        )
        matches = pattern.findall(remaining)
        
        if len(matches) >= 4:
            choices["A"], choices["B"], choices["C"], choices["D"] = [m.strip() for m in matches[:4]]
        else:
            # 쉼표 구분자 대응 (CSV 형태 대비용)
            comma_parts = [p.strip() for p in remaining.split(",") if p.strip()]
            if len(comma_parts) >= 4:
                start_idx = 1 if comma_parts[0].isdigit() and len(comma_parts) > 4 else 0
                choices["A"], choices["B"], choices["C"], choices["D"] = comma_parts[start_idx:start_idx+4]

        return category, question.strip(), choices

    def _parse_prediction_label(self, text: str) -> str:
        """LLM 답변에서 정답 레이블(A, B, C, D)을 추출합니다."""
        if not text:
            return "ERR"

        # 텍스트에서 A, B, C, D 중 하나만 추출 (맨 앞 글자 우선 확인)
        text_clean = text.strip().upper()
        if text_clean and text_clean[0] in ["A", "B", "C", "D"]:
            return text_clean[0]

        # 정규식으로 단어 경계가 명확한 A-D 검색
        import re
        match = re.search(r"\b([A-D])\b", text_clean)
        if match:
            return match.group(1)

        return "ERR"
