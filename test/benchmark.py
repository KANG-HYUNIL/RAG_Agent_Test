import sys
import os
import io
import logging

# 콘솔 출력 시 인코딩 문제 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# src 모듈 임포트를 위해 src 폴더를 PYTHONPATH에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra
from omegaconf import DictConfig, OmegaConf

# RAG 에이전트 모듈 임포트
from agent.data_loader import DataLoader
from agent.chunker import Chunker
from agent.embedder import Embedder
from app.service.openai_service import OpenAIService
from agent.retriever import Retriever

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info("="*50)
    log.info("RAG Pipeline Benchmark")
    log.info("="*50)
    log.info("1. Active Configuration:")
    log.info("\n" + OmegaConf.to_yaml(cfg))
    log.info("="*50)

    # 1. 파이프라인 초기화
    data_loader = DataLoader()
    chunker = Chunker()
    openai_service = OpenAIService()
    embedder = Embedder(openai_service=openai_service)
    retriever = Retriever(config=cfg.retrieval, embedding_dim=1536) # text-embedding-3-small의 차원 1536
    
    # 2. 데이터 경로 결정 (어느 환경에서도 동작하도록 절대 경로 이용)
    train_data_path = os.path.join(project_root, "data", "train.csv")
    
    # --- [오프라인 인덱싱 페이즈 (Offline Indexing Phase)] ---
    log.info("\n[Phase 1] Indexing (데이터 적재)")
    
    # 2.1 DataLoader를 통해 데이터 로드
    log.info("[1/4] 데이터 로드 중...")
    train_rows = data_loader.load_csv(train_data_path)
    log.info(f"  -> 총 {len(train_rows)}개의 학습 문서 로드 완료.")
    
    # 2.2 Chunker를 통해 정규화
    log.info("[2/4] 데이터 청킹 (Chunking)...")
    chunks = chunker.chunk_data(train_rows)
    log.info(f"  -> {len(chunks)}개의 청크 구조화 완료.")
    
    # 2.3 Embedder 전처리 및 텍스트화
    log.info(f"[3/4] 전처리 및 임베딩 텍스트 추출 중... (전략: {cfg.serialization.method})")
    expand_chunks = []
    texts_to_embed = []
    
    for chunk in chunks:
        preprocessed = embedder.preprocess(chunk["content_dict"], method=cfg.serialization.method)
        # 메서드가 여러 문자열의 리스트를 반환하는 경우 (예: Dual Representation)
        if isinstance(preprocessed, list):
            for text in preprocessed:
                expand_chunks.append(chunk.copy())
                texts_to_embed.append(text)
        else:
            expand_chunks.append(chunk)
            texts_to_embed.append(preprocessed)
            
    log.info(f"  -> API 호출을 통하여 총 {len(texts_to_embed)} 개의 벡터를 생성합니다.")
    
    # 2.4 배치 임베딩 추출 (API Threshold 고려해 청크 단위로 요청)
    batch_size = 500
    all_embeddings = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch_text = texts_to_embed[i:i+batch_size]
        batch_embeds = embedder.embed_batch(batch_text)
        all_embeddings.extend(batch_embeds)
        log.info(f"  -> ({min(i + batch_size, len(texts_to_embed))} / {len(texts_to_embed)}) 임베딩 생성 완료...")
    
    # 2.5 FAISS 벡터 DB에 적재
    log.info("[4/4] 로컬 InMemory FAISS Vector DB에 형태소 적재 중...")
    retriever.add_documents(expand_chunks, all_embeddings)
    log.info(f"  -> 적재 완료! (총 인덱스 수: {retriever.index.ntotal})")
    
    log.info("\n[✔] RAG 시스템 초기화 완료 및 준비 끝. 평가 루프를 시작할 수 있습니다.")

    # 3. Prompt Builder 초기화
    from agent.prompt_builder import PromptBuilder
    prompt_builder = PromptBuilder()

    # --- [온라인 추론 및 평가 페이즈 (Online Inference & Evaluation Phase)] ---
    log.info("\n[Phase 2] Inference & Evaluation (dev 데이터 평가)")
    dev_data_path = os.path.join(project_root, "data", "dev.csv")
    dev_rows = data_loader.load_csv(dev_data_path)
    log.info(f"  -> 평가 데이터 로드 완료 (총 {len(dev_rows)}개)")
    
    correct_count = 0
    total_eval = len(dev_rows)

    top_k = cfg.retrieval.k if 'k' in cfg.retrieval else 3
    
    log.info("\n평가를 시작합니다...")
    
    # 1,2,3,4 정답을 A,B,C,D로 맵핑하기 위한 딕셔너리
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}

    for idx, row in enumerate(dev_rows):
        question = str(row.get("question", ""))
        
        # 정답(1, 2, 3, 4)를 A, B, C, D로 변환합니다. 변환 실패 시 그대로 사용
        raw_answer_idx = str(row.get("answer", "")).strip()
        answer_label = label_map.get(raw_answer_idx, raw_answer_idx)
        
        # 보기도 A, B, C, D 키로 전달합니다.
        choices = {
            "A": str(row.get("A", "")),
            "B": str(row.get("B", "")),
            "C": str(row.get("C", "")),
            "D": str(row.get("D", ""))
        }
        
        # 실제 질의에는 질문 내용만 사용하거나, 보기와 결합하여 사용할 수 있습니다.
        # 여기서는 baseline으로 질문 자체만 임베딩하여 Retrieval을 수행합니다.
        try:
            # 1. 쿼리 임베딩
            query_vector = embedder.embed(question)
            
            # 2. 관련 컨텍스트 검색
            retrieved_nodes = retriever.search(query_vector, top_k=top_k)
            
            # 3. LLM Prompt 생성
            prompt = prompt_builder.build_mcq_prompt(question, choices, retrieved_nodes)
            
            # 4. LLM 답변 생성
            generated_answer = openai_service.generate_text(prompt, system_prompt=prompt_builder.system_prompt)
            
            # 5. 정답 확인 (생성된 답변에 정답 알파벳이 정확히 포함되어 있는지)
            pred = generated_answer.strip().upper()
            
            is_correct = False
            # LLM이 단일 알파벳만 뱉도록 가이드했으나, 혹시 모를 주변 텍스트 포함 시 in 검사 사용
            if answer_label == pred or answer_label in pred:
                correct_count += 1
                is_correct = True
                
            if idx % 10 == 0:
                log.info(f"[{idx}/{total_eval}] Q: {question[:30]}... | GT: {answer_label} | Pred: {pred} | Corr: {is_correct}")
                
        except Exception as e:
            if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
                log.error("\n[!] OpenAI API Key가 올바르지 않거나 설정되지 않았습니다. 테스트/평가 루프를 중지합니다.")
                log.error("[!] 실제 API Key를 설정 후 다시 실행해주세요. (export OPENAI_API_KEY='sk-...')")
                break
            else:
                log.error(f"[!] Error on row {idx}: {e}")
                
    if total_eval > 0 and correct_count > 0:
        accuracy = (correct_count / total_eval) * 100
        log.info(f"\n[평가 결과] 총 {total_eval} 문제 중 {correct_count} 문제 정답 (정확도: {accuracy:.2f}%)")
    else:
        log.warning("\n[평가 결과] 평가가 성공적으로 완료되지 않았거나 정답을 맞춘 문제가 0건입니다.")

if __name__ == "__main__":
    main()
