import os
import time
import json
import csv
import logging
import subprocess
import httpx
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 상수 설정
IMAGE_NAME = "legal-rag-inference:latest"
CONTAINER_NAME = "legal-rag-eval-final"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
QUERY_ENDPOINT = f"{BASE_URL}/query"

# 출력 경로 설정
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"outputs/final_eval/run_{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "evaluation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, check=True):
    """쉘 명령어를 실행합니다."""
    logger.info(f"Executing command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def build_docker():
    """도커 이미지를 빌드합니다."""
    logger.info("Building Docker image...")
    run_command(f"docker build -t {IMAGE_NAME} .")

def start_container():
    """최종 8-3 옵션으로 컨테이너를 실행합니다."""
    logger.info("Starting container with Strategy 8-3 (Full Complex Pipeline)...")
    
    # 기존 컨테이너가 있으면 제거
    run_command(f"docker rm -f {CONTAINER_NAME}", check=False)
    
    # 환경 변수 준비
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env or environment variables.")
    
    # 8-3 환경 변수 설정
    cmd = (
        f"docker run -d --name {CONTAINER_NAME} -p {SERVER_PORT}:8000 "
        f"-e OPENAI_API_KEY={api_key} "
        f"-e RAG_RETRIEVAL_STRATEGY=rerank_stage7_total "
        f"-e RAG_QUERY_REPR_STRATEGY=question_plus_choices "
        f"-e RAG_PROMPT_STRATEGY=raw_stuffing "
        f"{IMAGE_NAME}"
    )
    run_command(cmd)

def wait_for_server(timeout=120):
    """서버가 준비될 때까지 대기합니다."""
    logger.info(f"Waiting for server to be healthy (timeout: {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(HEALTH_ENDPOINT)
            if response.status_code == 200:
                logger.info("Server is healthy!")
                return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("Server health check timed out.")

def format_query(row):
    """서버 규격에 맞게 쿼리를 생성합니다: [Category] Question? Choices"""
    category = row['Category']
    question = row['question']
    choices = f"A: {row['A']}, B: {row['B']}, C: {row['C']}, D: {row['D']}"
    return f"[{category}] {question} {choices}"

def map_label(csv_label):
    """CSV의 1,2,3,4를 A,B,C,D로 변환합니다."""
    mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', "1": 'A', "2": 'B', "3": 'C', "4": 'D'}
    return mapping.get(csv_label, "ERR")

def run_inference(test_csv_path):
    """테스트 데이터에 대해 추론을 수행합니다."""
    df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(df)} rows from {test_csv_path}")
    
    results = []
    correct_count = 0
    start_eval_time = time.time()
    
    for idx, row in df.iterrows():
        query_text = format_query(row)
        expected = map_label(row['answer'])
        
        logger.info(f"[{idx+1}/{len(df)}] Querying: {query_text[:50]}...")
        
        request_start = time.time()
        try:
            resp = httpx.post(QUERY_ENDPOINT, json={"text": query_text}, timeout=60.0)
            resp_data = resp.json()
            prediction = resp_data.get("prediction", "ERR")
            latency = time.time() - request_start
        except Exception as e:
            logger.error(f"Request failed: {e}")
            prediction = "ERR"
            latency = 0
            resp_data = {"error": str(e)}
            
        is_correct = (prediction == expected)
        if is_correct:
            correct_count += 1
            
        results.append({
            "idx": idx,
            "category": row['Category'],
            "question": row['question'],
            "expected": expected,
            "predicted": prediction,
            "correct": is_correct,
            "latency": latency,
            "response_raw": resp_data
        })
        
    total_time = time.time() - start_eval_time
    avg_latency = total_time / len(df) if len(df) > 0 else 0
    accuracy = (correct_count / len(df) * 100) if len(df) > 0 else 0
    
    return results, accuracy, avg_latency

def save_results(results, accuracy, avg_latency, test_csv_name):
    """결과를 파일로 저장합니다."""
    summary = {
        "test_file": test_csv_name,
        "timestamp": TIMESTAMP,
        "accuracy": f"{accuracy:.2f}%",
        "avg_latency": f"{avg_latency:.2f}s",
        "total_count": len(results),
        "strategy": "8-3 (rerank_stage7_total)"
    }
    
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "details.csv", index=False, encoding="utf-8-sig")
    
    # 카테고리별 정확도
    cat_acc = res_df.groupby("category")["correct"].mean() * 100
    cat_acc.to_csv(OUTPUT_DIR / "category_accuracy.csv", encoding="utf-8-sig")
    
    logger.info("=" * 50)
    logger.info(f"Evaluation Complete!")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Avg Latency: {avg_latency:.2f}s")
    logger.info(f"Category Accuracy:\n{cat_acc}")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 50)

def main():
    # 1. 파일 경로 결정
    # 환경변수 TEST_CSV 확인 (예: TEST_CSV=test.csv)
    # data/ 폴더 하위에 있는 것으로 간주함. 전체 경로가 주어지면 그대로 사용.
    test_file_input = os.getenv("TEST_CSV", "dev.csv")
    if os.path.exists(test_file_input):
        test_csv_path = test_file_input
    else:
        test_csv_path = f"data/{test_file_input}"
        
    if not os.path.exists(test_csv_path):
        logger.error(f"Test file not found: {test_csv_path}")
        return

    try:
        # 2. 도커 환경 자동 구축
        build_docker()
        start_container()
        wait_for_server()
        
        # 3. 평가 실행
        results, accuracy, avg_latency = run_inference(test_csv_path)
        
        # 4. 결과 정리
        save_results(results, accuracy, avg_latency, test_file_input)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up container...")
        run_command(f"docker rm -f {CONTAINER_NAME}", check=False)

if __name__ == "__main__":
    main()
