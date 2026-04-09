import os
import sys
import time
import subprocess
import csv
import json
from datetime import datetime
from pathlib import Path
import httpx
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Windows 환경에서 한국어 인코딩 문제 해결
if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
DEV_CSV = DATA_DIR / "dev.csv"
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs" / "stage8"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

IMAGE_NAME = "legal-rag-inference:latest"
CONTAINER_NAME = "legal-rag-test-container"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
HEALTH_URL = f"{BASE_URL}/health"
INFERENCE_URL = f"{BASE_URL}/"

# 실험군 설정 (Candidate Grid)
# 8차 실험의 핵심은 'No Category Filter' 환경에서의 성능 검증입니다.
# 7-2차 결과(Answer 필드가 성능에 부정적 영향)를 반영하여 answer_removal(raw_stuffing의 기본값)을 상정합니다.
CANDIDATES = [
    {
        "id": "8-1",
        "retrieval": "score_threshold",
        "overrides": "++retrieval.category_filter.enabled=false,retrieval.score_threshold.value=0.35,retrieval.top_k=5",
        "desc": "Baseline (Score Threshold / No Filter)"
    },
    {
        "id": "8-2",
        "retrieval": "rerank_stage7_total",
        "overrides": "++retrieval.category_filter.enabled=false,retrieval.rerank.penalty_polarity=1.0,retrieval.rerank.bonus_choice=1.0",
        "desc": "Statute-focused (Polarity/Choice Penalty Disabled / No Filter)"
    },
    {
        "id": "8-3",
        "retrieval": "rerank_stage7_total",
        "overrides": "++retrieval.category_filter.enabled=false",
        "desc": "Full Complex Pipeline (Statute+Polarity+Choice / No Filter)"
    }
]

def setup_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def build_docker_image():
    print(f"[*] Building Docker image: {IMAGE_NAME}...")
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."], check=True, cwd=PROJECT_ROOT)

def run_container(candidate: Dict[str, str]):
    print(f"[*] Starting container for experiment {candidate['id']}...")
    env = [
        "-e", f"OPENAI_API_KEY={OPENAI_API_KEY}",
        "-e", f"RAG_RETRIEVAL_STRATEGY={candidate['retrieval']}",
    ]
    if candidate.get("overrides"):
        env += ["-e", f"RAG_EXTRA_OVERRIDES={candidate['overrides']}"]

    subprocess.run([
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "-p", f"{PORT}:8000",
        *env,
        IMAGE_NAME
    ], check=True)

def stop_container():
    print("[*] Stopping and removing container...")
    subprocess.run(["docker", "stop", CONTAINER_NAME], check=False, capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], check=False, capture_output=True)

def save_container_logs(exp_dir: Path):
    """컨테이너 로그를 바이너리 모드로 저장하여 인코딩 오류 방지"""
    print(f"[*] Saving container logs to {exp_dir / 'container.log'}...")
    try:
        # text=True 제거 (기본값 False), 바이너리로 캡처
        result = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, check=False)
        with open(exp_dir / "container.log", "wb") as f:
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
    except Exception as e:
        print(f"[!] Failed to save logs: {e}")

def wait_for_server(timeout=120):
    print("[*] Waiting for server to be ready (Max 120s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with httpx.Client() as client:
                response = client.get(HEALTH_URL, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok":
                        print(f"[+] Server is ready! ({int(time.time() - start_time)}s)")
                        return True
        except Exception:
            pass
        time.sleep(5)
    
    # 실패 시 진단 로그 출력 (전체 로그 출력)
    print(f"[!] Server failed to start within {timeout}s.")
    print("[*] --- FULL CONTAINER LOGS FOR DIAGNOSIS ---")
    try:
        diag = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, check=False)
        # Windows 콘솔 출력을 위해 에어 무시하고 디코딩 시도
        stdout_text = diag.stdout.decode('utf-8', errors='replace')
        stderr_text = diag.stderr.decode('utf-8', errors='replace')
        if stdout_text:
            print(stdout_text)
        if stderr_text:
            print("--- STDERR ---")
            print(stderr_text)
    except Exception as e:
        print(f"[!] Could not retrieve diagnostic logs: {e}")
    print("[*] -----------------------------------------")
    return False

def convert_row_to_query(row: Dict[str, Any]) -> str:
    """AgentCore._parse_query_text가 정상 분리할 수 있는 형태로 직렬화"""
    question = row['question']
    if not question.endswith('?'):
        question += '?'
    
    # 규격화된 포맷: [Category] Question? 1. A, 2. B, 3. C, 4. D
    category = row.get('Category', 'Unknown')
    choices_str = f"1. {row['A']}, 2. {row['B']}, 3. {row['C']}, 4. {row['D']}"
    
    return f"[{category}] {question} {choices_str}"

def run_evaluation(candidate: Dict[str, str], exp_dir: Path) -> List[Dict[str, Any]]:
    print(f"[*] Running evaluation for {candidate['id']}...")
    results = []
    
    # 데이터 로드
    df = pd.read_csv(DEV_CSV)
    
    # HTTPX Client 세션 사용으로 성능 최적화
    with httpx.Client(timeout=60.0) as client:
        for idx, row in df.iterrows():
            query_text = convert_row_to_query(row.to_dict())
            
            payload = {
                "query": query_text
            }
            
            start_ts = time.time()
            try:
                response = client.post(INFERENCE_URL, json=payload)
                elapsed = time.time() - start_ts
                
                if response.status_code == 200:
                    data = response.json()
                    pred = str(data["answer"])
                    gt_raw = str(row["answer"])
                    # Answer mapping: 서버 응답(A..D)을 CSV 라벨(1..4)로 매칭
                    ans_map = {
                        "A": "1", "B": "2", "C": "3", "D": "4"
                    }
                    correct = ans_map.get(pred) == gt_raw
                    
                    results.append({
                        "ID": idx,
                        "Category": row["Category"],
                        "GT": gt_raw,
                        "Pred": pred,
                        "Correct": "O" if correct else "X",
                        "Latency": elapsed
                    })
                else:
                    print(f"[!] Error at ID {idx}: {response.text}")
                    results.append({"ID": idx, "Error": response.text})
            except Exception as e:
                print(f"[!] Exception at ID {idx}: {str(e)}")
                results.append({"ID": idx, "Error": str(e)})

            if (idx + 1) % 10 == 0:
                print(f"    - Progress: {idx+1}/{len(df)}")

    # 개별 결과 CSV 저장
    res_df = pd.DataFrame(results)
    csv_path = exp_dir / f"benchmark_{candidate['id']}.csv"
    res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 메타데이터 저장
    with open(exp_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(candidate, f, indent=4, ensure_ascii=False)

    return results

def aggregate_all_results(all_exp_results: List[Dict], run_dir: Path, ts: str):
    """stage7_sweep.py와 동일한 집계 로직 제공"""
    summary_data = []
    for exp in all_exp_results:
        if not exp.get("data"):
            print(f"[!] No data for experiment {exp['id']}, skipping summary entry.")
            continue
            
        df = pd.DataFrame(exp["data"])
        if "Correct" in df.columns:
            total = len(df)
            valid_df = df[df["Correct"].isin(["O", "X"])]
            if valid_df.empty:
                acc = 0
                avg_lat = 0
            else:
                correct_count = (valid_df["Correct"] == "O").sum()
                acc = (correct_count / total * 100) if total > 0 else 0
                avg_lat = pd.to_numeric(valid_df["Latency"]).mean()
            
            summary_row = {
                "ID": exp["id"],
                "Accuracy": f"{acc:.2f}%",
                "AvgLatency": f"{avg_lat:.2f}s",
                "Description": exp["desc"]
            }
            
            # Category별 Accuracy (Law, Criminal Law)
            for cat in ["Law", "Criminal Law"]:
                cat_df = df[df["Category"] == cat]
                if not cat_df.empty and "Correct" in cat_df.columns:
                    c_total = len(cat_df)
                    c_correct = (cat_df["Correct"] == "O").sum()
                    summary_row[f"Acc_{cat}"] = f"{(c_correct/c_total*100):.2f}%"
                else:
                    summary_row[f"Acc_{cat}"] = "0.00%"
            
            summary_data.append(summary_row)

    summary_df = pd.DataFrame(summary_data)
    
    # Markdown 리포트 생성
    md_path = run_dir / f"summary_table_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Stage 8 Inference E2E Test Summary\n\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("### Analysis Points:\n")
        f.write("- **No Category Filter**: Actual inference environment evaluation.\n")
        f.write("- **Answer Removal**: Prompt strategy applied to prevent data leakage/bias.\n")
        f.write("- **Docker E2E**: Verified with full container interaction.\n")

    print(f"\n[+] All experiments completed. Summary saved to: {md_path}")
    print(summary_df.to_string(index=False))

def main():
    if not OPENAI_API_KEY:
        print("[!] Error: OPENAI_API_KEY environment variable is not set.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = setup_run_dir()
    
    # 이미지 빌드는 루프 밖에서 1회 수행
    build_docker_image()
    
    all_exp_results = []
    
    for candidate in CANDIDATES:
        exp_dir = run_dir / candidate["id"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[*] Starting Experiment: {candidate['id']} ({candidate['desc']})")
        print(f"{'='*60}")
        
        try:
            stop_container() # 사전 정리
            run_container(candidate)
            
            if wait_for_server():
                results_data = run_evaluation(candidate, exp_dir)
                all_exp_results.append({
                    "id": candidate["id"],
                    "desc": candidate["desc"],
                    "data": results_data
                })
            else:
                print(f"[!] Server failed to start for {candidate['id']}")
            
        except Exception as e:
            print(f"[!] Experiment {candidate['id']} failed: {str(e)}")
        finally:
            save_container_logs(exp_dir)
            stop_container()
            print(f"[*] Cooldown (5s)...")
            time.sleep(5)

    aggregate_all_results(all_exp_results, run_dir, ts)

if __name__ == "__main__":
    main()
