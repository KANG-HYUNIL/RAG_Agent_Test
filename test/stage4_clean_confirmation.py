import os
import subprocess
import pandas as pd
import glob
import sys
from datetime import datetime

# 후보군 설정 (4차-1 실험군)
CANDIDATES = [
    {"name": "stage4__confirm__k5__t035", "overrides": ["retrieval=score_threshold", "retrieval.top_k=5", "retrieval.score_threshold.value=0.35"]},
    {"name": "stage4__confirm__k5__t045", "overrides": ["retrieval=score_threshold", "retrieval.top_k=5", "retrieval.score_threshold.value=0.45"]},
    {"name": "stage4__confirm__topk__k5", "overrides": ["retrieval=top_k", "retrieval.top_k=5"]},
    {"name": "stage4__confirm__k5__t040", "overrides": ["retrieval=score_threshold", "retrieval.top_k=5", "retrieval.score_threshold.value=0.40"]},
    {"name": "stage4__confirm__k5__t030", "overrides": ["retrieval=score_threshold", "retrieval.top_k=5", "retrieval.score_threshold.value=0.30"]},
    {"name": "stage4__confirm__k7__t030", "overrides": ["retrieval=score_threshold", "retrieval.top_k=7", "retrieval.score_threshold.value=0.30"]},
]

def run_experiment(candidate):
    name = candidate["name"]
    overrides = candidate["overrides"]
    
    # 4차-1 고정 조건 명시 (혹시 config.yaml이 변경되었을 경우 대비)
    fixed_conditions = [
        "serialization=kv_pairs",
        "prompt=raw_stuffing",
        "query_representation=question_plus_choices",
        f"experiment_name={name}",
        f"hydra.run.dir=outputs/stage4_rerun/{name}"
    ]
    
    cmd = [sys.executable, "test/benchmark.py"] + fixed_conditions + overrides
    print(f"\n>>> Running Experiment: {name}")
    print(f">>> Command: {' '.join(cmd)}")
    
    try:
        # 실시간 로그 확인을 위해 stdout/stderr를 그대로 노출
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Experiment {name} failed with error: {e}")
        return False

def aggregate_results():
    print("\n>>> Aggregating All Stage 4-1 Results...")
    
    # 최근 생성된 outputs 디렉토리를 탐색하여 summary csv 수집
    # outputs/YYYY-MM-DD/HH-MM-SS 디렉토리 구조 가정
    base_output_dir = "outputs"
    all_summaries = []
    
    # 모든 benchmark_*.csv 파일을 찾아서 실험명(experiment_name)과 매칭
    # 여기서는 각 실험 실행 직후의 최신 파일을 찾는 로직이 필요하거나, 
    # 특정 패턴으로 저장된 파일을 수집합니다.
    
    csv_files = glob.glob(os.path.join(base_output_dir, "**", "benchmark_*.csv"), recursive=True)
    
    for info in CANDIDATES:
        name = info["name"]
        # 해당 실험명으로 생성된 가장 최신 리포트 찾기
        # (간단하게 파일 내용 중 strategy나 summary 파트에서 확인하거나 파일 경로 시간순 정렬)
        relevant_files = [f for f in csv_files if any(name in f for name in [name])] # 경로에 name이 포함된 경우
        
        # 만약 경로에 name이 없다면, 파일 내부를 읽어서 확인해야 할 수도 있으나 
        # 일단은 가장 최근 생성된 파일들 중 CANDIDATES 수만큼 가져오는 방식으로 시도
        pass

    # 위 방식보다 더 확실한 방식: 최근 24시간 내 생성된 모든 benchmark_*.csv 읽어서 merge
    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # 요약 행 추출 (맨 아래 'SUMMARY' 행)
            summary_row = df[df["#"] == "SUMMARY"]
            if not summary_row.empty:
                # 파싱: total=..., avg=..., acc=...
                time_info = str(summary_row.iloc[0]["Time(s)"])
                # acc 추출
                acc_val = time_info.split("acc=")[-1].replace("%", "") if "acc=" in time_info else "0"
                
                # Category별 성능 등 추가 정보가 benchmark_summary.csv에 있다면 그것도 수집 가능
                # 현재 benchmark.py는 CSV 하단에 요약 행을 추가함
                
                all_data.append({
                    "Experiment": summary_row.iloc[0]["Question"].replace("strategy=", ""),
                    "Accuracy": float(acc_val),
                    "Result_File": os.path.basename(f)
                })
        except:
            continue

    if all_data:
        report_df = pd.DataFrame(all_data)
        report_df = report_df.sort_values(by="Accuracy", ascending=False)
        report_path = f"stage4_final_comparison_{datetime.now().strftime('%m%d_%H%M')}.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n[✔] Final Comparison Report Created: {report_path}")
        print(report_df.to_string(index=False))
    else:
        print("\n[!] No results found to aggregate.")

if __name__ == "__main__":
    success_count = 0
    for cand in CANDIDATES:
        if run_experiment(cand):
            success_count += 1
    
    print(f"\n>>> Completed {success_count}/{len(CANDIDATES)} experiments.")
    aggregate_results()
