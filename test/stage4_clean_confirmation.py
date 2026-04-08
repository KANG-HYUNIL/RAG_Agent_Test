import glob
import os
import subprocess
import sys

import pandas as pd

# 후보군 설정 (4차-1 실험군)
CANDIDATES = [
    {
        "name": "stage4__confirm__k5__t035",
        "overrides": [
            "retrieval=score_threshold",
            "retrieval.top_k=5",
            "retrieval.score_threshold.value=0.35",
        ],
        "remark": "3차 공동 최고",
    },
    {
        "name": "stage4__confirm__k5__t045",
        "overrides": [
            "retrieval=score_threshold",
            "retrieval.top_k=5",
            "retrieval.score_threshold.value=0.45",
        ],
        "remark": "3차 공동 최고",
    },
    {
        "name": "stage4__confirm__topk__k5",
        "overrides": ["retrieval=top_k", "retrieval.top_k=5"],
        "remark": "기존 강한 기준선",
    },
    {
        "name": "stage4__confirm__k5__t040",
        "overrides": [
            "retrieval=score_threshold",
            "retrieval.top_k=5",
            "retrieval.score_threshold.value=0.40",
        ],
        "remark": "최적점 주변",
    },
    {
        "name": "stage4__confirm__k5__t030",
        "overrides": [
            "retrieval=score_threshold",
            "retrieval.top_k=5",
            "retrieval.score_threshold.value=0.30",
        ],
        "remark": "느슨한 cutoff",
    },
    {
        "name": "stage4__confirm__k7__t030",
        "overrides": [
            "retrieval=score_threshold",
            "retrieval.top_k=7",
            "retrieval.score_threshold.value=0.30",
        ],
        "remark": "시간 오염 재확인",
    },
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
        f"hydra.run.dir=outputs/stage4_rerun/{name}",
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

    summary_results = []
    category_results = []

    for info in CANDIDATES:
        name = info["name"]
        remark = info.get("remark", "")

        # 해당 실험 폴더 내의 CSV 파일 찾기
        exp_dir = f"outputs/stage4_rerun/{name}"
        csv_files = glob.glob(os.path.join(exp_dir, "benchmark_*.csv"))

        if not csv_files:
            summary_results.append(
                {
                    "Experiment": name,
                    "Serialization": "kv_pairs",
                    "Retrieval": "-",
                    "Prompt": "raw_stuffing",
                    "QueryRepr": "-",
                    "Accuracy": 0.0,
                    "CorrectTotal": "N/A",
                    "TotalTime": 0.0,
                    "AvgTime": 0.0,
                    "Status": "FAILED",
                    "Remark": remark,
                }
            )
            continue

        # 가장 최신 CSV 파일 로드
        latest_csv = max(csv_files, key=os.path.getmtime)
        try:
            df = pd.read_csv(latest_csv)
            # 데이터 행만 필터링 (SUMMARY 행 제외)
            data_df = df[df["#"] != "SUMMARY"].copy()
            # benchmark.py는 'Correct' 컬럼에 'O' 또는 'X'를 기록함
            data_df["IsCorrect"] = data_df["Correct"].map({"O": 1, "X": 0})

            total_count = len(data_df)
            correct_count = data_df["IsCorrect"].sum()
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            total_time = data_df["Time(s)"].sum()
            avg_time = data_df["Time(s)"].mean()

            # 메인 요약 정보 저장
            # retrieval 정보 추출 (overrides에서 파싱)
            ret_type = (
                "score_threshold"
                if "retrieval=score_threshold" in info["overrides"]
                else "top_k"
            )

            summary_results.append(
                {
                    "Experiment": name,
                    "Serialization": "kv_pairs",
                    "Retrieval": ret_type,
                    "Prompt": "raw_stuffing",
                    "QueryRepr": "question_plus_choices",
                    "Accuracy": f"{accuracy:.2f}",
                    "CorrectTotal": f"{correct_count}/{total_count}",
                    "TotalTime": f"{total_time:.2f}",
                    "AvgTime": f"{avg_time:.3f}",
                    "Status": "COMPLETED",
                    "Remark": remark,
                }
            )

            # 카테고리별 정보 계산
            cat_stats = data_df.groupby("Category")["IsCorrect"].agg(["sum", "count"])

            law_stats = cat_stats.loc["Law"] if "Law" in cat_stats.index else None
            crim_stats = (
                cat_stats.loc["Criminal Law"]
                if "Criminal Law" in cat_stats.index
                else None
            )

            cat_info = {"Experiment": name, "Remark": remark}
            if law_stats is not None:
                cat_info["LawCorrect"] = (
                    f"{int(law_stats['sum'])}/{int(law_stats['count'])}"
                )
                cat_info["LawAcc"] = (
                    f"{(law_stats['sum'] / law_stats['count'] * 100):.2f}"
                )
            else:
                cat_info["LawCorrect"], cat_info["LawAcc"] = "-", "-"

            if crim_stats is not None:
                cat_info["CrimCorrect"] = (
                    f"{int(crim_stats['sum'])}/{int(crim_stats['count'])}"
                )
                cat_info["CrimAcc"] = (
                    f"{(crim_stats['sum'] / crim_stats['count'] * 100):.2f}"
                )
            else:
                cat_info["CrimCorrect"], cat_info["CrimAcc"] = "-", "-"

            category_results.append(cat_info)

        except Exception as e:
            print(f"[!] Error processing {latest_csv}: {e}")
            continue

    # --- 테이블 1 출력 ---
    print("\n#### 4차-1 실험 기록표\n")
    print(
        "| 실험명 | serialization | retrieval | prompt | query_repr | accuracy(%) | correct/total | total_time(s) | avg_time(s/q) | status | 비고 |"
    )
    print("| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |")
    for r in summary_results:
        print(
            f"| {r['Experiment']} | {r['Serialization']} | {r['Retrieval']} | {r['Prompt']} | {r['QueryRepr']} | {r['Accuracy']} | {r['CorrectTotal']} | {r['TotalTime']} | {r['AvgTime']} | {r['Status']} | {r['Remark']} |"
        )

    # --- 테이블 2 출력 ---
    print("\n#### 4차-1 Category별 정확도 기록표\n")
    print(
        "| 실험명 | Law correct/total | Law accuracy(%) | Criminal Law correct/total | Criminal Law accuracy(%) | 비고 |"
    )
    print("| --- | --- | ---: | --- | ---: | --- |")
    for r in category_results:
        print(
            f"| {r['Experiment']} | {r['LawCorrect']} | {r['LawAcc']} | {r['CrimCorrect']} | {r['CrimAcc']} | {r['Remark']} |"
        )


if __name__ == "__main__":
    success_count = 0
    for cand in CANDIDATES:
        if run_experiment(cand):
            success_count += 1

    print(f"\n>>> Completed {success_count}/{len(CANDIDATES)} experiments.")
    aggregate_results()
