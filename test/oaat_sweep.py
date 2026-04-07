"""
oaat_sweep.py — One-At-A-Time (OAAT) Baseline + 단일 축 변경 실험 자동화

목적:
  - baseline 1회 실행
  - 각 실험 축(serialization / retrieval / prompt)을 한 번에 하나씩 변경하여 실험
  - 결과를 outputs/oaat_sweep/{timestamp}/ 에 요약 CSV + JSON 및 개별 실험 로그로 저장

실행 방법 (프로젝트 루트에서):
  python test/oaat_sweep.py                        # stage1_oaat (기본)
  python test/oaat_sweep.py --mode baseline_only   # baseline만
  python test/oaat_sweep.py --mode stage1_oaat     # baseline + 축별 1차 실험
  python test/oaat_sweep.py --yes                  # 확인 프롬프트 스킵
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime

# ─── Baseline 정의 (config.yaml defaults와 일치) ──────────────
BASELINE: dict[str, str] = {
    "serialization": "kv_pairs",
    "retrieval": "top_k",
    "prompt": "raw_stuffing",
}

# ─── 축별 후보 (yaml 파일명 = Hydra group value) ──────────────
AXES: dict[str, list[str]] = {
    "serialization": ["raw", "kv_pairs", "narrative", "weighted", "dual", "synthetic"],
    "retrieval": ["top_k", "score_threshold", "mmr", "hybrid"],
    "prompt": [
        "raw_stuffing",
        "compress_summarize",
        "few_shot_envelope",
        "labeled_context",
        "structured_context",
    ],
}

# ─── 결과 CSV 필드 순서 ──────────────────────────────────────
_FIELDS = [
    "name",
    "axis",
    "serialization",
    "retrieval",
    "prompt",
    "accuracy_pct",
    "correct",
    "total",
    "total_time_s",
    "avg_time_s",
    "status",
]


def build_run_list(mode: str) -> list[dict]:
    """실행 목록 생성. baseline은 항상 첫 번째."""
    runs: list[dict] = []

    runs.append({"name": "baseline", "axis": "baseline", **BASELINE})

    if mode == "baseline_only":
        return runs

    # stage1_oaat: 각 축을 하나씩 바꾸고 나머지는 baseline 고정
    for axis, variants in AXES.items():
        for variant in variants:
            if BASELINE[axis] == variant:
                continue  # baseline과 동일 → 스킵
            cfg = {**BASELINE, axis: variant}
            runs.append(
                {
                    "name": f"{axis}__{variant}",
                    "axis": axis,
                    **cfg,
                }
            )

    return runs


def run_experiment(run_cfg: dict, hydra_root: str) -> dict:
    """
    단일 실험을 subprocess로 실행하고 결과를 파싱하여 반환.

    benchmark.py는 Hydra를 통해 실행되며,
    hydra.run.dir 오버라이드로 각 실험의 출력 경로를 분리합니다.
    """
    name = run_cfg["name"]
    folder_name = (
        f"serial={run_cfg['serialization']}"
        f"__ret={run_cfg['retrieval']}"
        f"__prompt={run_cfg['prompt']}"
    )
    run_dir = os.path.join(hydra_root, folder_name)

    cmd = [
        sys.executable,
        "test/benchmark.py",
        f"serialization={run_cfg['serialization']}",
        f"retrieval={run_cfg['retrieval']}",
        f"prompt={run_cfg['prompt']}",
        f"experiment_name={name}",
        f"hydra.run.dir={run_dir}",
    ]

    print(f"\n{'─' * 60}")
    print(f"[RUN] {name}")
    print(
        f"      serial={run_cfg['serialization']}"
        f"  retrieval={run_cfg['retrieval']}"
        f"  prompt={run_cfg['prompt']}"
    )
    print(f"{'─' * 60}")

    result: dict = {
        "name": name,
        "axis": run_cfg["axis"],
        "serialization": run_cfg["serialization"],
        "retrieval": run_cfg["retrieval"],
        "prompt": run_cfg["prompt"],
        "accuracy_pct": None,
        "correct": None,
        "total": None,
        "total_time_s": None,
        "avg_time_s": None,
        "status": "pending",
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        combined = proc.stdout + proc.stderr

        # "[✔] 최종 정확도: 72.50%  (29/40)"
        acc_m = re.search(r"최종 정확도:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", combined)
        if acc_m:
            result["accuracy_pct"] = float(acc_m.group(1))
            result["correct"] = int(acc_m.group(2))
            result["total"] = int(acc_m.group(3))

        # "[✔] 전체 소요 시간: 42.31s  |  평균 1.058s/문항"
        time_m = re.search(r"전체 소요 시간:\s*([\d.]+)s", combined)
        if time_m:
            result["total_time_s"] = float(time_m.group(1))

        avg_m = re.search(r"평균\s*([\d.]+)s/문항", combined)
        if avg_m:
            result["avg_time_s"] = float(avg_m.group(1))

        if proc.returncode == 0:
            result["status"] = "ok"
        else:
            result["status"] = f"error(rc={proc.returncode})"

        # ── run.log 저장 (subprocess stdout+stderr 전체) ────
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(combined)

        # ── 진행 상황 출력 ──────────────────────────────────
        if result["accuracy_pct"] is not None:
            print(
                f"[OK]  accuracy={result['accuracy_pct']:.2f}%"
                f"  ({result['correct']}/{result['total']})"
                f"  total={result['total_time_s']:.1f}s"
                f"  avg={result['avg_time_s']:.3f}s/q"
            )
        else:
            print(f"[WARN] 결과 파싱 실패. status={result['status']}")
            if proc.returncode != 0:
                # 마지막 500자만 출력
                tail = combined[-500:].strip()
                print(f"--- STDERR tail ---\n{tail}\n---")

    except Exception as exc:
        result["status"] = f"exception:{exc}"
        print(f"[ERROR] {exc}")

    return result


def print_summary_table(results: list[dict]) -> None:
    """완료 후 콘솔 요약 테이블 출력."""
    # 축 순서: baseline → serialization → retrieval → prompt
    axis_order = {"baseline": 0, "serialization": 1, "retrieval": 2, "prompt": 3}
    sorted_results = sorted(
        results,
        key=lambda r: (axis_order.get(r["axis"], 99), -(r["accuracy_pct"] or -1)),
    )

    header = f"{'축':<15} {'실험명':<35} {'정확도':>9} {'correct':>8} {'time(s)':>9}"
    print(f"\n{'=' * 60}")
    print("OAAT SWEEP 결과 요약")
    print("=" * 60)
    print(header)
    print("─" * 60)
    for r in sorted_results:
        acc = f"{r['accuracy_pct']:.2f}%" if r["accuracy_pct"] is not None else "N/A"
        correct = f"{r['correct']}/{r['total']}" if r["correct"] is not None else "N/A"
        t = f"{r['total_time_s']:.1f}" if r["total_time_s"] is not None else "N/A"
        print(f"{r['axis']:<15} {r['name']:<35} {acc:>9} {correct:>8} {t:>9}")
    print("=" * 60)


def save_summary(results: list[dict], output_dir: str) -> None:
    """결과를 CSV + JSON으로 저장."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"oaat_summary_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, f"oaat_summary_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[SAVED] CSV  → {csv_path}")
    print(f"[SAVED] JSON → {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OAAT Sweep: baseline + 축별 단독 변경 실험 자동 실행"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline_only", "stage1_oaat"],
        default="stage1_oaat",
        help="실험 모드 (기본: stage1_oaat)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/oaat_sweep",
        help="결과 저장 루트 디렉터리 (기본: outputs/oaat_sweep)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="실행 전 확인 프롬프트를 스킵합니다",
    )
    args = parser.parse_args()

    # sweep 실행마다 타임스탬프 폴더로 격리
    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.output_dir, sweep_ts)

    runs = build_run_list(args.mode)

    # ── 실행 계획 출력 ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"OAAT Sweep  |  mode={args.mode}  |  총 {len(runs)}개 실험")
    print(f"출력 경로   |  {sweep_dir}")
    print("=" * 60)
    print(f"{'#':<4} {'실험명':<35} {'축':<15}")
    print(f"{'─' * 4} {'─' * 35} {'─' * 15}")
    for i, r in enumerate(runs, 1):
        print(f"{i:<4} {r['name']:<35} {r['axis']:<15}")

    if not args.yes:
        print("\n실험을 시작합니다. Enter를 누르면 계속, Ctrl+C로 취소합니다...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n취소됨.")
            sys.exit(0)

    # ── 실험 실행 ────────────────────────────────────────────
    results: list[dict] = []
    for run_cfg in runs:
        result = run_experiment(run_cfg, sweep_dir)
        results.append(result)

    # ── 결과 출력 + 저장 ─────────────────────────────────────
    print_summary_table(results)
    save_summary(results, sweep_dir)


if __name__ == "__main__":
    main()
