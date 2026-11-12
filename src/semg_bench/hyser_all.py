"""HYSER PR 전체(또는 범위) 세션을 하나씩 돌려 세션별 CSV 로 저장한다. 이미 끝난 세션은 건너뛴다.

진입점: scripts/run_hyser_all.py
결과: results/hyser/accuracy/<세션>.csv, results/hyser/confusion/<세션>_<모델>.csv,
      results/hyser/summary_by_session.csv, results/hyser/summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from semg_bench.data_scope import write_csv_atomic
from semg_bench.hyser_baseline import add_scope_args, run_session, scoped_sessions


def accuracy_path(out: Path, name: str, task: str) -> Path:
    return out / "accuracy" / f"{name}_{task}.csv"


def save_session(out: Path, name: str, acc: pd.DataFrame, cms: dict[str, np.ndarray], labels: np.ndarray) -> None:
    """혼동행렬을 먼저, 정확도 CSV 를 마지막에 쓴다. 정확도 CSV 가 있으면 그 세션은 완료다."""
    task = acc["task"].iloc[0]
    for model, cm in cms.items():
        df = pd.DataFrame(cm, index=labels, columns=labels).rename_axis("true").reset_index()
        write_csv_atomic(df, out / "confusion" / f"{name}_{task}_{model}.csv")
    write_csv_atomic(acc, accuracy_path(out, name, task))


def summarize(out: Path, names: list[str], task: str) -> pd.DataFrame:
    """범위 안에서 완료된 세션 CSV 를 모아 세션별·전체 요약을 저장하고 돌려준다."""
    paths = [p for n in names if (p := accuracy_path(out, n, task)).exists()]
    if not paths:
        return pd.DataFrame()
    acc = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    by_session = acc.groupby(["session", "model"], as_index=False)[["window_acc", "trial_acc"]].mean()
    write_csv_atomic(by_session, out / "summary_by_session.csv")
    summary = by_session.groupby("model")[["window_acc", "trial_acc"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    write_csv_atomic(summary.reset_index(), out / "summary.csv")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_scope_args(p)
    p.add_argument("--out", type=Path, default=Path("results") / "hyser")
    p.add_argument("--redo", action="store_true", help="완료된 세션도 다시 돌린다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    sessions = scoped_sessions(args)
    failed: list[str] = []
    for i, row in enumerate(sessions.itertuples(), 1):
        if accuracy_path(args.out, row.name, args.task).exists() and not args.redo:
            print(f"[{i}/{len(sessions)}] {row.name}: 완료됨, 건너뜀")
            continue
        try:
            acc, cms, labels = run_session(row.path, args.task, args.seed)
        except (OSError, ValueError) as e:  # 손상·누락 파일은 기록하고 다음 세션으로
            print(f"[{i}/{len(sessions)}] {row.name}: 실패 - {e}")
            failed.append(row.name)
            continue
        save_session(args.out, row.name, acc, cms, labels)
        print(f"[{i}/{len(sessions)}] {row.name}: trial_acc {acc.groupby('model')['trial_acc'].mean().round(3).to_dict()}")
    summary = summarize(args.out, sessions["name"].tolist(), args.task)
    print(summary.round(3).to_string() if not summary.empty else "완료된 세션 없음")
    if failed:
        print(f"실패 {len(failed)}개: {failed}")
    return 1 if failed else 0
