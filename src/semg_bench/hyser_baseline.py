"""HYSER PR 34 개 동작 분류 베이스라인: RMS·WL 특징 + LDA / 로지스틱 회귀.

실행: uv run python -m semg_bench.hyser_baseline --data $DATA_ROOT/hyser --subjects 1 --sessions 1 --seed 0
결과: results/hyser_baseline_*.csv, results/hyser_baseline_confusion_*.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from semg_bench.data_scope import (
    data_root,
    find_hyser_sessions,
    parse_id_list,
    select_hyser_sessions,
)
from semg_bench.hyser_loader import load_hyser_session, rms_wl_features

RESULTS_DIR = Path("results")
EPS = 1e-8


def window_dataset(x: np.ndarray, y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(시도, 채널, 샘플) 신호를 창 단위 특징 행렬, 창 라벨, 창이 속한 시도 번호로 바꾼다."""
    rms, wl = rms_wl_features(x, fs)  # (시도, 창, 채널)
    feats = np.concatenate([np.log(rms + EPS), np.log(wl + EPS)], axis=-1)
    n_trials, n_windows, n_feats = feats.shape
    trial_ids = np.repeat(np.arange(n_trials), n_windows)
    return feats.reshape(-1, n_feats), np.repeat(y, n_windows), trial_ids


MODEL_NAMES = ("lda", "logreg")


def make_models(seed: int, names: tuple[str, ...] = MODEL_NAMES) -> dict[str, Pipeline]:
    """비교할 분류기. 특징 수(512)가 클래스당 표본보다 많아 LDA 는 shrinkage 를 쓴다.

    학습 시간(겹 하나, 태블릿): LDA 0.3 초, 로지스틱 회귀 약 40 초.
    """
    models = {
        "lda": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
    }
    return {name: models[name]() for name in names}


def trial_folds(y_trial: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """시도 단위로 층화 k 겹 분할한다. 같은 시도의 창은 항상 같은 겹에 들어간다."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y_trial)), y_trial))


def trial_predictions(proba: np.ndarray, classes: np.ndarray, trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """창 확률을 시도별로 평균해 시도 단위 예측을 만든다."""
    trials = np.unique(trial_ids)
    mean_proba = np.stack([proba[trial_ids == t].mean(axis=0) for t in trials])
    return trials, classes[mean_proba.argmax(axis=1)]


def run_cv(
    feats: np.ndarray, y_win: np.ndarray, trial_ids: np.ndarray, y_trial: np.ndarray, seed: int, n_splits: int = 5,
    models: tuple[str, ...] = MODEL_NAMES,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """모델별 시도 단위 교차검증. 겹별 정확도 표와 창 단위 혼동행렬(겹 합계)을 돌려준다."""
    labels = np.unique(y_trial)
    rows, cms = [], {name: np.zeros((len(labels), len(labels)), dtype=np.int64) for name in models}
    for fold, (train_t, test_t) in enumerate(trial_folds(y_trial, n_splits, seed)):
        train, test = np.isin(trial_ids, train_t), np.isin(trial_ids, test_t)
        for name, model in make_models(seed, models).items():
            model.fit(feats[train], y_win[train])
            proba = model.predict_proba(feats[test])
            pred = model.classes_[proba.argmax(axis=1)]
            trials, trial_pred = trial_predictions(proba, model.classes_, trial_ids[test])
            cms[name] += confusion_matrix(y_win[test], pred, labels=labels)
            rows.append({
                "model": name, "fold": fold, "n_test_trials": len(test_t),
                "window_acc": float(np.mean(pred == y_win[test])),
                "trial_acc": float(np.mean(trial_pred == y_trial[trials])),
            })
    return pd.DataFrame(rows), cms


def save_confusion(cm: np.ndarray, labels: np.ndarray, name: str, out_dir: Path) -> None:
    """혼동행렬을 개수 CSV 와 행 정규화 PNG 로 저장한다."""
    pd.DataFrame(cm, index=labels, columns=labels).rename_axis("true\\pred").to_csv(
        out_dir / f"hyser_baseline_confusion_{name}.csv"
    )
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xlabel="predicted", ylabel="true",
           title=f"HYSER 34 gestures - {name} (window-level, row-normalized)")
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"hyser_baseline_confusion_{name}.png", dpi=150)
    plt.close(fig)


def run_session(
    session_dir: Path, task: str, seed: int, models: tuple[str, ...] = MODEL_NAMES
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """세션 하나를 읽어 시도 단위 5 겹 CV 를 돌린다. 반환: 겹별 정확도, 혼동행렬, 라벨 목록."""
    session = load_hyser_session(session_dir, task=task)
    y_trial = session.y
    feats, y_win, trial_ids = window_dataset(session.x, y_trial, session.fs)
    del session  # 원신호(수백 MB)는 특징 추출 뒤 버린다
    print(f"{session_dir.name}/{task}: 시도 {len(y_trial)}, 창 {len(y_win)}, 특징 {feats.shape[1]}")
    acc, cms = run_cv(feats, y_win, trial_ids, y_trial, seed, models=models)
    return acc.assign(session=session_dir.name, task=task, seed=seed), cms, np.unique(y_trial)


def add_scope_args(p: argparse.ArgumentParser) -> None:
    """HYSER 범위 인자: 데이터 루트, 피험자·세션 목록, 과제, seed."""
    p.add_argument("--data", type=Path, default=data_root() / "hyser", help="HYSER 루트 (기본: $DATA_ROOT/hyser)")
    p.add_argument("--subjects", default="all", help="피험자 번호 목록, 예: 1-5,8 (기본 all)")
    p.add_argument("--sessions", default="all", help="세션 번호 목록, 예: 1 또는 1,2 (기본 all)")
    p.add_argument("--task", default="dynamic", choices=["dynamic", "maintenance"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", default=",".join(MODEL_NAMES), help="쉼표 구분: lda,logreg (Colab 전체 실행은 lda 만 권장)")


def model_names(args: argparse.Namespace) -> tuple[str, ...]:
    names = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if not names or set(names) - set(MODEL_NAMES):
        raise SystemExit(f"--models 는 {MODEL_NAMES} 중에서 고른다: {args.models}")
    return names


def scoped_sessions(args: argparse.Namespace) -> pd.DataFrame:
    """인자로 받은 범위의 HYSER 세션 목록. 하나도 없으면 오류."""
    found = find_hyser_sessions(args.data)
    sel = select_hyser_sessions(found, parse_id_list(args.subjects), parse_id_list(args.sessions))
    if sel.empty:
        raise SystemExit(f"{args.data} 에서 범위(subjects={args.subjects}, sessions={args.sessions})에 맞는 세션이 없다")
    return sel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_scope_args(p)
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    accs, cms, labels = [], {}, None
    for row in scoped_sessions(args).itertuples():
        acc, session_cms, labels = run_session(row.path, args.task, args.seed, model_names(args))
        accs.append(acc)
        for name, cm in session_cms.items():
            cms[name] = cms.get(name, 0) + cm
    acc = pd.concat(accs, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    acc.to_csv(args.out / "hyser_baseline_accuracy.csv", index=False)
    summary = acc.groupby("model")[["window_acc", "trial_acc"]].agg(["mean", "std"])
    summary.to_csv(args.out / "hyser_baseline_summary.csv")
    for name, cm in cms.items():
        save_confusion(cm, labels, name, args.out)
    print(summary.round(3).to_string())
    print(f"세션 {acc['session'].nunique()}개, 저장: {args.out}/hyser_baseline_*.csv, *.png")


if __name__ == "__main__":
    main()
