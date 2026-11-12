"""emg2pose 손 자세 회귀 베이스라인: 1 초 창의 채널별 log RMS -> 관절 각도 릿지 회귀.

실행: uv run python -m semg_bench.emg2pose_baseline --data $DATA_ROOT/emg2pose --users all --shards all --seed 0
결과: results/emg2pose_baseline_joint_mae.csv, _split.csv, _joint_mae.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from semg_bench.data_scope import data_root, parse_id_list
from semg_bench.emg2pose_loader import list_recordings, load_recording, make_windows

RESULTS_DIR = Path("results")
EPS = 1e-8
ALPHAS = np.logspace(-3, 3, 13)
MIN_VALID = 0.9  # 창 안에서 관절 각도가 유효(IK 성공)한 샘플 비율의 하한


def assign_split(meta: pd.DataFrame, seed: int, val_frac: float = 0.2) -> pd.Series:
    """CSV split 에 train 과 평가 분할이 같이 있으면 그대로, 아니면 stage 단위로 val 을 떼어 낸다."""
    splits = set(meta["split"])
    if "train" in splits and splits - {"train"}:
        return meta["split"].copy()
    stages = np.sort(meta["stage"].unique())
    n_val = max(1, round(val_frac * len(stages)))
    val_stages = np.random.default_rng(seed).choice(stages, size=n_val, replace=False)
    return pd.Series(np.where(meta["stage"].isin(val_stages), "val", "train"), index=meta.index)


def window_features(
    emg_w: np.ndarray, ang_w: np.ndarray, min_valid: float = MIN_VALID
) -> tuple[np.ndarray, np.ndarray]:
    """창별 채널 log RMS 특징과 목표(유효 샘플만의 평균 관절 각도)를 만든다.

    관절 각도가 하나라도 NaN 인 샘플은 무효로 보고, 유효 샘플 비율이 min_valid 미만인 창은 버린다.
    """
    valid = np.isfinite(ang_w).all(axis=2)  # (창, 창 길이)
    keep = valid.mean(axis=1) >= min_valid
    rms = np.sqrt(np.mean(np.square(emg_w[keep], dtype=np.float64), axis=1))
    masked = np.where(valid[keep][:, :, None], ang_w[keep], 0.0).astype(np.float64)
    target = masked.sum(axis=1) / valid[keep].sum(axis=1, keepdims=True)
    return np.log(rms + EPS), target


def recording_features(
    path: Path | str, member: str | None, win_s: float, min_valid: float = MIN_VALID
) -> tuple[np.ndarray, np.ndarray]:
    """녹화 하나를 읽어(IK 실패 구간은 NaN) 창 특징과 목표를 만든다. 원신호는 함수가 끝나면 버려진다."""
    rec = load_recording(path, member)
    return window_features(*make_windows(rec.emg, rec.angles, rec.fs, win_s), min_valid=min_valid)


def build_dataset(
    recs: pd.DataFrame, win_s: float, min_valid: float = MIN_VALID
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """녹화마다 창 특징을 만들어 특징, 목표, 녹화 번호를 쌓는다."""
    feats, targets, rec_ids = [], [], []
    for i, row in enumerate(recs.itertuples()):
        x, y = recording_features(row.path, row.member, win_s, min_valid)
        feats.append(x)
        targets.append(y)
        rec_ids.append(np.full(len(y), i))
    return np.concatenate(feats), np.concatenate(targets), np.concatenate(rec_ids)


def fit_ridge(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """표준화 + RidgeCV(alpha 는 학습 데이터 안에서 LOO 로 고른다)."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(x, y)


def select_recordings(recs: pd.DataFrame, users: list[str] | None, shards: list[str] | None) -> pd.DataFrame:
    """피험자(user) 목록과 샤드(tar 파일 이름 또는 stem) 목록으로 녹화를 거른다. None 은 전체."""
    keep = pd.Series(True, index=recs.index)
    if users is not None:
        keep &= recs["user"].astype(str).isin(users)
    if shards is not None:
        tar = recs["path"].map(lambda p: Path(p).name)
        keep &= recs["member"].notna() & (tar.isin(shards) | tar.str.removesuffix(".tar").isin(shards))
    return recs[keep].reset_index(drop=True)


def evaluate(
    x: np.ndarray, y: np.ndarray, is_train: np.ndarray, eval_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """train 으로 릿지를 학습해 평가 창의 관절별 MAE 와 train 평균 예측기의 MAE, 선택된 alpha 를 돌려준다."""
    model = fit_ridge(x[is_train], y[is_train])
    mae = np.abs(model.predict(x[eval_mask]) - y[eval_mask]).mean(axis=0)
    mean_mae = np.abs(y[is_train].mean(axis=0) - y[eval_mask]).mean(axis=0)
    return mae, mean_mae, float(model[-1].alpha_)


def save_mae_plot(table: pd.DataFrame, split: str, out: Path) -> None:
    """관절별 MAE 가로 막대(릿지)와 평균 예측기 기준선(회색 눈금)을 그린다."""
    t = table[table["split"] == split].sort_values("mae_deg")
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(t["joint"], t["mae_deg"], height=0.6, color="#2a78d6", label="ridge (log RMS)")
    ax.scatter(t["mean_baseline_mae_deg"], t["joint"], marker="|", s=120, linewidths=2, color="#52514e",
               label="predict train mean", zorder=3)
    ax.set_xlabel("MAE (degrees)", color="#52514e")
    ax.set_title(f"emg2pose per-joint MAE ({split}, 1 s windows)", color="#0b0b0b", loc="left")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"emg2pose_baseline_joint_mae_{split}.png", dpi=150)
    plt.close(fig)


def add_scope_args(p: argparse.ArgumentParser) -> None:
    """emg2pose 범위 인자: 피험자(user)·샤드 목록, 창 길이, seed."""
    p.add_argument("--users", default="all", help="user ID 목록, 쉼표 구분 (기본 all)")
    p.add_argument("--shards", default="all", help="샤드 tar 이름 목록, 쉼표 구분 (기본 all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--win-s", type=float, default=1.0)
    p.add_argument("--min-valid", type=float, default=MIN_VALID, help="창 안 IK 유효 샘플 비율 하한")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=data_root() / "emg2pose", help="emg2pose 루트 (기본: $DATA_ROOT/emg2pose)")
    add_scope_args(p)
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def joint_table(results: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> pd.DataFrame:
    """평가 분할별 관절 MAE 를 rad·deg 로 정리한다. 관절 이름은 HDF5 에 없어 번호로 둔다."""
    rows = [
        {"split": split, "joint": f"joint_{j:02d}", "mae_rad": m, "mae_deg": np.degrees(m),
         "mean_baseline_mae_rad": b, "mean_baseline_mae_deg": np.degrees(b), "alpha": alpha}
        for split, (mae, base, alpha) in results.items()
        for j, (m, b) in enumerate(zip(mae, base, strict=True))
    ]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    recs = select_recordings(list_recordings(args.data), parse_id_list(args.users), parse_id_list(args.shards))
    if recs.empty:
        raise SystemExit(f"범위(users={args.users}, shards={args.shards})에 맞는 녹화가 없다")
    recs["eval_split"] = assign_split(recs, args.seed)
    x, y, rec_ids = build_dataset(recs, args.win_s, args.min_valid)
    win_split = recs["eval_split"].to_numpy()[rec_ids]
    is_train = win_split == "train"
    results = {s: evaluate(x, y, is_train, win_split == s) for s in sorted(set(win_split) - {"train"})}
    table = joint_table(results)
    args.out.mkdir(parents=True, exist_ok=True)
    table.assign(seed=args.seed, win_s=args.win_s).to_csv(args.out / "emg2pose_baseline_joint_mae.csv", index=False)
    recs[["filename", "stage", "side", "split", "eval_split"]].to_csv(
        args.out / "emg2pose_baseline_split.csv", index=False
    )
    for split in results:
        save_mae_plot(table, split, args.out)
    print(recs.groupby("eval_split").agg(files=("filename", "size"), stages=("stage", "nunique")).to_string())
    print(f"창: train {is_train.sum()}, 평가 {(~is_train).sum()}, 특징 {x.shape[1]}, 관절 {y.shape[1]}")
    print(table.groupby("split")[["mae_deg", "mean_baseline_mae_deg", "alpha"]].mean().round(2).to_string())
    print(f"저장: {args.out}/emg2pose_baseline_*.csv, *.png")


if __name__ == "__main__":
    main()
