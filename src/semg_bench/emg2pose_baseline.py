"""emg2pose 손 자세 회귀 베이스라인: 1 초 창(0.25 초 간격)의 채널별 log RMS·log WL -> 관절 각도, 손별 릿지 회귀.

설정 근거(미니 데이터, stage 분할 seed 0-2): WL 추가와 창 겹침이 평균 예측 대비 개선폭을 1.85 -> 2.86 도로,
손별 모델이 3.27 도로 늘렸다. 대역 파워(+0.04 도)와 녹화별 정규화(개선 소멸)는 채택하지 않았다.

실행: uv run python -m semg_bench.emg2pose_baseline --data $DATA_ROOT/emg2pose --users all --shards all --seed 0
결과: results/emg2pose_baseline_joint_mae.csv, _split.csv, _joint_mae.png
"""

import argparse
from pathlib import Path
from typing import NamedTuple

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
FEATURE_FUNCS = {
    "rms": lambda w: np.log(np.sqrt(np.mean(np.square(w), axis=1)) + EPS),
    "wl": lambda w: np.log(np.mean(np.abs(np.diff(w, axis=1)), axis=1) + EPS),  # 샘플당 파형 길이
}


class FeatureConfig(NamedTuple):
    """창·특징 설정. tag 는 특징 캐시 디렉터리 이름에 쓰여 설정이 다른 예전 특징이 섞이지 않게 한다."""

    win_s: float = 1.0
    step_s: float = 0.25
    min_valid: float = MIN_VALID
    features: tuple[str, ...] = ("rms", "wl")

    @property
    def tag(self) -> str:
        return f"win{self.win_s:g}_step{self.step_s:g}_valid{self.min_valid:g}_{'+'.join(self.features)}"


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
    emg_w: np.ndarray, ang_w: np.ndarray, min_valid: float = MIN_VALID, features: tuple[str, ...] = ("rms", "wl")
) -> tuple[np.ndarray, np.ndarray]:
    """창별 채널 특징(features 순서로 채널 블록을 이어 붙임)과 목표(유효 샘플만의 평균 관절 각도)를 만든다.

    관절 각도가 하나라도 NaN 인 샘플은 무효로 보고, 유효 샘플 비율이 min_valid 미만인 창은 버린다.
    """
    valid = np.isfinite(ang_w).all(axis=2)  # (창, 창 길이)
    keep = valid.mean(axis=1) >= min_valid
    emg = emg_w[keep].astype(np.float64)
    x = np.hstack([FEATURE_FUNCS[f](emg) for f in features]) if len(emg) else np.empty((0, 0))
    masked = np.where(valid[keep][:, :, None], ang_w[keep], 0.0).astype(np.float64)
    target = masked.sum(axis=1) / valid[keep].sum(axis=1, keepdims=True)
    return x.reshape(len(target), -1), target


def recording_features(path: Path | str, member: str | None, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """녹화 하나를 읽어(IK 실패 구간은 NaN) 창 특징과 목표를 만든다. 원신호는 함수가 끝나면 버려진다."""
    rec = load_recording(path, member)
    windows = make_windows(rec.emg, rec.angles, rec.fs, cfg.win_s, cfg.step_s)
    return window_features(*windows, min_valid=cfg.min_valid, features=cfg.features)


def build_dataset(recs: pd.DataFrame, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """녹화마다 창 특징을 만들어 특징, 목표, 녹화 번호를 쌓는다."""
    feats, targets, rec_ids = [], [], []
    for i, row in enumerate(recs.itertuples()):
        x, y = recording_features(row.path, row.member, cfg)
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


def fit_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray, g_tr: np.ndarray | None = None, g_ev: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """그룹(예: 손)별로 릿지와 평균 예측기를 학습해 평가 창을 예측한다. 그룹이 없으면 하나로 본다.

    반환: 릿지 예측, 같은 그룹의 train 평균 예측(공정한 기준선), 그룹별 alpha 평균.
    """
    g_tr = np.zeros(len(x_tr)) if g_tr is None else g_tr
    g_ev = np.zeros(len(x_ev)) if g_ev is None else g_ev
    pred, mean_pred, alphas = np.empty((len(x_ev), y_tr.shape[1])), np.empty((len(x_ev), y_tr.shape[1])), []
    for g in np.unique(g_ev):
        tr, ev = g_tr == g, g_ev == g
        if not tr.any():
            raise ValueError(f"그룹 {g} 의 학습 창이 없다")
        model = fit_ridge(x_tr[tr], y_tr[tr])
        pred[ev], mean_pred[ev] = model.predict(x_ev[ev]), y_tr[tr].mean(axis=0)
        alphas.append(model[-1].alpha_)
    return pred, mean_pred, float(np.mean(alphas))


def evaluate(
    x: np.ndarray, y: np.ndarray, is_train: np.ndarray, eval_mask: np.ndarray, groups: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """train 으로 (그룹별) 릿지를 학습해 평가 창의 관절별 MAE, 평균 예측기의 MAE, alpha 를 돌려준다."""
    g_tr, g_ev = (None, None) if groups is None else (groups[is_train], groups[eval_mask])
    pred, mean_pred, alpha = fit_predict(x[is_train], y[is_train], x[eval_mask], g_tr, g_ev)
    return np.abs(pred - y[eval_mask]).mean(axis=0), np.abs(mean_pred - y[eval_mask]).mean(axis=0), alpha


def save_mae_plot(table: pd.DataFrame, split: str, out: Path, label: str) -> None:
    """관절별 MAE 가로 막대(릿지)와 평균 예측기 기준선(회색 눈금)을 그린다."""
    t = table[table["split"] == split].sort_values("mae_deg")
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(t["joint"], t["mae_deg"], height=0.6, color="#2a78d6", label=label)
    ax.scatter(t["mean_baseline_mae_deg"], t["joint"], marker="|", s=120, linewidths=2, color="#52514e",
               label="predict train mean", zorder=3)
    ax.set_xlabel("MAE (degrees)", color="#52514e")
    ax.set_title(f"emg2pose per-joint MAE ({split})", color="#0b0b0b", loc="left")
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
    """emg2pose 범위·설정 인자: 피험자(user)·샤드 목록, seed, 창·특징 설정, 손별 모델 여부."""
    p.add_argument("--users", default="all", help="user ID 목록, 쉼표 구분 (기본 all)")
    p.add_argument("--shards", default="all", help="샤드 tar 이름 목록, 쉼표 구분 (기본 all)")
    p.add_argument("--seed", type=int, default=0)
    d = FeatureConfig()
    p.add_argument("--win-s", type=float, default=d.win_s)
    p.add_argument("--step-s", type=float, default=d.step_s, help="창 간격(초)")
    p.add_argument("--min-valid", type=float, default=d.min_valid, help="창 안 IK 유효 샘플 비율 하한")
    p.add_argument("--features", default=",".join(d.features), help=f"쉼표 구분, 가능: {','.join(FEATURE_FUNCS)}")
    p.add_argument("--pooled", action="store_true", help="손(side) 구분 없이 모델 하나로 학습")


def feature_config(args: argparse.Namespace) -> FeatureConfig:
    features = tuple(f.strip() for f in args.features.split(",") if f.strip())
    unknown = set(features) - set(FEATURE_FUNCS)
    if unknown or not features:
        raise SystemExit(f"알 수 없는 특징 {sorted(unknown)}. 가능: {list(FEATURE_FUNCS)}")
    return FeatureConfig(args.win_s, args.step_s, args.min_valid, features)


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
    cfg = feature_config(args)
    recs["eval_split"] = assign_split(recs, args.seed)
    x, y, rec_ids = build_dataset(recs, cfg)
    win_split = recs["eval_split"].to_numpy()[rec_ids]
    groups = None if args.pooled else recs["side"].to_numpy()[rec_ids]
    is_train = win_split == "train"
    results = {s: evaluate(x, y, is_train, win_split == s, groups) for s in sorted(set(win_split) - {"train"})}
    table = joint_table(results)
    args.out.mkdir(parents=True, exist_ok=True)
    table.assign(seed=args.seed, config=cfg.tag, per_side=not args.pooled).to_csv(
        args.out / "emg2pose_baseline_joint_mae.csv", index=False
    )
    recs[["filename", "stage", "side", "split", "eval_split"]].to_csv(
        args.out / "emg2pose_baseline_split.csv", index=False
    )
    for split in results:
        save_mae_plot(table, split, args.out, f"ridge ({'+'.join(cfg.features)}{'' if args.pooled else ', per side'})")
    print(recs.groupby("eval_split").agg(files=("filename", "size"), stages=("stage", "nunique")).to_string())
    print(f"설정 {cfg.tag}, 손별 {not args.pooled} | 창: train {is_train.sum()}, 평가 {(~is_train).sum()}, 특징 {x.shape[1]}")
    print(table.groupby("split")[["mae_deg", "mean_baseline_mae_deg", "alpha"]].mean().round(2).to_string())
    print(f"저장: {args.out}/emg2pose_baseline_*.csv, *.png")


if __name__ == "__main__":
    main()
