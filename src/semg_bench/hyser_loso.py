"""HYSER 피험자 간 일반화: leave-one-subject-out(LOSO) 과 채널 정규화·전극 어레이 증강 비교.

1 단계(extract): 세션마다 창 특징(log RMS·log WL)을 어레이 격자 순서로 results/hyser_loso/features/ 에 캐시.
2 단계(eval): 피험자 한 명을 빼고 나머지로 학습해 그 사람으로 평가. 조건(정규화×증강)×피험자마다 CSV 저장.
둘 다 끝난 것은 건너뛰므로 중단 후 같은 명령으로 이어 돌릴 수 있다.

정규화
- none: 그대로
- window: 창마다 특징 종류·어레이별 평균을 뺀다(log 공간). 전극 접촉·피부 임피던스 같은 어레이 전체 이득을 없앤다.
- session: 세션별 열 z-score. 라벨은 쓰지 않지만 평가 세션의 통계를 쓰는 전이적(transductive) 보정이다.
증강(학습 데이터만, 정규화 뒤)
- none: 그대로
- shift: 어레이마다 상하좌우 1 칸 이동(가장자리 복제) 4 사본 + 사본마다 어레이별 log 이득 지터(N(0, 0.2)).
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from semg_bench.data_scope import write_csv_atomic
from semg_bench.hyser_baseline import (
    add_scope_args,
    make_models,
    model_names,
    scoped_sessions,
    trial_predictions,
    window_dataset,
)
from semg_bench.hyser_loader import load_hyser_session

CHANNEL_RE = re.compile(r"^([A-Za-z]+)-(\d+)-(\d+)$")
FEATURE_TYPES = ("rms", "wl")  # window_dataset 의 특징 블록 순서
NORMS = ("none", "window", "session")
AUGS = ("none", "shift")
SHIFTS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GAIN_JITTER = 0.2


def grid_order(channel_names: list[str]) -> tuple[np.ndarray, tuple[int, int, int]]:
    """채널명(어레이-행-열)을 파싱해 (어레이, 행, 열) 순서의 채널 인덱스와 격자 모양을 돌려준다."""
    parsed = []
    for i, name in enumerate(channel_names):
        m = CHANNEL_RE.match(name)
        if not m:
            raise ValueError(f"채널명 '{name}' 이 어레이-행-열 형식이 아니다")
        parsed.append((m.group(1), int(m.group(2)), int(m.group(3)), i))
    arrays = sorted({p[0] for p in parsed})
    rows, cols = sorted({p[1] for p in parsed}), sorted({p[2] for p in parsed})
    if len(parsed) != len(arrays) * len(rows) * len(cols):
        raise ValueError("채널이 완전한 어레이 격자를 이루지 않는다")
    order = [i for _, _, _, i in sorted(parsed, key=lambda p: (arrays.index(p[0]), p[1], p[2]))]
    return np.array(order), (len(arrays), len(rows), len(cols))


def feature_path(out: Path, name: str, task: str) -> Path:
    return out / "features" / f"{name}_{task}.csv.gz"


def session_features(session_dir: Path, task: str) -> pd.DataFrame:
    """세션 하나의 창 특징을 격자 순서 열(rms_ED-1-1 ...)로, 라벨·시도 번호와 함께 표로 만든다."""
    s = load_hyser_session(session_dir, task=task)
    feats, y_win, trial_ids = window_dataset(s.x, s.y, s.fs)
    order, _ = grid_order(s.channel_names)
    n_ch = len(s.channel_names)
    cols, blocks = [], []
    for k, ftype in enumerate(FEATURE_TYPES):
        blocks.append(feats[:, k * n_ch : (k + 1) * n_ch][:, order])
        cols += [f"{ftype}_{s.channel_names[i]}" for i in order]
    df = pd.DataFrame(np.hstack(blocks).astype(np.float32), columns=cols)
    return df.assign(label=y_win, trial=trial_ids)


def extract_all(sessions: pd.DataFrame, out: Path, task: str) -> list[str]:
    """특징 캐시가 없는 세션만 추출한다. 실패한 세션 이름을 돌려준다."""
    failed = []
    for i, row in enumerate(sessions.itertuples(), 1):
        path = feature_path(out, row.name, task)
        if path.exists():
            continue
        try:
            write_csv_atomic(session_features(row.path, task), path)
            print(f"[특징 {i}/{len(sessions)}] {row.name}")
        except (OSError, ValueError) as e:
            print(f"[특징 {i}/{len(sessions)}] {row.name}: 실패 - {e}")
            failed.append(row.name)
    return failed


def load_all(sessions: pd.DataFrame, out: Path, task: str) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int, int]]:
    """캐시된 특징을 모아 (특징 행렬, 메타 표[subject, session, label, trial_key], 격자 모양)을 만든다."""
    xs, metas, fcols = [], [], None
    for row in sessions.itertuples():
        if not feature_path(out, row.name, task).exists():
            continue
        df = pd.read_csv(feature_path(out, row.name, task))
        cols = [c for c in df.columns if c.split("_", 1)[0] in FEATURE_TYPES]
        if fcols is not None and cols != fcols:
            raise ValueError(f"{row.name}: 특징 열(채널 구성)이 다른 세션과 다르다")
        fcols = cols
        xs.append(df[cols].to_numpy(np.float32))
        metas.append(pd.DataFrame({"subject": row.subject, "session": row.name, "label": df["label"],
                                   "trial_key": row.name + ":" + df["trial"].astype(str)}))
    if not xs:
        raise SystemExit("특징 캐시가 없다. 먼저 --stage extract 를 돌려라")
    _, shape = grid_order([c.split("_", 1)[1] for c in fcols if c.startswith(FEATURE_TYPES[0] + "_")])
    return np.vstack(xs), pd.concat(metas, ignore_index=True), shape


def normalize(x: np.ndarray, mode: str, groups: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """정규화 모드를 적용한다. groups 는 session 모드에서 통계를 따로 낼 단위(세션)."""
    if mode == "none":
        return x
    if mode == "window":
        g = x.reshape(len(x), len(FEATURE_TYPES), shape[0], -1)
        return (g - g.mean(axis=3, keepdims=True)).reshape(len(x), -1)
    if mode == "session":
        out = np.empty_like(x)
        for g in np.unique(groups):
            m = groups == g
            out[m] = (x[m] - x[m].mean(axis=0)) / (x[m].std(axis=0) + 1e-6)
        return out
    raise ValueError(f"알 수 없는 정규화 {mode}")


def shift_grid(g: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """(..., 행, 열) 격자를 가장자리 복제로 (dr, dc) 만큼 민다. 전극 어레이가 피부 위에서 밀린 상황."""
    rows, cols = g.shape[-2:]
    pad = np.pad(g, [(0, 0)] * (g.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    return pad[..., 1 - dr : 1 - dr + rows, 1 - dc : 1 - dc + cols]


def augment(
    x: np.ndarray, y: np.ndarray, shape: tuple[int, int, int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """원본 + 이동 사본 4 개(사본마다 어레이별 log 이득 지터)를 돌려준다."""
    g = x.reshape(len(x), len(FEATURE_TYPES), *shape)
    copies = [x]
    for dr, dc in SHIFTS:
        jitter = rng.normal(0, GAIN_JITTER, size=(len(x), 1, shape[0], 1, 1)).astype(x.dtype)
        copies.append((shift_grid(g, dr, dc) + jitter).reshape(len(x), -1))
    return np.vstack(copies), np.tile(y, len(copies))


def loso_fold(
    xn: np.ndarray, meta: pd.DataFrame, shape: tuple[int, int, int], subject: int, norm: str, aug: str,
    seed: int, models: tuple[str, ...],
) -> pd.DataFrame:
    """정규화된 특징 xn 에서 피험자 subject 를 평가로, 나머지를 학습으로 한 겹. 모델별 정확도 행을 돌려준다."""
    test = (meta["subject"] == subject).to_numpy()
    y = meta["label"].to_numpy()
    x_tr, y_tr = xn[~test], y[~test]
    if aug == "shift":
        x_tr, y_tr = augment(x_tr, y_tr, shape, np.random.default_rng(seed + subject))
    rows = []
    for name, model in make_models(seed, models).items():
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(xn[test])
        pred = model.classes_[proba.argmax(axis=1)]
        trials, trial_pred = trial_predictions(proba, model.classes_, meta.loc[test, "trial_key"].to_numpy())
        trial_true = meta[test].groupby("trial_key")["label"].first().loc[trials].to_numpy()
        rows.append({"subject": subject, "norm": norm, "aug": aug, "model": name, "seed": seed,
                     "window_acc": float(np.mean(pred == y[test])), "trial_acc": float(np.mean(trial_pred == trial_true)),
                     "n_train": len(x_tr), "n_test": int(test.sum())})
    return pd.DataFrame(rows)


def result_path(out: Path, norm: str, aug: str, task: str, subject: int) -> Path:
    return out / "loso" / f"{norm}_{aug}" / f"{task}_subject{subject:02d}.csv"


def run_loso(args: argparse.Namespace, sessions: pd.DataFrame, models: tuple[str, ...]) -> None:
    """조건×피험자 LOSO 를 돌린다. 결과 CSV 에 요청 모델이 다 있으면 건너뛴다."""
    x, meta, shape = load_all(sessions, args.out, args.task)
    subjects = sorted(meta["subject"].unique())
    if len(subjects) < 2:
        raise SystemExit(f"LOSO 에는 피험자가 2 명 이상 필요하다(현재 {subjects})")
    norms, augs = parse_choice(args.norms, NORMS), parse_choice(args.augs, AUGS)
    for i, norm in enumerate(norms):
        xn = normalize(x, norm, meta["session"].to_numpy(), shape)  # 겹과 무관하므로 조건마다 한 번
        for j, aug in enumerate(augs):
            run_condition(args, xn, meta, shape, subjects, norm, aug, models, f"{i * len(augs) + j + 1}/{len(norms) * len(augs)}")


def run_condition(
    args: argparse.Namespace, xn: np.ndarray, meta: pd.DataFrame, shape: tuple[int, int, int],
    subjects: list[int], norm: str, aug: str, models: tuple[str, ...], label: str,
) -> None:
    """한 조건(정규화+증강)에서 모든 피험자 겹을 돌리고 피험자별 CSV 로 저장한다."""
    for subject in subjects:
        path = result_path(args.out, norm, aug, args.task, subject)
        if path.exists() and set(models) <= set(pd.read_csv(path)["model"]):
            continue
        res = loso_fold(xn, meta, shape, subject, norm, aug, args.seed, models)
        write_csv_atomic(res, path)
        print(f"[조건 {label} {norm}+{aug}] subject{subject:02d}: "
              f"trial_acc {res.set_index('model')['trial_acc'].round(3).to_dict()}")


def summarize(out: Path, task: str) -> pd.DataFrame:
    """조건별 평균·표준편차와 기준(none+none) 대비 피험자 쌍 차이, Wilcoxon 부호순위 p 값."""
    paths = sorted((out / "loso").glob(f"*/{task}_subject*.csv"))
    if not paths:
        return pd.DataFrame()
    res = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    base = res[(res["norm"] == "none") & (res["aug"] == "none")].set_index(["model", "subject"])["trial_acc"]
    rows = []
    for (norm, aug, model), g in res.groupby(["norm", "aug", "model"]):
        acc = g.set_index("subject")["trial_acc"]
        row = {"norm": norm, "aug": aug, "model": model, "n_subjects": len(acc),
               "trial_acc_mean": acc.mean(), "trial_acc_std": acc.std(), "window_acc_mean": g["window_acc"].mean()}
        paired = base.get(model, pd.Series(dtype=float)).reindex(acc.index).dropna()
        diff = acc.loc[paired.index] - paired
        row["delta_vs_none"] = diff.mean() if len(diff) else np.nan
        row["wilcoxon_p"] = wilcoxon(diff).pvalue if len(diff) >= 5 and diff.abs().sum() > 0 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["model", "trial_acc_mean"], ascending=[True, False])
    write_csv_atomic(summary, out / f"loso_summary_{task}.csv")
    write_csv_atomic(res, out / f"loso_all_{task}.csv")
    save_plot(res, out, task)
    return summary


def save_plot(res: pd.DataFrame, out: Path, task: str) -> None:
    """조건별 피험자 시도 정확도(점)와 평균(가로 막대)을 모델마다 한 그림으로 저장한다."""
    for model, g in res.groupby("model"):
        conds = g.groupby(["norm", "aug"])["trial_acc"].mean().sort_values().index.tolist()
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(conds) + 1.5), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        for k, (norm, aug) in enumerate(conds):
            acc = g[(g["norm"] == norm) & (g["aug"] == aug)]["trial_acc"]
            ax.scatter(acc, np.full(len(acc), k), s=18, color="#2a78d6", alpha=0.6, linewidths=0)
            ax.plot([acc.mean()] * 2, [k - 0.3, k + 0.3], color="#0b0b0b", linewidth=2)
        ax.set_yticks(range(len(conds)), [f"{n} + {a}" for n, a in conds])
        ax.set_xlabel("trial accuracy (held-out subject)", color="#52514e")
        ax.set_title(f"HYSER LOSO - {model}, {task} (dots: subjects, bar: mean)", color="#0b0b0b", loc="left")
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#52514e", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"loso_{task}_{model}.png", dpi=150)
        plt.close(fig)


def parse_choice(text: str, allowed: tuple[str, ...]) -> list[str]:
    items = [t.strip() for t in text.split(",") if t.strip()]
    if not items or set(items) - set(allowed):
        raise SystemExit(f"{text!r}: {allowed} 중에서 고른다")
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_scope_args(p)
    p.set_defaults(models="lda")  # 로지스틱 회귀는 LOSO×조건 수만큼 돌기엔 느리다
    p.add_argument("--norms", default=",".join(NORMS))
    p.add_argument("--augs", default=",".join(AUGS))
    p.add_argument("--stage", choices=["all", "extract", "eval"], default="all")
    p.add_argument("--out", type=Path, default=Path("results") / "hyser_loso")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    sessions, models = scoped_sessions(args), model_names(args)
    failed = extract_all(sessions, args.out, args.task) if args.stage in ("all", "extract") else []
    if args.stage in ("all", "eval"):
        run_loso(args, sessions, models)
        summary = summarize(args.out, args.task)
        print(summary.round(3).to_string(index=False))
        print(f"저장: {args.out}/loso_summary_{args.task}.csv, loso_all_{args.task}.csv")
    if failed:
        print(f"특징 추출 실패 {len(failed)}개: {failed}")
    return 1 if failed else 0
