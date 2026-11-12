"""emg2pose 전체 데이터(Drive 의 2GB tar 샤드 + index.csv) 처리.

필요한 샤드만 순차로 읽어 해당 HDF5 만 로컬에 풀고, 창 특징을 녹화별 CSV 로 저장한 뒤 바로 지운다.
특징 CSV 가 이미 있는 녹화는 건너뛰므로 중단 후 같은 명령으로 이어서 돌릴 수 있다.

진입점: scripts/run_emg2pose_shards.py
중간 결과: results/emg2pose/features_win<창>_valid<비율>/<user>/<filename>.csv.gz (설정이 바뀌면 새 디렉터리)
최종 결과: results/emg2pose/per_user/<user>.csv, results/emg2pose/joint_mae.csv
"""

import argparse
import tarfile
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from semg_bench.data_scope import data_root, parse_id_list, write_csv_atomic
from semg_bench.emg2pose_baseline import (
    add_scope_args,
    assign_split,
    fit_ridge,
    joint_table,
    recording_features,
)
from semg_bench.emg2pose_loader import METADATA_CSV

DEFAULT_SHARDS_DIR = Path("/content/drive/MyDrive/TabDev/datasets/emg2poseshards")
META_COLS = ["filename", "user", "split", "stage", "side"]


def hdf5_stem(value: str) -> str:
    return Path(str(value)).name.removesuffix(".hdf5")


def detect_index_columns(
    index: pd.DataFrame, known: set[str], file_col: str | None = None, shard_col: str | None = None
) -> tuple[str, str]:
    """index.csv 에서 HDF5 파일 열과 샤드(tar) 열을 찾는다. 열 이름은 가정하지 않고 값으로 판별한다."""
    if shard_col is None:
        tars = [c for c in index.columns if index[c].astype(str).str.endswith(".tar").all()]
        shard_col = tars[0] if tars else None
    if file_col is None:
        overlap = {c: index[c].map(hdf5_stem).isin(known).mean() for c in index.columns if c != shard_col}
        best = max(overlap, key=overlap.get, default=None)
        file_col = best if best is not None and overlap[best] > 0.5 else None
    if shard_col is None or file_col is None:
        raise ValueError(f"index.csv 열 {list(index.columns)} 에서 파일/샤드 열을 찾지 못했다. --file-col, --shard-col 로 지정하라")
    return file_col, shard_col


def load_plan(shards_dir: Path, metadata: Path, file_col: str | None, shard_col: str | None) -> pd.DataFrame:
    """index.csv 와 메타데이터를 합쳐 녹화별 (filename, shard, user, split, stage, side) 표를 만든다."""
    index = pd.read_csv(shards_dir / "index.csv", dtype=str)
    meta = pd.read_csv(metadata, dtype={"user": str, "filename": str})
    fcol, scol = detect_index_columns(index, set(meta["filename"]), file_col, shard_col)
    plan = pd.DataFrame({"filename": index[fcol].map(hdf5_stem), "shard": index[scol]}).drop_duplicates("filename")
    merged = plan.merge(meta[META_COLS], on="filename", how="inner", validate="one_to_one")
    if len(merged) < len(plan):
        print(f"경고: index.csv 의 {len(plan) - len(merged)}개 파일이 메타데이터에 없어 제외")
    print(f"index.csv: 파일 열 '{fcol}', 샤드 열 '{scol}', 녹화 {len(merged)}개, 샤드 {merged['shard'].nunique()}개")
    return merged


def filter_plan(plan: pd.DataFrame, users: list[str] | None, shards: list[str] | None) -> pd.DataFrame:
    """user 목록과 샤드 목록(tar 이름 또는 stem)으로 거른다. None 은 전체."""
    keep = pd.Series(True, index=plan.index)
    if users is not None:
        keep &= plan["user"].isin(users)
    if shards is not None:
        name = plan["shard"].map(lambda s: Path(s).name)
        keep &= name.isin(shards) | name.str.removesuffix(".tar").isin(shards)
    return plan[keep].reset_index(drop=True)


class FeatureConfig(NamedTuple):
    """특징 추출 설정. 캐시 디렉터리 이름에 들어가 설정이 다른 예전 특징이 섞이지 않게 한다."""

    win_s: float
    min_valid: float

    @property
    def tag(self) -> str:
        return f"win{self.win_s:g}_valid{self.min_valid:g}"


def feature_path(out: Path, cfg: FeatureConfig, user: str, filename: str) -> Path:
    return out / f"features_{cfg.tag}" / user / f"{filename}.csv.gz"


def features_frame(x: np.ndarray, y: np.ndarray, row: pd.Series) -> pd.DataFrame:
    """창 특징(rms_XX)과 목표(joint_XX)에 녹화 메타데이터를 붙인 표."""
    df = pd.DataFrame(np.hstack([x, y]), columns=[f"rms_{i:02d}" for i in range(x.shape[1])]
                      + [f"joint_{j:02d}" for j in range(y.shape[1])])
    return df.assign(**{c: row[c] for c in META_COLS}, window=np.arange(len(df)))


def process_shard(shard_path: Path, rows: pd.DataFrame, out: Path, cfg: FeatureConfig, local_dir: Path) -> list[str]:
    """샤드를 스트림으로 한 번 읽으며 필요한 HDF5 만 하나씩 로컬에 풀어 특징을 저장하고 지운다. 실패 목록 반환."""
    needed = {r["filename"]: r for _, r in rows.iterrows()}
    done, failed = set(), []
    with tarfile.open(shard_path, mode="r|*") as tar:
        for member in tar:
            name = hdf5_stem(member.name)
            if not member.isfile() or not member.name.endswith(".hdf5") or name not in needed:
                continue
            tar.extract(member, local_dir, filter="data")
            local = local_dir / member.name
            try:
                x, y = recording_features(local, None, cfg.win_s, cfg.min_valid)
                row = needed[name]
                write_csv_atomic(features_frame(x, y, row), feature_path(out, cfg, row["user"], name))
            except (OSError, KeyError, ValueError) as e:
                print(f"  {name}: 실패 - {e}")
                failed.append(name)
            finally:
                local.unlink(missing_ok=True)
            done.add(name)
            if len(done) == len(needed):
                break  # 남은 샤드는 읽지 않는다
    return failed + sorted(set(needed) - done)


def extract_features(plan: pd.DataFrame, shards_dir: Path, out: Path, cfg: FeatureConfig, local_dir: Path) -> list[str]:
    """특징 CSV 가 없는 녹화만 골라 샤드별로 처리한다. 실패·누락 녹화 이름을 돌려준다."""
    exists = plan.apply(lambda r: feature_path(out, cfg, r["user"], r["filename"]).exists(), axis=1)
    pending = plan[~exists]
    print(f"녹화 {len(plan)}개 중 완료 {int(exists.sum())}, 남음 {len(pending)} (샤드 {pending['shard'].nunique()}개)")
    local_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for i, (shard, rows) in enumerate(pending.groupby("shard", sort=True), 1):
        print(f"[{i}/{pending['shard'].nunique()}] {shard}: 녹화 {len(rows)}개")
        try:
            failed += process_shard(shards_dir / shard, rows, out, cfg, local_dir)
        except (OSError, tarfile.TarError) as e:  # 샤드가 없거나 손상돼도 나머지 샤드는 계속한다
            print(f"  {shard}: 샤드 실패 - {e}")
            failed += rows["filename"].tolist()
    return failed


def load_features(plan: pd.DataFrame, out: Path, cfg: FeatureConfig) -> pd.DataFrame:
    """범위 안의 녹화 특징 CSV 를 모두 읽어 하나의 표로 합친다."""
    paths = [p for _, r in plan.iterrows() if (p := feature_path(out, cfg, r["user"], r["filename"])).exists()]
    if not paths:
        raise SystemExit("특징 CSV 가 없다. 먼저 추출 단계를 돌려라")
    return pd.concat([pd.read_csv(p, dtype={"user": str}) for p in paths], ignore_index=True)


def evaluate_features(feats: pd.DataFrame, seed: int, out: Path) -> pd.DataFrame:
    """CSV split(train 이 없으면 stage 분할)으로 릿지를 학습해 전체·피험자별 관절 MAE 를 저장한다."""
    recs = feats.drop_duplicates("filename")[["filename", "split", "stage"]].reset_index(drop=True)
    eval_split = feats["filename"].map(dict(zip(recs["filename"], assign_split(recs, seed), strict=True)))
    xcols, ycols = feats.filter(like="rms_").columns, feats.filter(like="joint_").columns
    x, y = feats[xcols].to_numpy(), feats[ycols].to_numpy()
    train = (eval_split == "train").to_numpy()
    model = fit_ridge(x[train], y[train])
    err = pd.DataFrame(np.abs(model.predict(x[~train]) - y[~train]), columns=ycols)
    base = pd.DataFrame(np.abs(y[train].mean(axis=0) - y[~train]), columns=ycols)
    err["split"] = base["split"] = eval_split[~train].to_numpy()
    err["user"] = base["user"] = feats.loc[~train, "user"].to_numpy()
    alpha = float(model[-1].alpha_)
    for user, g in err.groupby("user"):
        b = base.loc[g.index]
        res = {s: (gs[ycols].mean().to_numpy(), b.loc[gs.index, ycols].mean().to_numpy(), alpha) for s, gs in g.groupby("split")}
        write_csv_atomic(joint_table(res).assign(user=user), out / "per_user" / f"{user}.csv")
    res = {s: (g[ycols].mean().to_numpy(), base.loc[g.index, ycols].mean().to_numpy(), alpha) for s, g in err.groupby("split")}
    table = joint_table(res).assign(seed=seed, n_train_windows=int(train.sum()))
    write_csv_atomic(table, out / "joint_mae.csv")
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR, help="tar 샤드와 index.csv 가 있는 디렉터리")
    p.add_argument("--metadata", type=Path, default=None, help="기본: 샤드 디렉터리 또는 $DATA_ROOT/emg2pose 의 메타데이터 CSV")
    p.add_argument("--local-dir", type=Path, default=Path(tempfile.gettempdir()) / "emg2pose_extract")
    p.add_argument("--out", type=Path, default=Path("results") / "emg2pose")
    p.add_argument("--file-col", default=None)
    p.add_argument("--shard-col", default=None)
    p.add_argument("--stage", choices=["all", "extract", "eval"], default="all")
    add_scope_args(p)
    return p.parse_args(argv)


def resolve_metadata(args: argparse.Namespace) -> Path:
    candidates = [args.metadata] if args.metadata else [args.shards_dir / METADATA_CSV, data_root() / "emg2pose" / METADATA_CSV]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"메타데이터 CSV 를 찾지 못했다: {candidates}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    plan = load_plan(args.shards_dir, resolve_metadata(args), args.file_col, args.shard_col)
    plan = filter_plan(plan, parse_id_list(args.users), parse_id_list(args.shards))
    if plan.empty:
        raise SystemExit(f"범위(users={args.users}, shards={args.shards})에 맞는 녹화가 없다")
    cfg = FeatureConfig(args.win_s, args.min_valid)
    failed = []
    if args.stage in ("all", "extract"):
        failed = extract_features(plan, args.shards_dir, args.out, cfg, args.local_dir)
    if args.stage in ("all", "eval"):
        table = evaluate_features(load_features(plan, args.out, cfg), args.seed, args.out)
        print(table.groupby("split")[["mae_deg", "mean_baseline_mae_deg"]].mean().round(2).to_string())
        print(f"저장: {args.out}/joint_mae.csv, {args.out}/per_user/*.csv")
    if failed:
        print(f"실패·누락 {len(failed)}개 (다시 실행하면 재시도): {failed[:10]}{' ...' if len(failed) > 10 else ''}")
    return 1 if failed else 0
