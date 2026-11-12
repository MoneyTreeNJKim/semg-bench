import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from semg_bench import emg2pose_shards
from semg_bench.emg2pose_shards import detect_index_columns, main

FS = 2000.0


def write_fake_hdf5(path: Path, seed: int, seconds: float = 4.0) -> None:
    """실제 emg2pose 와 같은 구조(emg2pose/timeseries 구조체 배열 + sample_rate 속성)의 작은 파일."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    dtype = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (16,))])
    ts = np.zeros(n, dtype=dtype)
    ts["time"] = 1.67e9 + np.arange(n) / FS
    gain = rng.uniform(0.5, 2.0, size=(n // int(FS) + 1, 16)).repeat(int(FS), axis=0)[:n]
    ts["emg"] = rng.standard_normal((n, 16)) * gain
    ts["joint_angles"] = np.log(gain) @ rng.standard_normal((16, 20)) * 0.1
    with h5py.File(path, "w") as f:
        g = f.create_group("emg2pose")
        g.create_dataset("timeseries", data=ts)
        g.attrs["sample_rate"] = FS


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    """tar 샤드 2개, 열 이름이 임의인 index.csv, 메타데이터 CSV 를 만든다."""
    d = tmp_path / "shards"
    d.mkdir()
    rows, meta = [], []
    for i in range(8):
        name = f"rec{i}"
        user, stage = f"u{i % 4}", f"s{i % 4}"
        split = "train" if i % 4 < 3 else "val"
        write_fake_hdf5(tmp_path / f"{name}.hdf5", seed=i)
        rows.append({"shard_file": f"shard{i // 4}.tar", "member_path": f"ds/{name}.hdf5"})
        meta.append({"filename": name, "user": user, "split": split, "stage": stage, "side": "left"})
    for s in range(2):
        with tarfile.open(d / f"shard{s}.tar", "w") as tar:
            for i in range(s * 4, s * 4 + 4):
                tar.add(tmp_path / f"rec{i}.hdf5", arcname=f"ds/rec{i}.hdf5")
    pd.DataFrame(rows).to_csv(d / "index.csv", index=False)
    pd.DataFrame(meta).to_csv(d / "emg2pose_metadata.csv", index=False)
    return d


def run(shard_dir: Path, tmp_path: Path, *extra: str) -> int:
    return main(["--shards-dir", str(shard_dir), "--out", str(tmp_path / "out"),
                 "--local-dir", str(tmp_path / "local"), *extra])


def test_detect_index_columns() -> None:
    index = pd.DataFrame({"a": ["x.tar", "y.tar"], "b": ["d/r1.hdf5", "d/r2.hdf5"], "c": ["1", "2"]})
    assert detect_index_columns(index, {"r1", "r2"}) == ("b", "a")
    with pytest.raises(ValueError):
        detect_index_columns(index[["c"]], {"r1"})


def test_extract_eval_and_resume(shard_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert run(shard_dir, tmp_path) == 0
    out = tmp_path / "out"
    feats = sorted((out / "features_win1_valid0.9").rglob("*.csv.gz"))
    assert len(feats) == 8
    df = pd.read_csv(feats[0])
    assert len(df) == 4 and df.filter(like="rms_").shape[1] == 16 and df.filter(like="joint_").shape[1] == 20
    assert not any((tmp_path / "local").rglob("*.hdf5"))  # 풀었던 파일은 지운다
    table = pd.read_csv(out / "joint_mae.csv")
    assert set(table["split"]) == {"val"} and len(table) == 20
    assert sorted(p.stem for p in (out / "per_user").glob("*.csv")) == ["u3"]

    def must_not_run(*args: object) -> None:
        raise AssertionError("완료된 녹화를 다시 처리했다")

    monkeypatch.setattr(emg2pose_shards, "process_shard", must_not_run)
    assert run(shard_dir, tmp_path) == 0  # 두 번째 실행은 추출을 건너뛴다


def test_scope_filters_shards(shard_dir: Path, tmp_path: Path) -> None:
    assert run(shard_dir, tmp_path, "--shards", "shard1", "--stage", "extract") == 0
    users = sorted(p.parent.name for p in (tmp_path / "out" / "features_win1_valid0.9").rglob("*.csv.gz"))
    assert users == ["u0", "u1", "u2", "u3"] and len(users) == 4


def test_missing_shard_does_not_stop_run(shard_dir: Path, tmp_path: Path) -> None:
    index = pd.read_csv(shard_dir / "index.csv")
    index.loc[index["member_path"] == "ds/rec0.hdf5", "shard_file"] = "missing.tar"
    index.to_csv(shard_dir / "index.csv", index=False)
    assert run(shard_dir, tmp_path, "--stage", "extract") == 1  # 실패를 종료 코드로 알린다
    assert len(list((tmp_path / "out" / "features_win1_valid0.9").rglob("*.csv.gz"))) == 7
