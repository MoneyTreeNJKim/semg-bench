import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semg_bench.emg2pose_baseline import assign_split, evaluate, window_features
from semg_bench.emg2pose_loader import list_recordings, load_recording, make_windows

DATA_DIR = Path(os.environ.get("DATA_ROOT", "./data")) / "emg2pose"
needs_data = pytest.mark.skipif(not (DATA_DIR / "emg2pose_metadata.csv").exists(), reason="emg2pose 데이터 없음")


def test_make_windows_shapes_and_alignment() -> None:
    t = np.arange(5500, dtype=np.float32)
    emg = np.stack([t, -t], axis=1)
    angles = np.stack([t * 2, t * 3, t * 4], axis=1)
    emg_w, ang_w = make_windows(emg, angles, fs=2000.0)
    assert emg_w.shape == (2, 2000, 2)
    assert ang_w.shape == (2, 2000, 3)
    np.testing.assert_array_equal(emg_w[1, :, 0], t[2000:4000])
    np.testing.assert_array_equal(ang_w[1, :, 0], 2 * emg_w[1, :, 0])


def test_make_windows_overlap_and_short() -> None:
    x = np.zeros((5000, 4), dtype=np.float32)
    assert make_windows(x, x[:, :2], 2000.0, step_s=0.5)[0].shape == (4, 2000, 4)  # 시작 0, 1000, 2000, 3000
    assert make_windows(x[:100], x[:100, :2], 2000.0)[0].shape == (0, 2000, 4)
    with pytest.raises(ValueError):
        make_windows(x, x[:10], 2000.0)


def test_window_features_values_and_nan() -> None:
    emg_w = np.full((3, 2000, 2), 2.0, dtype=np.float32)
    ang_w = np.ones((3, 2000, 4), dtype=np.float32)
    ang_w[1, 5, 0] = np.nan
    x, y = window_features(emg_w, ang_w)
    assert x.shape == (2, 2) and y.shape == (2, 4)
    np.testing.assert_allclose(x, np.log(2.0), rtol=1e-6)
    np.testing.assert_allclose(y, 1.0)


def test_assign_split_uses_csv_when_train_present() -> None:
    meta = pd.DataFrame({"split": ["train", "val", "test"], "stage": ["a", "a", "b"]})
    assert assign_split(meta, seed=0).tolist() == ["train", "val", "test"]


def test_assign_split_stage_fallback() -> None:
    meta = pd.DataFrame({"split": ["val"] * 20, "stage": [f"s{i % 10}" for i in range(20)]})
    split = assign_split(meta, seed=0)
    val_stages = set(meta.loc[split == "val", "stage"])
    assert len(val_stages) == 2
    assert not val_stages & set(meta.loc[split == "train", "stage"])
    assert split.equals(assign_split(meta, seed=0))


def test_evaluate_beats_mean_on_linear_data() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 16))
    y = x @ rng.standard_normal((16, 20)) * 0.1 + 0.01 * rng.standard_normal((400, 20))
    is_train = np.arange(400) < 300
    mae, mean_mae, _ = evaluate(x, y, is_train, ~is_train)
    assert mae.shape == mean_mae.shape == (20,)
    assert (mae < mean_mae).all()


@needs_data
def test_list_and_load_recording() -> None:
    recs = list_recordings(DATA_DIR)
    assert len(recs) > 0 and recs["split"].notna().all()
    row = recs.iloc[0]
    rec = load_recording(row.path, row.member)
    assert rec.emg.shape == (len(rec.time), 16)
    assert rec.angles.shape == (len(rec.time), 20)
    assert rec.fs == 2000.0
    np.testing.assert_allclose(np.median(np.diff(rec.time)), 1 / rec.fs, rtol=1e-3)
    assert np.isfinite(rec.emg).all()
    assert np.nanmax(np.abs(rec.angles)) < 2 * np.pi  # 라디안
    assert rec.attrs["filename"] == row.filename
