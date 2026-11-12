import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from semg_bench.emg2pose_baseline import (
    FeatureConfig,
    assign_split,
    evaluate,
    fit_predict,
    window_features,
)
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
    x, y = window_features(emg_w, ang_w, min_valid=1.0, features=("rms",))
    assert x.shape == (2, 2) and y.shape == (2, 4)
    np.testing.assert_allclose(x, np.log(2.0), rtol=1e-6)
    np.testing.assert_allclose(y, 1.0)


def test_window_features_partial_ik_mask() -> None:
    emg_w = np.ones((3, 100, 2), dtype=np.float32)
    ang_w = np.ones((3, 100, 4), dtype=np.float32)
    ang_w[0, :5] = np.nan  # 5% 무효 -> 유지, 유효 샘플만 평균
    ang_w[0, 5:10] = 3.0
    ang_w[1, :20] = np.nan  # 20% 무효 -> 버림
    x, y = window_features(emg_w, ang_w, min_valid=0.9, features=("rms",))
    assert x.shape == (2, 2)
    np.testing.assert_allclose(y[0], (5 * 3.0 + 90 * 1.0) / 95)
    np.testing.assert_allclose(y[1], 1.0)


def test_loader_masks_ik_failures(tmp_path: Path) -> None:
    dtype = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (16,))])
    ts = np.zeros(4, dtype=dtype)
    ts["time"] = np.arange(4) / 2000.0
    ts["joint_angles"][0] = 0.5
    ts["joint_angles"][2, 3] = 0.1  # 일부 관절만 0 인 행은 정상 자세
    with h5py.File(tmp_path / "r.hdf5", "w") as f:
        f.create_group("emg2pose").create_dataset("timeseries", data=ts)
        f["emg2pose"].attrs["sample_rate"] = 2000.0
    rec = load_recording(tmp_path / "r.hdf5")
    assert np.isnan(rec.angles).all(axis=1).tolist() == [False, True, False, True]
    raw = load_recording(tmp_path / "r.hdf5", mask_ik_failures=False)
    assert np.isfinite(raw.angles).all()


def test_window_features_rms_and_wl_blocks() -> None:
    t = np.arange(2000, dtype=np.float64)
    emg_w = np.stack([np.where(t % 2 == 0, 1.0, -1.0), np.full(2000, 3.0)], axis=1)[None]  # (1, 2000, 2)
    x, _ = window_features(emg_w, np.zeros((1, 2000, 4)), features=("rms", "wl"))
    assert x.shape == (1, 4)  # [rms_ch0, rms_ch1, wl_ch0, wl_ch1]
    np.testing.assert_allclose(x[0, :2], np.log([1.0, 3.0]), atol=1e-6)
    np.testing.assert_allclose(np.exp(x[0, 2]), 2.0, rtol=1e-6)  # ±1 교대 -> 샘플당 변화 2
    assert x[0, 3] < -15  # 상수 채널의 WL 은 0 (log EPS)


def test_feature_config_tag_changes_with_settings() -> None:
    assert FeatureConfig().tag == "win1_step0.25_valid0.9_rms+wl"
    assert FeatureConfig(features=("rms",)).tag != FeatureConfig().tag


def test_fit_predict_per_group_uses_group_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 3))
    g = np.repeat(["left", "right"], 100)
    y = np.where(g[:, None] == "left", 1.0, -1.0) + x @ np.ones((3, 2)) * 0.1
    _, mean_pred, _ = fit_predict(x, y, x, g, g)
    np.testing.assert_allclose(mean_pred[g == "left"], np.broadcast_to(y[g == "left"].mean(axis=0), (100, 2)))
    np.testing.assert_allclose(mean_pred[g == "right"], np.broadcast_to(y[g == "right"].mean(axis=0), (100, 2)))
    pooled, _, _ = fit_predict(x, y, x)
    grouped, _, _ = fit_predict(x, y, x, g, g)
    assert np.abs(grouped - y).mean() < np.abs(pooled - y).mean()


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
