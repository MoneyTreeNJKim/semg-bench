import os
from pathlib import Path

import numpy as np
import pytest

from semg_bench.hyser_loader import load_hyser_session, rms_wl_features, window_params

SESSION_DIR = Path(os.environ.get("DATA_ROOT", "./data")) / "hyser" / "subject01_session1"
needs_data = pytest.mark.skipif(not SESSION_DIR.exists(), reason="HYSER 샘플 데이터 없음")
FS = 2048.0


def test_window_params_at_2048hz() -> None:
    assert window_params(FS, 256, 128) == (524, 262)


def test_features_shape() -> None:
    x = np.random.default_rng(0).standard_normal((3, 256, 2048)).astype(np.float32)
    rms, wl = rms_wl_features(x, FS)
    n_windows = 1 + (2048 - 524) // 262
    assert rms.shape == wl.shape == (3, n_windows, 256)


def test_features_constant_signal() -> None:
    x = np.full((2, 4, 2048), -0.5, dtype=np.float64)
    rms, wl = rms_wl_features(x, FS)
    np.testing.assert_allclose(rms, 0.5)
    np.testing.assert_allclose(wl, 0.0)


def test_features_sine_values() -> None:
    t = np.arange(4096) / FS
    x = (2.0 * np.sin(2 * np.pi * 64 * t))[None, :]  # 64 Hz, 창(524 샘플)에 약 16.4 주기
    rms, wl = rms_wl_features(x, FS)
    np.testing.assert_allclose(rms, 2.0 / np.sqrt(2), rtol=0.02)
    # 한 주기의 파형 길이는 4 * 진폭
    np.testing.assert_allclose(wl, 4 * 2.0 * 64 * 523 / FS, rtol=0.05)


def test_features_rejects_short_signal() -> None:
    with pytest.raises(ValueError):
        rms_wl_features(np.zeros((1, 100)), FS)


@needs_data
@pytest.mark.parametrize(("task", "n_samples"), [("dynamic", 2048), ("maintenance", 8192)])
def test_load_session_shape_and_range(task: str, n_samples: int) -> None:
    session = load_hyser_session(SESSION_DIR, task=task, max_trials=5)
    assert session.x.shape == (5, 256, n_samples)
    assert session.x.dtype == np.float32
    assert session.y.shape == (5,)
    assert session.fs == FS
    assert len(session.channel_names) == 256
    assert np.isfinite(session.x).all()
    assert np.abs(session.x).max() < 100  # 헤더 단위는 V 이지만 값은 mV 수준
    assert ((session.y >= 1) & (session.y <= 34)).all()


@needs_data
def test_full_label_set() -> None:
    session = load_hyser_session(SESSION_DIR, task="maintenance", max_trials=None)
    assert session.x.shape[0] == len(session.y) == 68
    assert set(session.y.tolist()) == set(range(1, 35))


@needs_data
def test_features_on_real_data() -> None:
    session = load_hyser_session(SESSION_DIR, max_trials=3)
    rms, wl = rms_wl_features(session.x, session.fs)
    assert rms.shape == wl.shape == (3, 6, 256)
    assert np.isfinite(rms).all() and np.isfinite(wl).all()
    assert (rms > 0).all() and (wl > 0).all()
    assert (rms <= np.abs(session.x).max()).all()
