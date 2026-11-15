import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semg_bench import hyser_loso
from semg_bench.hyser_loso import (
    augment,
    grid_order,
    loso_fold,
    normalize,
    session_features,
    shift_grid,
)

SHAPE = (2, 4, 4)  # 어레이 2 개 x 4x4 (합성)
ARRAYS = ("AA", "BB")
SESSION_DIR = Path(os.environ.get("DATA_ROOT", "./data")) / "hyser" / "subject01_session1"


def channel_names() -> list[str]:
    return [f"{a}-{r}-{c}" for a in ARRAYS for r in range(1, 5) for c in range(1, 5)]


def synthetic_subjects(n_subjects: int = 6, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """클래스마다 격자 위 활성 패턴이 다르고, 피험자마다 어레이 이득(log 오프셋)이 크게 다른 특징."""
    rng = np.random.default_rng(seed)
    templates = rng.normal(0, 1, size=(4, 2, *SHAPE))  # 클래스 4 개, 특징 종류 2 개
    xs, rows = [], []
    for s in range(1, n_subjects + 1):
        offset = rng.normal(0, 3, size=(1, 1, SHAPE[0], 1, 1))  # 피험자별 어레이 이득
        for trial in range(20):
            label = trial % 4
            x = templates[label] + offset[0] + rng.normal(0, 0.5, size=(3, 2, *SHAPE))
            xs.append(x.reshape(3, -1))
            rows += [{"subject": s, "session": f"s{s}", "label": label + 1, "trial_key": f"s{s}:{trial}"}] * 3
    return np.vstack(xs).astype(np.float32), pd.DataFrame(rows)


def test_grid_order_sorts_array_row_col() -> None:
    names = channel_names()[::-1]  # 역순으로 섞어도
    order, shape = grid_order(names)
    assert shape == SHAPE
    assert [names[i] for i in order] == channel_names()
    with pytest.raises(ValueError):
        grid_order(["AA-1-1", "bad"])


def test_shift_grid_edge_replicate() -> None:
    g = np.arange(9.0).reshape(3, 3)
    np.testing.assert_array_equal(shift_grid(g, 1, 0), [[0, 1, 2], [0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(shift_grid(g, 0, -1), [[1, 2, 2], [4, 5, 5], [7, 8, 8]])


def test_normalize_window_removes_array_gain() -> None:
    x, meta = synthetic_subjects(2)
    g = x.reshape(len(x), 2, SHAPE[0], -1)
    shifted = (g + np.array([5.0, -3.0])[None, None, :, None]).reshape(len(x), -1).astype(np.float32)
    np.testing.assert_allclose(normalize(shifted, "window", None, SHAPE), normalize(x, "window", None, SHAPE), atol=1e-4)
    z = normalize(x, "session", meta["session"].to_numpy(), SHAPE)
    for s in meta["session"].unique():
        np.testing.assert_allclose(z[(meta["session"] == s).to_numpy()].mean(axis=0), 0, atol=1e-4)


def test_augment_shapes_and_original_first() -> None:
    x, meta = synthetic_subjects(1)
    xa, ya = augment(x, meta["label"].to_numpy(), SHAPE, np.random.default_rng(0))
    assert xa.shape == (5 * len(x), x.shape[1]) and len(ya) == len(xa)
    np.testing.assert_array_equal(xa[: len(x)], x)
    assert not np.allclose(xa[len(x) : 2 * len(x)], x)


def shifted_test_subject(offset: float, shift: tuple[int, int], seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """학습 피험자 1-5 는 기준 상태, 평가 피험자 6 만 어레이 이득 offset·전극 이동 shift 가 있는 데이터.

    학습 피험자끼리 이득이 다양하면 LDA 가 그 방향을 스스로 무시하므로, 학습에서 못 본 변화로 효과를 검증한다.
    """
    rng = np.random.default_rng(seed)
    templates = rng.normal(0, 1, size=(4, 2, *SHAPE))
    xs, rows = [], []
    for s in range(1, 7):
        for trial in range(20):
            g = templates[trial % 4] + rng.normal(0, 0.7, size=(3, 2, *SHAPE))
            if s == 6:
                g = shift_grid(g, *shift) + np.array([offset, -offset])[None, None, :, None, None]
            xs.append(g.reshape(3, -1))
            rows += [{"subject": s, "session": f"s{s}", "label": trial % 4 + 1, "trial_key": f"s{s}:{trial}"}] * 3
    return np.vstack(xs).astype(np.float32), pd.DataFrame(rows)


def held_out_acc(x: np.ndarray, meta: pd.DataFrame, norm: str, aug: str) -> float:
    xn = normalize(x, norm, meta["session"].to_numpy(), SHAPE)
    return float(loso_fold(xn, meta, SHAPE, 6, norm, aug, 0, ("lda",))["window_acc"].iloc[0])


def test_window_norm_fixes_unseen_array_gain() -> None:
    x, meta = shifted_test_subject(offset=4.0, shift=(0, 0))
    assert held_out_acc(x, meta, "window", "none") > held_out_acc(x, meta, "none", "none") + 0.2


def test_shift_aug_helps_with_electrode_displacement() -> None:
    x, meta = shifted_test_subject(offset=0.0, shift=(1, 0))
    assert held_out_acc(x, meta, "none", "shift") > held_out_acc(x, meta, "none", "none") + 0.15


def test_main_resumes_and_summarizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    x, meta = synthetic_subjects(5)
    cols = [f"{t}_{n}" for t in ("rms", "wl") for n in channel_names()]
    for s in range(1, 6):
        (tmp_path / "data" / f"subject{s:02d}_session1").mkdir(parents=True)
        m = (meta["subject"] == s).to_numpy()
        df = pd.DataFrame(x[m], columns=cols).assign(label=meta.loc[m, "label"].to_numpy(), trial=np.repeat(range(20), 3))
        hyser_loso.write_csv_atomic(df, hyser_loso.feature_path(tmp_path / "out", f"subject{s:02d}_session1", "dynamic"))
    argv = ["--data", str(tmp_path / "data"), "--out", str(tmp_path / "out"), "--augs", "none"]
    assert hyser_loso.main(argv) == 0
    summary = pd.read_csv(tmp_path / "out" / "loso_summary_dynamic.csv")
    assert set(summary["norm"]) == {"none", "window", "session"} and (summary["n_subjects"] == 5).all()
    assert summary.loc[summary["norm"] == "none", "delta_vs_none"].iloc[0] == 0
    assert (tmp_path / "out" / "loso_dynamic_lda.png").exists()

    def must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("완료된 겹을 다시 돌렸다")

    monkeypatch.setattr(hyser_loso, "loso_fold", must_not_run)
    assert hyser_loso.main(argv) == 0


def test_loso_needs_two_subjects(tmp_path: Path) -> None:
    x, meta = synthetic_subjects(1)
    (tmp_path / "data" / "subject01_session1").mkdir(parents=True)
    cols = [f"{t}_{n}" for t in ("rms", "wl") for n in channel_names()]
    df = pd.DataFrame(x, columns=cols).assign(label=meta["label"].to_numpy(), trial=np.repeat(range(20), 3))
    hyser_loso.write_csv_atomic(df, hyser_loso.feature_path(tmp_path / "out", "subject01_session1", "dynamic"))
    with pytest.raises(SystemExit):
        hyser_loso.main(["--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")])


@pytest.mark.skipif(not SESSION_DIR.exists(), reason="HYSER 샘플 데이터 없음")
def test_session_features_grid_order_real() -> None:
    df = session_features(SESSION_DIR, "dynamic")
    fcols = [c for c in df.columns if c.startswith(("rms_", "wl_"))]
    assert len(fcols) == 512 and len(df) == 202 * 6
    assert fcols[0] == "rms_ED-1-1" and fcols[255] == "rms_FP-8-8" and fcols[256] == "wl_ED-1-1"
    assert np.isfinite(df[fcols].to_numpy()).all()
    assert df["label"].between(1, 34).all() and df["trial"].nunique() == 202
