import numpy as np

from semg_bench.hyser_baseline import run_cv, trial_folds, window_dataset


def synthetic_trials(n_classes: int = 4, per_class: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """클래스마다 채널 진폭 패턴이 다른 합성 sEMG 시도."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(1, n_classes + 1), per_class)
    gains = rng.uniform(0.2, 2.0, size=(n_classes, 8))
    x = rng.standard_normal((len(y), 8, 2048)) * gains[y - 1][:, :, None]
    return x.astype(np.float32), y


def test_window_dataset_shapes() -> None:
    x, y = synthetic_trials()
    feats, y_win, trial_ids = window_dataset(x, y, 2048.0)
    assert feats.shape == (len(y) * 6, 16)
    assert np.isfinite(feats).all()
    assert np.array_equal(y_win, np.repeat(y, 6))
    assert np.array_equal(np.bincount(trial_ids), np.full(len(y), 6))


def test_trial_folds_partition_trials() -> None:
    y = np.repeat(np.arange(1, 35), 6)
    folds = trial_folds(y, n_splits=5, seed=0)
    tests = np.concatenate([test for _, test in folds])
    assert sorted(tests.tolist()) == list(range(len(y)))
    for train, test in folds:
        assert not set(train) & set(test)
    assert [t.tolist() for _, t in folds] == [t.tolist() for _, t in trial_folds(y, 5, seed=0)]


def test_run_cv_separable_data() -> None:
    x, y = synthetic_trials()
    feats, y_win, trial_ids = window_dataset(x, y, 2048.0)
    acc, cms = run_cv(feats, y_win, trial_ids, y, seed=0)
    assert set(acc["model"]) == {"lda", "logreg"}
    assert len(acc) == 2 * 5
    assert acc[["window_acc", "trial_acc"]].stack().between(0, 1).all()
    assert acc["trial_acc"].mean() > 0.9
    for cm in cms.values():
        assert cm.shape == (4, 4)
        assert cm.sum() == len(y_win)
