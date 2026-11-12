from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semg_bench import hyser_all


@pytest.fixture
def fake_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """세션 디렉터리만 만들고 run_session 은 가짜 결과를 돌려주게 바꾼다(호출 기록)."""
    for name in ["subject01_session1", "subject02_session1", "subject02_session2"]:
        (tmp_path / "data" / name).mkdir(parents=True)
    calls: list[str] = []

    def fake_run(session_dir: Path, task: str, seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
        calls.append(session_dir.name)
        acc = pd.DataFrame({"model": ["lda", "lda"], "fold": [0, 1], "window_acc": [0.8, 0.9], "trial_acc": [0.9, 1.0]})
        return acc.assign(session=session_dir.name, task=task, seed=seed), {"lda": np.eye(2, dtype=int)}, np.array([1, 2])

    monkeypatch.setattr(hyser_all, "run_session", fake_run)
    return calls


def test_run_all_resumes(tmp_path: Path, fake_sessions: list[str]) -> None:
    argv = ["--data", str(tmp_path / "data"), "--out", str(tmp_path / "out"), "--subjects", "2"]
    assert hyser_all.main(argv) == 0
    assert fake_sessions == ["subject02_session1", "subject02_session2"]
    assert (tmp_path / "out" / "accuracy" / "subject02_session1_dynamic.csv").exists()
    assert (tmp_path / "out" / "confusion" / "subject02_session2_dynamic_lda.csv").exists()

    assert hyser_all.main(argv + ["--subjects", "1-2"]) == 0  # 두 번째 실행: 새 세션만 돈다
    assert fake_sessions == ["subject02_session1", "subject02_session2", "subject01_session1"]
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert summary.loc[0, "trial_acc_count"] == 3
    assert summary.loc[0, "trial_acc_mean"] == pytest.approx(0.95)


def test_run_all_no_match(tmp_path: Path, fake_sessions: list[str]) -> None:
    with pytest.raises(SystemExit):
        hyser_all.main(["--data", str(tmp_path / "data"), "--subjects", "7"])
