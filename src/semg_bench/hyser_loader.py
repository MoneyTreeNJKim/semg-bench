"""HYSER PR 데이터셋 한 세션 로더와 창 단위 RMS·파형 길이 특징 추출."""

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import wfdb

TASKS = ("dynamic", "maintenance")
KINDS = ("preprocess", "raw")


class HyserSession(NamedTuple):
    x: np.ndarray  # (시도 수, 채널, 샘플) float32
    y: np.ndarray  # (시도 수,) int64, 동작 번호 1~34
    fs: float
    channel_names: list[str]


def trial_records(session_dir: Path, task: str, kind: str) -> list[str]:
    """레코드 이름을 sample 번호 순(1, 2, ..., 10)으로 정렬해 돌려준다."""
    pattern = re.compile(rf"{task}_{kind}_sample(\d+)\.hea$")
    found = [(int(m.group(1)), p.stem) for p in session_dir.iterdir() if (m := pattern.match(p.name))]
    return [name for _, name in sorted(found)]


def read_labels(path: Path) -> np.ndarray:
    """쉼표로 구분된 라벨 파일을 정수 배열로 읽는다."""
    return np.array([int(v) for v in path.read_text().strip().split(",")], dtype=np.int64)


def load_hyser_session(
    session_dir: Path | str,
    task: str = "dynamic",
    kind: str = "preprocess",
    max_trials: int | None = None,
) -> HyserSession:
    """PR 세션 하나를 읽어 (시도 수, 채널, 샘플) 배열과 동작 라벨을 돌려준다."""
    if task not in TASKS or kind not in KINDS:
        raise ValueError(f"task 는 {TASKS}, kind 는 {KINDS} 중 하나여야 한다")
    session_dir = Path(session_dir)
    names = trial_records(session_dir, task, kind)
    labels = read_labels(session_dir / f"label_{task}.txt")
    if len(names) != len(labels):
        raise ValueError(f"레코드 {len(names)}개와 라벨 {len(labels)}개 수가 다르다")
    names, labels = names[:max_trials], labels[:max_trials]
    first = wfdb.rdheader(str(session_dir / names[0]))
    x = np.empty((len(names), first.n_sig, first.sig_len), dtype=np.float32)
    for i, name in enumerate(names):
        rec = wfdb.rdrecord(str(session_dir / name))
        if rec.fs != first.fs or rec.sig_name != first.sig_name or rec.sig_len != first.sig_len:
            raise ValueError(f"{name}: fs·채널·길이가 첫 레코드와 다르다")
        x[i] = rec.p_signal.T
    return HyserSession(x, labels, float(first.fs), list(first.sig_name))


def window_params(fs: float, win_ms: float, step_ms: float) -> tuple[int, int]:
    """창 길이와 간격(ms)을 샘플 수로 바꾼다."""
    win, step = round(win_ms * fs / 1000), round(step_ms * fs / 1000)
    if win < 2 or step < 1:
        raise ValueError("창은 2 샘플 이상, 간격은 1 샘플 이상이어야 한다")
    return win, step


def rms_wl_features(
    x: np.ndarray, fs: float, win_ms: float = 256.0, step_ms: float = 128.0
) -> tuple[np.ndarray, np.ndarray]:
    """(..., 채널, 샘플) 신호를 창으로 잘라 채널별 RMS 와 파형 길이를 구한다.

    반환 모양은 둘 다 (..., 창 수, 채널). 끝에 남는 창보다 짧은 구간은 버린다.
    """
    win, step = window_params(fs, win_ms, step_ms)
    n_samples = x.shape[-1]
    if n_samples < win:
        raise ValueError(f"신호 길이 {n_samples} 가 창 길이 {win} 보다 짧다")
    starts = range(0, n_samples - win + 1, step)
    rms = np.stack([np.sqrt(np.mean(np.square(x[..., s : s + win]), axis=-1)) for s in starts], axis=-2)
    wl = np.stack([np.sum(np.abs(np.diff(x[..., s : s + win], axis=-1)), axis=-1) for s in starts], axis=-2)
    return rms, wl
