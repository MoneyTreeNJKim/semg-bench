"""emg2pose HDF5 로더: 시간 정렬된 sEMG·관절 각도 배열과 1 초 창 자르기."""

import tarfile
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

METADATA_CSV = "emg2pose_metadata.csv"


class Emg2PoseRecording(NamedTuple):
    emg: np.ndarray  # (T, 채널) float32
    angles: np.ndarray  # (T, 관절) float32, emg 와 같은 시간축
    time: np.ndarray  # (T,) float64, 초 단위 타임스탬프
    fs: float
    attrs: dict[str, object]


def find_recordings(root: Path) -> pd.DataFrame:
    """root 아래 HDF5 를 찾아 filename, path, member(tar 안이면 멤버 이름, 아니면 None) 표로 돌려준다."""
    rows = [{"filename": p.stem, "path": str(p), "member": None} for p in sorted(root.rglob("*.hdf5"))]
    for tar_path in sorted(root.rglob("*.tar")):
        with tarfile.open(tar_path) as tar:
            members = sorted(n for n in tar.getnames() if n.endswith(".hdf5"))
        rows += [{"filename": Path(n).stem, "path": str(tar_path), "member": n} for n in members]
    return pd.DataFrame(rows, columns=["filename", "path", "member"]).drop_duplicates("filename")


def list_recordings(root: Path | str) -> pd.DataFrame:
    """찾은 HDF5 목록을 메타데이터 CSV(split, stage, side 등)와 filename 으로 합친다."""
    root = Path(root)
    found = find_recordings(root)
    meta = pd.read_csv(root / METADATA_CSV, dtype={"user": str, "filename": str})
    merged = found.merge(meta, on="filename", how="left", validate="one_to_one")
    missing = merged["split"].isna().sum()
    if missing:
        raise ValueError(f"메타데이터에 없는 파일 {missing}개")
    return merged.reset_index(drop=True)


def read_recording(f: h5py.File, mask_ik_failures: bool = True) -> Emg2PoseRecording:
    """열린 HDF5 에서 emg2pose/timeseries 를 읽는다. emg 와 각도는 같은 행이라 이미 정렬돼 있다.

    mask_ik_failures: 관절 각도가 전부 정확히 0 인 행은 IK(역기구학) 실패 구간이므로 NaN 으로 바꾼다.
    미니 데이터셋에서 평균 15%, 많게는 47% 의 행이 이렇다.
    """
    group = f["emg2pose"]
    ts = group["timeseries"][:]
    attrs = dict(group.attrs)
    time = ts["time"].astype(np.float64)
    if np.any(np.diff(time) <= 0):
        raise ValueError("timeseries 의 time 이 단조 증가하지 않는다")
    emg = np.ascontiguousarray(ts["emg"], dtype=np.float32)
    angles = np.array(ts["joint_angles"], dtype=np.float32)
    if mask_ik_failures:
        angles[np.all(angles == 0, axis=1)] = np.nan
    return Emg2PoseRecording(emg, angles, time, float(attrs["sample_rate"]), attrs)


def load_recording(path: Path | str, member: str | None = None, mask_ik_failures: bool = True) -> Emg2PoseRecording:
    """HDF5 파일 하나를 읽는다. member 가 있으면 path 를 tar 로 보고 풀지 않고 스트림으로 연다."""
    if member is None:
        with h5py.File(path, "r") as f:
            return read_recording(f, mask_ik_failures)
    with tarfile.open(path) as tar, h5py.File(tar.extractfile(member), "r") as f:
        return read_recording(f, mask_ik_failures)


def make_windows(
    emg: np.ndarray, angles: np.ndarray, fs: float, win_s: float = 1.0, step_s: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(T, 채널) 신호를 창으로 자른다. 반환: (창 수, 창 길이, 채널) 모양의 emg·각도 뷰.

    step_s 가 없으면 겹치지 않는 창. 끝에 창보다 짧게 남는 구간은 버린다. 복사 없는 읽기 전용 뷰다.
    """
    if emg.shape[0] != angles.shape[0]:
        raise ValueError("emg 와 angles 의 시간 길이가 다르다")
    win = round(win_s * fs)
    step = win if step_s is None else round(step_s * fs)
    if emg.shape[0] < win:
        return np.empty((0, win, emg.shape[1]), emg.dtype), np.empty((0, win, angles.shape[1]), angles.dtype)
    emg_w = sliding_window_view(emg, win, axis=0)[::step]  # (창, 채널, 창 길이)
    ang_w = sliding_window_view(angles, win, axis=0)[::step]
    return emg_w.transpose(0, 2, 1), ang_w.transpose(0, 2, 1)
