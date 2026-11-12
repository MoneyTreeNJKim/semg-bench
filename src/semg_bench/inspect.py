"""HYSER(WFDB) 와 emg2pose(HDF5) 샘플 파일 하나씩을 열어 구조를 표로 정리한다.

실행: DATA_ROOT=./data uv run python -m semg_bench.inspect
결과: results/inspect.md, results/inspect.csv
"""

import os
import tarfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb

HYSER_SESSION = "hyser/subject01_session1"
EMG2POSE_DIR = "emg2pose"
RESULTS_DIR = Path("results")


def read_labels(path: Path) -> list[int]:
    """쉼표로 구분된 HYSER 라벨 파일을 정수 리스트로 읽는다."""
    return [int(x) for x in path.read_text().strip().split(",")]


def label_summary(labels: list[int]) -> str:
    """라벨 개수, 클래스 범위, 반복 횟수를 한 줄로 요약한다."""
    counts = Counter(labels)
    mode = Counter(counts.values()).most_common(1)[0][0]
    odd = {k: v for k, v in sorted(counts.items()) if v != mode}
    text = f"{len(labels)}개, 동작 {min(counts)}~{max(counts)} ({len(counts)}클래스), 동작당 {mode}회"
    return text + (f", 예외 {odd}" if odd else "")


def inspect_hyser(session_dir: Path, task: str) -> dict[str, object]:
    """HYSER 과제(dynamic/maintenance) 의 첫 레코드와 라벨 파일을 요약한다."""
    names = sorted(p.stem for p in session_dir.glob(f"{task}_preprocess_sample*.hea"))
    rec = wfdb.rdrecord(str(session_dir / f"{task}_preprocess_sample1"))
    lens = Counter(wfdb.rdheader(str(session_dir / n)).sig_len for n in names)
    grids = sorted({name.split("-")[0] for name in rec.sig_name})
    return {
        "dataset": "HYSER",
        "file": f"{task}_preprocess_sample1 (.hea/.dat)",
        "n_channels": rec.n_sig,
        "fs_hz": rec.fs,
        "n_samples": rec.sig_len,
        "duration_s": rec.sig_len / rec.fs,
        "signal": f"p_signal {rec.p_signal.shape} {rec.p_signal.dtype}, 단위 {set(rec.units)}, "
        f"채널명 {rec.sig_name[0]}..{rec.sig_name[-1]} (전극 그리드 {grids})",
        "label": f"label_{task}.txt: " + label_summary(read_labels(session_dir / f"label_{task}.txt")),
        "notes": f"레코드 {len(names)}개(raw/preprocess 쌍), 길이 분포 {dict(lens)}",
    }


def read_timeseries(f: h5py.File) -> tuple[np.ndarray, dict[str, object], list]:
    """HDF5 에서 emg2pose/timeseries 배열, 그룹 속성, dtype 설명을 읽는다."""
    group = f["emg2pose"]
    return group["timeseries"][:], dict(group.attrs), group["timeseries"].dtype.descr


def read_emg2pose(root: Path) -> tuple[np.ndarray, dict[str, object], list, str]:
    """emg2pose 첫 HDF5 를 읽는다. 풀린 .hdf5 가 없으면 tar 안에서 스트림으로 연다."""
    files = sorted(root.rglob("*.hdf5"))
    if files:
        with h5py.File(files[0], "r") as f:
            return *read_timeseries(f), files[0].name
    with tarfile.open(min(root.glob("*.tar"))) as tar:
        member = min((m for m in tar.getmembers() if m.name.endswith(".hdf5")), key=lambda m: m.name)
        with h5py.File(tar.extractfile(member), "r") as f:
            return *read_timeseries(f), Path(member.name).name


def inspect_emg2pose(root: Path) -> dict[str, object]:
    """emg2pose HDF5 하나의 timeseries 구조와 메타데이터를 요약한다."""
    data, attrs, descr, name = read_emg2pose(root)
    emg, angles, t = data["emg"], data["joint_angles"], data["time"]
    fs = float(attrs["sample_rate"])
    return {
        "dataset": "emg2pose",
        "file": name,
        "n_channels": emg.shape[1],
        "fs_hz": fs,
        "n_samples": len(data),
        "duration_s": round(len(data) / fs, 2),
        "signal": f"emg2pose/timeseries 구조체 배열 {descr}; "
        f"emg {emg.shape} {emg.dtype}, 범위 [{emg.min():.3g}, {emg.max():.3g}]",
        "label": f"joint_angles {angles.shape} {angles.dtype} (관절 20개, 매 샘플), "
        f"범위 [{np.nanmin(angles):.3f}, {np.nanmax(angles):.3f}], NaN {int(np.isnan(angles).sum())}개; "
        f"time 간격 중앙값 {np.median(np.diff(t)):.6f}s",
        "notes": ", ".join(f"{k}={attrs[k]}" for k in ("side", "stage", "split", "user", "num_channels")),
    }


def to_markdown(df: pd.DataFrame) -> str:
    """tabulate 의존 없이 DataFrame 을 Markdown 표로 바꾼다."""
    cells = df.astype(str).map(lambda s: s.replace("|", "\\|"))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    data_root = Path(os.environ.get("DATA_ROOT", "./data"))
    rows = [
        inspect_hyser(data_root / HYSER_SESSION, "dynamic"),
        inspect_hyser(data_root / HYSER_SESSION, "maintenance"),
        inspect_emg2pose(data_root / EMG2POSE_DIR),
    ]
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "inspect.csv", index=False)
    (RESULTS_DIR / "inspect.md").write_text("# 데이터 구조 요약\n\n" + to_markdown(df))
    print(df[["dataset", "file", "n_channels", "fs_hz", "n_samples", "duration_s"]].to_string(index=False))
    print(f"저장: {RESULTS_DIR / 'inspect.md'}, {RESULTS_DIR / 'inspect.csv'}")


if __name__ == "__main__":
    main()
