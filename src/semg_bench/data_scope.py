"""데이터 범위 공통 처리: DATA_ROOT, 피험자·샤드 목록 인자, HYSER 세션 탐색, 원자적 CSV 저장."""

import os
import re
from pathlib import Path

import pandas as pd

HYSER_SESSION_RE = re.compile(r"^subject(\d+)_session(\d+)$")


def data_root() -> Path:
    """환경 변수 DATA_ROOT (태블릿 ./data, Colab /content/data). 없으면 ./data."""
    return Path(os.environ.get("DATA_ROOT", "./data"))


def parse_id_list(text: str | None) -> list[str] | None:
    """'1-3,5' 나 'a1b2,c3d4' 같은 목록 인자를 문자열 리스트로 바꾼다. 비었거나 'all' 이면 None(전체)."""
    if text is None or text.strip().lower() in ("", "all"):
        return None
    ids: list[str] = []
    for token in (t.strip() for t in text.split(",") if t.strip()):
        m = re.fullmatch(r"(\d+)-(\d+)", token)
        ids += [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)] if m else [token]
    return list(dict.fromkeys(ids))


def find_hyser_sessions(root: Path) -> pd.DataFrame:
    """root 아래에서 subjectNN_sessionM 디렉터리를 찾아 subject, session, path 표로 돌려준다."""
    rows = []
    for p in sorted(root.rglob("subject*_session*")):
        m = HYSER_SESSION_RE.match(p.name)
        if p.is_dir() and m:
            rows.append({"subject": int(m.group(1)), "session": int(m.group(2)), "name": p.name, "path": p})
    return pd.DataFrame(rows, columns=["subject", "session", "name", "path"])


def select_hyser_sessions(
    sessions: pd.DataFrame, subjects: list[str] | None, session_ids: list[str] | None
) -> pd.DataFrame:
    """피험자·세션 번호 목록(None 이면 전체)으로 세션 표를 거른다."""
    keep = pd.Series(True, index=sessions.index)
    if subjects is not None:
        keep &= sessions["subject"].isin([int(s) for s in subjects])
    if session_ids is not None:
        keep &= sessions["session"].isin([int(s) for s in session_ids])
    return sessions[keep].reset_index(drop=True)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """임시 파일에 쓴 뒤 이름을 바꿔, 중단돼도 반쯤 쓴 CSV 가 '완료'로 보이지 않게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip" if path.suffix == ".gz" else None)
    tmp.replace(path)
