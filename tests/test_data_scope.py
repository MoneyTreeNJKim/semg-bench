from pathlib import Path

import pandas as pd

from semg_bench.data_scope import (
    find_hyser_sessions,
    parse_id_list,
    select_hyser_sessions,
    write_csv_atomic,
)


def test_parse_id_list() -> None:
    assert parse_id_list("1-3,5,2") == ["1", "2", "3", "5"]
    assert parse_id_list("d387095792, 29ddab35d7") == ["d387095792", "29ddab35d7"]
    assert parse_id_list("all") is None
    assert parse_id_list(None) is None


def make_hyser_tree(root: Path) -> None:
    for name in ["subject01_session1", "subject01_session2", "subject02_session1", "subject10_session2"]:
        (root / "pr_dataset" / name).mkdir(parents=True)
    (root / "pr_dataset" / "subject03_notes.txt").write_text("x")


def test_find_and_select_hyser_sessions(tmp_path: Path) -> None:
    make_hyser_tree(tmp_path)
    found = find_hyser_sessions(tmp_path)
    assert found["name"].tolist() == ["subject01_session1", "subject01_session2", "subject02_session1", "subject10_session2"]
    sel = select_hyser_sessions(found, parse_id_list("1-2"), parse_id_list("1"))
    assert sel["name"].tolist() == ["subject01_session1", "subject02_session1"]
    assert len(select_hyser_sessions(found, None, None)) == 4


def test_write_csv_atomic_gzip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "sub" / "t.csv.gz"
    write_csv_atomic(df, path)
    assert path.read_bytes()[:2] == b"\x1f\x8b"  # gzip
    assert pd.read_csv(path).equals(df)
    assert not list(path.parent.glob("*.tmp"))
