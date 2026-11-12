"""Colab 에서 HYSER PR 전체(또는 일부) 세션을 돌린다. 끝난 세션은 건너뛰어 중단 후 이어서 돌 수 있다.

예: DATA_ROOT=/content/data uv run python scripts/run_hyser_all.py --subjects 1-20 --sessions 1,2 --seed 0
"""

from semg_bench.hyser_all import main

if __name__ == "__main__":
    raise SystemExit(main())
