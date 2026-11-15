"""Colab 에서 emg2pose 전체 샤드를 처리한다. Drive 의 index.csv 로 필요한 tar 만 읽고, 끝난 녹화는 건너뛴다.

예: DATA_ROOT=/content/data uv run python scripts/run_emg2pose_shards.py \
      --shards-dir /content/drive/MyDrive/TabDev/datasets/emg2poseshards --users all --seed 0
"""

from semg_bench.emg2pose_shards import main

if __name__ == "__main__":
    raise SystemExit(main())
