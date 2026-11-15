"""Colab 에서 HYSER 피험자 간 일반화(LOSO)와 정규화·증강 비교를 돌린다. 끝난 세션·겹은 건너뛴다.

예: uv run python scripts/run_hyser_loso.py --data <Drive>/TabDev/datasets/hyser --subjects all --sessions all \
      --out <Drive>/results/hyser_loso --seed 0
"""

from semg_bench.hyser_loso import main

if __name__ == "__main__":
    raise SystemExit(main())
