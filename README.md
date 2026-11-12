# Colab 전체 실행
- 공통: 저장소 루트에서 `export DATA_ROOT=/content/data`, 결과를 보존하려면 `--out` 을 Drive 경로로.
- HYSER: `uv run python scripts/run_hyser_all.py --data <Drive>/TabDev/datasets/hyser --subjects all --sessions all --models lda --seed 0`
  (로지스틱 회귀는 세션당 약 4 분이라 필요할 때만 `--models lda,logreg`. 끝난 세션은 건너뛴다.)
- emg2pose: `uv run python scripts/run_emg2pose_shards.py --shards-dir <Drive>/TabDev/datasets/emg2poseshards --seed 0`
  (첫 줄의 `index.csv: 파일 열 …, 샤드 열 …` 확인. 끝난 녹화는 건너뛰고 `--stage extract|eval` 로 나눠 돌릴 수 있다.)
- emg2pose 기본 설정: 1 초 창·0.25 초 간격, log RMS + log WL, IK 실패 구간 제외(유효 90% 이상 창), 손별 릿지.
