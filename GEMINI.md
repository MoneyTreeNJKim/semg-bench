# 프로젝트 개요
- 목적: 공개 sEMG 데이터셋(emg2pose, HYSER)으로 손동작 분류와 손 자세 회귀 모델의 베이스라인을 
만들고 개선한다.
- 환경: Python 3.12, uv. 태블릿(CPU, 샘플 데이터)에서 개발하고 전체 데이터는 Colab(GPU)에서 
돌린다.
# 데이터
- data/emg2pose/: 미니 데이터셋(HDF5). 전체(431GiB)는 Google Drive TabDev/datasets/emg2poseshards/ 에 2GB tar 샤드 + index.csv 로 있다.
- data/hyser/: HYSER PR 데이터셋 피험자 1 명 세션 1(WFDB .hea/.dat). 전체(142.8GB)는 Drive 
TabDev/datasets/hyser/ 에 PhysioNet 구조 그대로 있다.
- HDF5 와 WFDB 의 내부 구조는 하드코딩하지 말고 h5py, wfdb 로 먼저 열어 확인한 뒤 코드를 쓴다.
- 모든 로더와 스크립트는 데이터 루트를 환경 변수 DATA_ROOT 로 받는다(태블릿: ./data, Colab: 
/content/data).
# 자주 쓰는 명령
- 테스트: uv run pytest -q
- 린트: uv run ruff check .
- 베이스라인: uv run python -m semg_bench.hyser_baseline --data $DATA_ROOT/hyser
# 코드 규칙
- - - 무새답변작위성은  의·존주석성 추가 전에 ·seed 커밋인자로 고정한다 메시지는 한국어반드시 물어본다. . 결과는 타입 힌트 .
results/ 필수. 에 함수는 CSV 로 저장하고 50 줄 이하.
콘솔에도 요약한다.
# 금지
- data/ 수정·삭제, git push, 테스트를 통과시키려고 테스트 자체를 약화시키는 것
# 작업 방식
- 파일을 수정하기 전에 계획을 3 줄 이내로 먼저 제시한다.
- 작업 후 테스트를 실행해 결과를 보고한다.
