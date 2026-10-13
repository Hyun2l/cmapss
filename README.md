# CMAPSS RUL Prediction

NASA CMAPSS 데이터를 활용해 항공 엔진의 잔여수명(RUL)을 예측하는 머신러닝 프로젝트입니다.

## Setup

```bash
conda create -n cmapss python=3.11 -y
conda activate cmapss
pip install -r requirements.txt   # = pip install -e ".[dev]"
```

`pip install -e .`로 `src/cmapss` 패키지가 설치되므로 노트북/테스트 어디서든 `from cmapss... import ...`가 동작합니다.
VS Code / Jupyter 커널은 `cmapss` env를 선택하세요.

## Project structure

```
configs/config.yaml      # 데이터셋, 경로, 전처리 하이퍼파라미터 (단일 진실 공급원)
data/raw/                # 원본 CMAPSS txt (git 추적)
data/interim/            # RUL 라벨링된 DataFrame (생성물, git 제외)
data/processed/          # 모델 입력 배열 .npy (생성물, git 제외)
notebooks/
  01_EDA.ipynb
  02_preprocessing.ipynb # interim/processed 생성
  03_baseline_models.ipynb
src/cmapss/
  config.py              # load_config(), raw_files()
  data.py                # 컬럼 정의, 로더
  features.py            # 죽은 센서, RUL 라벨, split, 스케일링, 윈도우
  models.py              # 베이스라인 모델 (Ridge, RandomForest)
  evaluate.py            # RMSE, PHM score, naive baseline
  viz.py                 # 시각화
tests/                   # pytest
```

## Workflow

1. `notebooks/02_preprocessing.ipynb` 실행 → `data/interim`, `data/processed` 생성
2. `notebooks/03_baseline_models.ipynb` 실행 → 베이스라인 성능 확인
3. 테스트: `pytest`

## Dataset overview (sensor descriptions from the paper)

| Column | Symbol | Description |
|---|---|---|
| sensor_1 | T2 | Fan inlet temperature (팬 입구 온도) |
| sensor_2 | T24 | LPC outlet temperature (저압압축기 출구 온도) |
| sensor_3 | T30 | HPC outlet temperature (고압압축기 출구 온도) |
| sensor_4 | T50 | LPT outlet temperature (저압터빈 출구 온도) |
| sensor_5 | P2 | Fan inlet pressure (팬 입구 압력) |
| sensor_6 | P15 | Bypass-duct pressure (바이패스 덕트 압력) |
| sensor_7 | P30 | HPC outlet pressure (고압압축기 출구 압력) |
| sensor_8 | Nf | Physical fan speed (팬 물리적 회전속도) |
| sensor_9 | Nc | Physical core speed (코어 물리적 회전속도) |
| sensor_10 | epr | Engine pressure ratio (엔진 압력비) |
| sensor_11 | Ps30 | HPC outlet static pressure (고압압축기 출구 정압) |
| sensor_12 | phi | Fuel flow / Ps30 (연료유량 대 정압비) |
| sensor_13 | NRf | Corrected fan speed (보정 팬속도) |
| sensor_14 | NRc | Corrected core speed (보정 코어속도) |
| sensor_15 | BPR | Bypass ratio (바이패스비) |
| sensor_16 | farB | Burner fuel-air ratio (연소기 연료-공기비) |
| sensor_17 | htBleed | Bleed enthalpy (블리드 엔탈피) |
| sensor_18 | Nf_dmd | Demanded fan speed (요구/설정 팬속도) |
| sensor_19 | PCNfR_dmd | Demanded corrected fan speed (요구/설정 보정 팬속도) |
| sensor_20 | W31 | HPT coolant bleed (고압터빈 냉각공기 블리드) |
| sensor_21 | W32 | LPT coolant bleed (저압터빈 냉각공기 블리드) |
