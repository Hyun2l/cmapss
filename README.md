# CMAPSS RUL Prediction

NASA CMAPSS 데이터를 활용해 항공 엔진의 잔여수명(RUL)을 예측하는 머신러닝 프로젝트

## Setup

```bash
conda create -n cmapss python=3.11 -y
conda activate cmapss
pip install -r requirements.txt   # = pip install -e ".[dev]"
pip install -e ".[dl]"            # LSTM / MC Dropout 실험을 돌릴 때만
```

`pip install -e .`로 `src/cmapss` 패키지가 설치되므로 노트북/테스트 어디서든 `from cmapss... import ...`가 동작합니다.
VS Code / Jupyter 커널은 `cmapss` env를 선택.

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
  lstm.py                # LSTM + MC Dropout 불확실성 추정
  evaluate.py            # RMSE, PHM score, naive baseline, coverage
  viz.py                 # 시각화
scripts/
  run_experiment.py      # 베이스라인 vs LSTM vs MC Dropout 비교 → results/experiment_results.json
  seed_stability.py      # RF vs LSTM을 seed 5개로 반복 (결과가 우연인지 확인)
  diag_target_scaling.py # LSTM 학습 실패(상수 출력) 진단
results/                 # 실험 결과 JSON
tests/                   # pytest
```

## Workflow

1. `notebooks/02_preprocessing.ipynb` 실행 → `data/interim`, `data/processed` 생성
2. `notebooks/03_baseline_models.ipynb` 실행 → 베이스라인 성능 확인
3. `python scripts/run_experiment.py` → 베이스라인 + LSTM + MC Dropout 비교
4. `python scripts/seed_stability.py` → RF vs LSTM seed 5개 반복
5. 테스트: `pytest`

## Results (FD001)

**평가 기준**
- test = 엔진 100대 각각의 마지막 윈도우 1개.
- **capped**: 정답 RUL도 학습 라벨과 같이 125로 자른 값(문헌 비교 기준). **raw**: `RUL_FD001.txt` 원본(125 초과 엔진 11대에서 모델이 구조적으로 손해).
- val = 엔진 20대의 모든 슬라이딩 윈도우. PHM score는 합계라 샘플 수가 다른 val/test끼리 비교 X.
- 단일 실행은 seed 42 (`python scripts/run_experiment.py`).

| 모델 | val RMSE | test RMSE (capped) | test RMSE (raw) | PHM (capped) |
|---|---|---|---|---|
| naive (train RUL 평균) | — | 40.58 | 41.89 | — |
| Ridge | 18.34 | 17.73 | 19.03 | 605 |
| RandomForest | 12.28 | **12.65** | **13.90** | **260** |
| LSTM | **11.72** | 13.72 | 14.97 | 346 |
| LSTM + MC Dropout (100 samples) | — | 13.64 | 14.91 | 341 |

트리 모델 입력은 윈도우 요약 통계(마지막값·평균·기울기, 14×3=42개), LSTM 입력은 (30, 14) 시퀀스 그대로.

**seed 5개 반복** (`scripts/seed_stability.py`, capped)

| 모델 | val RMSE | test RMSE |
|---|---|---|
| RandomForest | 12.27 ± 0.04 | **12.64 ± 0.12** |
| LSTM | **11.86 ± 0.29** | 13.80 ± 0.48 |

- **LSTM은 5개 seed 모두에서 val은 이기고 test는 RandomForest에 진다.** FD001은 운전 조건과 고장 모드가 하나씩뿐이라, 기울기로 열화 속도를 넣어준 트리 모델이 충분히 강하다. 또 val(모든 윈도우, 건강 구간 비중 큼)과 test(엔진당 마지막 윈도우)는 RUL 분포가 달라, val 순위가 test 순위를 보장하지 않는다. LSTM은 seed에 따른 편차도 RF보다 4배 크다.
- **MC Dropout** 80% 예측구간의 실제 포함률은 **39%**(목표 80%). dropout이 만드는 모델 불확실성(epistemic)만 잡고 데이터 잡음(aleatoric)은 반영하지 못해 구간이 너무 좁다(과신). 그래도 예측 표준편차와 실제 절대오차의 상관이 **0.30**으로, 불확실성이 큰 곳에서 더 틀리는 경향은 잡는다. 보정이 필요하면 Conformal Prediction이 다음 단계.
- **구간별 LSTM 오차**(capped): 실제 RUL 0–50에서 RMSE **4.11**, 50–100에서 19.75, 100 이상에서 11.19. **고장이 임박한 구간에서 가장 정확**하며, 이는 예지정비에서 가장 중요한 구간이다. 50–100은 열화가 막 시작되는 전환 구간이라 가장 어렵다.

### 알려진 함정: 타깃 스케일
원본 RUL(평균 약 81)을 그대로 타깃으로 쓰면 LSTM이 입력과 무관한 상수를 출력 (예측 std = 0).
원인은 **출력 크기 불일치**다. 초기 출력은 0 근처인데, Adam은 스텝당 파라미터를 약 lr만큼만 움직여 출력을 81까지 끌어올리는 데 수천 스텝이 걸린다. 그동안 은닉 상태가 포화되어 입력 정보가 사라지는 현상 발생.

| 설정 (각 8 epoch) | val RMSE | 예측 평균 | 예측 std |
|---|---|---|---|
| A. 원본 타깃 | 63.48 | 32.1 | 0.00 |
| B. 원본 타깃 + 출력 bias를 평균으로 초기화 | 21.20 | 81.6 | 25.8 |
| C. 타깃 / 125 (채택) | **13.10** | 81.9 | 40.8 |

A는 40 epoch가 지나서야 평균 예측 수준(41.8)에 도달한다. B처럼 출력 시작점만 맞춰도 학습되므로 원인이 출력 크기라는 것을 확인했고, 출력·손실·기울기 크기를 모두 1 근처로 맞추는 C를 채택. `python scripts/diag_target_scaling.py`로 재현가능.

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
