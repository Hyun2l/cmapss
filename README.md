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
  models.py              # 베이스라인 모델 (Ridge, RandomForest, HistGB)
  lstm.py                # LSTM + MC Dropout 불확실성 추정, 시퀀스 모델 공용 학습 루프
  cnn.py                 # 1D-CNN (시간축 합성곱 + flatten)
  protocol.py            # 검증 프로토콜, paired bootstrap, delta method 구간
  evaluate.py            # RMSE, PHM score, naive baseline, coverage
  viz.py                 # 시각화
scripts/
  run_experiment.py      # 베이스라인 vs LSTM vs MC Dropout 비교 → results/experiment_results.json
  seed_stability.py      # RF vs LSTM을 seed 5개로 반복 (결과가 우연인지 확인)
  diag_target_scaling.py # LSTM 학습 실패(상수 출력) 진단
  judgement_experiment.py, judgement_analysis.py  # FD001–FD004 × seed 10 → results/judgement/
  delta_check.py         # delta method 반폭 vs bootstrap 검산
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

## 평가 방법론: test 세트는 얼마나 작은 차이를 구분하나

C-MAPSS test는 엔진당 마지막 윈도우 하나라 FD001/FD003은 100개, FD002/FD004는 259/248개 점수의 평균이다.
두 모델의 test RMSE가 0.3 차이 날 때 그게 모델 차이인지 test 엔진 표본의 우연인지 확인하려고, 같은 프로토콜로 FD001–FD004 × seed 10개 × 모델 5개(Ridge, RandomForest, HistGB, LSTM, 1D-CNN)를 돌렸다.

- 프로토콜 (모든 서브셋 동일): 표준 센서 14개, 윈도우 30, RUL cap 125, 운영조건이 여럿이면 조건별 MinMax(train 엔진에서만 fit), 엔진 단위 train/val 분할
- `python scripts/judgement_experiment.py --subsets FD001 FD002 FD003 FD004 --seeds 10` (1D-CNN은 `--models CNN --append`로 추가) → `python scripts/judgement_analysis.py` → `python scripts/delta_check.py`

**test 1·2등 차이의 95% 구간** (seed 10개 중앙값, capped)

| | 엔진 수 | 평균 1·2등 차이 | 그 차이의 seed 간 std | 95% 구간 반폭 | 구간이 0을 포함한 seed | 반폭 1.0에 필요한 엔진 | 반폭 0.5에 필요한 엔진 |
|---|---|---|---|---|---|---|---|
| FD001 | 100 | 0.29 | 0.23 | 1.07 | 10/10 | 112 | 448 |
| FD002 | 259 | 0.32 | 0.18 | 1.02 | 10/10 | 258 | 1030 |
| FD003 | 100 | 0.37 | 0.29 | 1.53 | 10/10 | 249 | 994 |
| FD004 | 248 | 0.48 | 0.30 | 0.94 | 9/10 | 222 | 887 |

- **test 1·2등은 거의 항상 통계적으로 구분되지 않는다.** 이 test 세트들은 약 1 RMSE(FD003은 1.5)보다 작은 개선을 우연과 구분하지 못한다. 1등 모델 seed 평균과 2등 모델 seed 평균의 차이도 FD001 0.03, FD003 0.00(RF와 HistGB)으로 이보다 훨씬 작다.
- **seed 간 std는 이 불확실성을 보여주지 못한다.** 차이의 seed std는 0.2–0.3인데 test 표본에서 오는 표준오차는 0.5–0.8(=반폭/1.96)이다. 학습 seed를 바꿔도 test 엔진은 그대로이기 때문이다. "± std"만으로 개선을 주장하면 불확실성을 2–3배 작게 보게 된다.
- 1등 모델은 데이터마다 다르다: FD001 RF/HistGB(구분 불가), FD002·FD003 HistGB, FD004 LSTM(seed 10개 중 9개). 조건 6개 + 고장모드 2개인 FD004에서만 시퀀스 모델이 일관되게 이긴다. 1D-CNN은 어느 서브셋에서도 1등을 못 했지만(FD001 13.68, FD004 15.88) FD004에서는 트리 모델(HistGB 16.01, RF 16.62)보다 나았다.

**사전 등록 판정과 사후 결과.** 판정 규칙은 실험 전에 `scripts/judgement_analysis.py`에 고정했다. 기준 A: test 1·2등 95% 구간 반폭 중앙값이 1.0 이상(이보다 작은 개선을 구분 못 함). 기준 B: test처럼 자른 val이 모델 선택의 regret을 표준오차 2배 넘게 줄임. 서브셋 2개 이상에서 A 또는 B가 성립하면 논문화(GO).

| | 모델 4개 (사전 등록) | 모델 5개 (1D-CNN 추가, 사후) |
|---|---|---|
| 기준 A 반폭 FD001/002/003/004 | 1.07 / 1.06 / 1.53 / 1.06 → **4/4** | 1.07 / 1.02 / 1.53 / 0.94 → **3/4** |
| 기준 B regret 개선 ± SE | −0.13±0.12 / 0.09±0.05 / 0.19±0.11 / −0.16±0.16 → **0/4** | −0.41±0.27 / 0.04±0.07 / 0.23±0.11 / −0.19±0.18 → **1/4** |
| 판정 | GO (임계값을 1.1로 올리면 A 1/4) | GO |

규칙상으로는 GO였지만 임계값 바로 위였고, 모델 하나를 더하자 두 기준이 모두 바뀌었다. 판정이 이 정도로 흔들린다는 것 자체가 결론이라고 보고 논문화하지 않았다. 사전 등록 결과는 커밋 `Add FD002-FD004 raw data and judgement experiment results`의 `summary.json`에 있다.

### 닫힌 식: delta method

구간을 bootstrap 없이 식으로 구하면 무엇이 반폭을 결정하는지 보인다. 엔진 $i$의 제곱오차를 $a_i=e_{a,i}^2,\ b_i=e_{b,i}^2$, $R_a=\sqrt{\bar a}$라 하면 $\text{RMSE}_a-\text{RMSE}_b=g(\bar a,\bar b)$, $g(u,v)=\sqrt u-\sqrt v$이고, 1차 근사(기울기 $1/2R_a,\ -1/2R_b$)로

$$\mathrm{SE}(\text{RMSE}_a-\text{RMSE}_b)\approx\frac{1}{\sqrt n}\,\mathrm{sd}\!\left(\frac{a_i}{2R_a}-\frac{b_i}{2R_b}\right)$$

엔진끼리 독립이라는 가정만 쓴다. 오차가 평균 0, 표준편차 $\sigma$, 두 모델 오차 상관 $\rho$인 정규분포라면 $\mathrm{Var}(e^2)=2\sigma^4$, $\mathrm{Cov}(e_a^2,e_b^2)=2\rho^2\sigma^4$를 넣어

$$\mathrm{SE}\approx\sigma\sqrt{\frac{1-\rho^2}{n}}$$

오차 자체의 차이 $e_a-e_b$의 표준편차 $\sigma\sqrt{2(1-\rho)}$와 헷갈리기 쉽지만, RMSE는 제곱오차의 함수라 $\rho$가 아니라 $\rho^2$가 들어간다. $\rho=0, 0.5, 0.9$ 시뮬레이션(n=100)에서 실제 SE는 1.21/1.03/0.52, 이 식은 1.20/1.04/0.52, $\sqrt{2(1-\rho)}$ 식은 1.70/1.20/0.54다.

- 반폭은 **모델 오차 크기 $\sigma$에 비례**하고, **엔진 수의 제곱근에 반비례**하며, **두 모델이 얼마나 다르게 틀리는지($\sqrt{1-\rho^2}$)**에 좌우된다. FD002/FD004는 엔진이 2.5배 많지만 $\sigma$가 커서 반폭이 FD001과 비슷하다. 같은 test 세트 안에서도 비교하는 쌍에 따라 해상도가 달라진다. FD003에서 RF vs HistGB($\rho\approx0.88$)는 반폭 1.51, RF vs 1D-CNN($\rho\approx0.67$)은 3.18이다. 같은 계열(트리끼리, 시퀀스 모델끼리)은 비슷하게 틀려서 작은 차이도 보이고, 계열이 다르면 잡음이 커진다.
- 실측 검산 (`scripts/delta_check.py`): delta 반폭 1.06/1.00/1.58/0.95 vs bootstrap 1.07/1.02/1.53/0.94. 정규 근사는 0.94/0.85/1.33/0.78로 11–18% 작다. RUL 오차의 첨도가 4.4–5.0(정규분포는 3)으로 꼬리가 두꺼워서다. 그래서 필요 엔진 수는 정규 근사가 아니라 실측 delta 반폭에서 $n'=n\,(h/h')^2$로 역산했다.
- 구현: `src/cmapss/protocol.py`의 `delta_rmse_diff`, `gaussian_half_width`, `engines_needed`. 정규 오차 시뮬레이션에서 bootstrap·반복 실험과 맞는지 `tests/test_protocol.py`로 확인한다.

### 시도했지만 지지되지 않은 것

val 순위가 test 순위와 어긋나는 원인이 "val은 모든 윈도우, test는 엔진당 마지막 윈도우"라는 분포 차이라고 보고, val도 test처럼 엔진을 임의 지점에서 잘라 평가해 봤다(`protocol.truncated_windows`). 고른 모델의 test 손해(regret)는 FD002/FD003에서 줄고 FD001/FD004에서 늘었다. 표준오차 2배를 넘은 건 FD003(0.36 → 0.13) 하나뿐이고, 모델 4개일 때는 그마저 넘지 못했다. 위 결과대로 후보 모델들이 test 해상도 안에서 비슷하면 어떤 검증 방식으로 골라도 차이가 잘 드러나지 않는다.

**한계:** 모델 5개 모두 문헌 최고 수준보다 약 1 RMSE 약하다(1D-CNN을 더해도 줄지 않음). 문헌 수준 모델끼리는 $\sigma$가 작고 $\rho$가 높아 반폭이 더 작을 수 있다. LSTM/CNN early stopping은 모든-윈도우 val 기준으로 고정, test 엔진은 공식 세트 하나로 고정.

## 개발 방식

EDA와 베이스라인(2026년 9월까지)은 직접 작성했다. 2026-10-14 프로젝트 구조 정리와 2026-10-17 LSTM·MC Dropout부터는 Claude(Anthropic)를 본격적으로 사용해 코드와 문서를 작성했다. 실험 질문, 판정 기준의 사전 고정, 결과 해석과 결정(예: 경계선 결과에서 논문화를 접은 것)은 직접 했고, 모든 수치는 저장소의 스크립트로 재현된다.

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
