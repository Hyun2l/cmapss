"""
피처 엔지니어링: 죽은 센서 판별(EDA) + RUL 라벨링 + 슬라이딩 윈도우 생성(전처리).

순서(반드시 지킬 것, data leakage 방지):
  RUL 라벨링 -> split_units(unit 단위 분리) -> fit_scaler(train에만) ->
  apply_scaler(val/test는 transform만) -> create_sequences
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from cmapss.data import SENSOR_COLS


# ------------------------------------------------------------------
# EDA: 죽은 센서 판별 (Phase 1)
# ------------------------------------------------------------------
def compute_sensor_variance(df: pd.DataFrame) -> pd.Series:
    """센서별 표준편차를 오름차순 정렬해서 반환."""
    return df[SENSOR_COLS].std().sort_values()


def identify_dead_sensors(sensor_std: pd.Series, threshold: float) -> list:
    """threshold 미만인 센서를 '죽은 센서'로 분류.

    threshold=0.01은 FD001 실제 데이터에서 std 정렬 -> 로그스케일 시각화로
    gap 확인 -> gap 배율 계산 -> 물리적 타당성 검토(고정 비행조건/제어
    목표값) -> 시계열 재검증의 5단계를 거쳐 확정한 값.
    죽은 센서군 (최대 0.0014)과 활성 센서군(최소 0.0375) 사이에 27배 이상의 명확한 간극 존재
    """
    return sensor_std[sensor_std < threshold].index.tolist()


# ------------------------------------------------------------------
# 전처리: RUL 라벨링 (Phase 2)
# ------------------------------------------------------------------
def get_unit_lifetimes(df: pd.DataFrame) -> pd.Series:
    """각 unit의 최대 cycle = 그 엔진의 전체 수명(고장 시점)."""
    return df.groupby("unit")["cycle"].max()


def add_rul_column(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """Piecewise Linear RUL 라벨 생성.

    rul_cap보다 큰 구간(아직 정상 상태로 보는 구간)은 전부 rul_cap 값으로
    고정. 순수 선형 RUL은 "아직 열화 신호가 전혀 없는 구간"까지
    모델이 억지로 정밀하게 구분하려다 성능이 떨어지기에 상한을 둠.
    rul_cap=125는 후속 연구들에서 실험적으로 가장 좋은 성능을 낸다고
    반복 보고된 표준값(보통 120~130 범위)사용 .
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul_raw = max_cycle - df["cycle"]
    df["RUL"] = rul_raw.clip(upper=rul_cap)
    return df


# ------------------------------------------------------------------
# 전처리: Train/Val Split (Phase 2)
# ------------------------------------------------------------------
def split_units(df: pd.DataFrame, val_ratio: float, seed: int):
    """unit(엔진) 단위로 train/val 분리 — data leakage 방지의 핵심.

    행 단위로 무작위 분리하면 같은 엔진의 인접 cycle(예: cycle 50, 51)이
    센서값이 거의 동일한 채로 train/val에 각각 섞여 들어가, val 성능이
    허위로 좋게 나옴. 따라서, 엔진 전체를 통째로 한쪽에 배정.
    """
    rng = np.random.RandomState(seed)
    units = df["unit"].unique()
    rng.shuffle(units)
    n_val = int(len(units) * val_ratio)
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])
    train_df = df[df["unit"].isin(train_units)].copy()
    val_df = df[df["unit"].isin(val_units)].copy()
    return train_df, val_df, sorted(train_units), sorted(val_units)


# ------------------------------------------------------------------
# 전처리: 정규화 (Phase 2)
# ------------------------------------------------------------------
def fit_scaler(train_df: pd.DataFrame, feature_cols: list) -> MinMaxScaler:
    """train 데이터에만 fit. val/test에는 절대 fit하지 않음 (data leakage 방지)."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ------------------------------------------------------------------
# 전처리: 슬라이딩 윈도우 (Phase 2)
# ------------------------------------------------------------------
def create_sequences(df: pd.DataFrame, feature_cols: list, window_size: int,
                      label_col: str = "RUL"):
    """엔진별로 슬라이딩 윈도우 생성 (train/val 용 — 한 엔진에서 여러 윈도우).

    엔진 길이가 192cycle, window_size=30이면 [0:30],[1:31],...로 163개
    윈도우가 나옴. 같은 엔진의 여러 시점을 최대한 학습에 활용하기 위함.
    """
    X, y, unit_ids = [], [], []
    for unit, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        data = group[feature_cols].values
        labels = group[label_col].values
        n = len(group)
        if n < window_size:
            continue
        for start in range(0, n - window_size + 1):
            end = start + window_size
            X.append(data[start:end])
            y.append(labels[end - 1])
            unit_ids.append(unit)
    return np.array(X), np.array(y), np.array(unit_ids)


def create_test_sequences(test_df: pd.DataFrame, feature_cols: list, window_size: int):
    """테스트 데이터: 엔진당 마지막 window_size 사이클 1개만 사용.

    실전 시나리오("지금까지의 데이터로 남은 수명 예측")에서는 그 엔진의
    가장 최근 구간 하나만 있으면 되므로, train/val과 달리 윈도우를
    여러 개 만들지 않음. 길이가 window_size보다 짧으면 첫 행을
    반복해서 패딩.
    """
    X, unit_ids = [], []
    for unit, group in test_df.groupby("unit"):
        group = group.sort_values("cycle")
        data = group[feature_cols].values
        n = len(data)
        if n >= window_size:
            window = data[-window_size:]
        else:
            pad_len = window_size - n
            pad = np.repeat(data[0:1], pad_len, axis=0)
            window = np.vstack([pad, data])
        X.append(window)
        unit_ids.append(unit)
    return np.array(X), np.array(unit_ids)
