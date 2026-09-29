"""
1. 데이터 로드 및 컬럼 정의
2. 컬럼명/센서명 정의
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from requests import head

# RUL = Entire life of the engine (The last cycle) - Current cycle

# 그래프에서 마이너스(-) 기호가 깨지는 문제
plt.rcParams['axes.unicode_minus'] = False

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op_setting1","op_setting2","op_setting3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1,22)] # 6th -26th columns (n=21)
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

SENSOR_NAMES = {
    "sensor_1": "T2 (Fan inlet temp, R)",
    "sensor_2": "T24 (LPC outlet temp, R)",
    "sensor_3": "T30 (HPC outlet temp, R)",
    "sensor_4": "T50 (LPT outlet temp, R)",
    "sensor_5": "P2 (Fan inlet pressure, psia)",
    "sensor_6": "P15 (Bypass-duct pressure, psia)",
    "sensor_7": "P30 (HPC outlet pressure, psia)",
    "sensor_8": "Nf (Physical fan speed, rpm)",
    "sensor_9": "Nc (Physical core speed, rpm)",
    "sensor_10": "epr (Engine pressure ratio)",
    "sensor_11": "Ps30 (HPC outlet static pressure, psia)",
    "sensor_12": "phi (Fuel flow / Ps30, pps/psi)",
    "sensor_13": "NRf (Corrected fan speed, rpm)",
    "sensor_14": "NRc (Corrected core speed, rpm)",
    "sensor_15": "BPR (Bypass ratio)",
    "sensor_16": "farB (Burner fuel-air ratio)",
    "sensor_17": "htBleed (Bleed enthalpy)",
    "sensor_18": "Nf_dmd (Demanded fan speed, rpm)",
    "sensor_19": "PCNfR_dmd (Demanded corrected fan speed, rpm)",
    "sensor_20": "W31 (HPT coolant bleed, lbm/s)",
    "sensor_21": "W32 (LPT coolant bleed, lbm/s)",
}

# Phase 1에서 threshold=0.01로 확정된 FD001의 죽은 센서 7개.
FD001_DEAD_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

def load_data(filepath : str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=r"\s+", header=None) # header not exists, no comma, (s+) all lengths of spaces would be separated since - sign in the dataset
    df = df.iloc[:, :len(ALL_COLS)]
    df.columns = ALL_COLS
    return df

def load_rul_labels(filepath: str):
    """RUL_FD00X.txt (엔진당 정답 RUL 1개씩, 헤더 없음)을 1차원 배열로 로드."""
    return pd.read_csv(filepath, sep=r"\s+", header=None).iloc[:, 0].values


