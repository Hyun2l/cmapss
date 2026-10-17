"""RandomForest vs LSTM 결과가 seed 하나의 우연이 아닌지 확인 (seed 0~4).

실행: python scripts/seed_stability.py   (CPU 약 4분)
test 지표는 정답 RUL을 rul_cap으로 자른 기준.
"""
import numpy as np

from cmapss.config import load_config
from cmapss.evaluate import rmse
from cmapss.lstm import predict, train_lstm
from cmapss.models import sequences_to_tabular, train_random_forest

cfg = load_config()
P = cfg["paths"]["processed_dir"]
RUL_CAP = cfg["preprocess"]["rul_cap"]

load = lambda n: np.load(P / f"{n}.npy")
X_tr, y_tr = load("X_train"), load("y_train")
X_va, y_va = load("X_val"), load("y_val")
X_te, y_te = load("X_test"), np.minimum(load("y_test"), RUL_CAP)
Xtr_tab, Xva_tab, Xte_tab = map(sequences_to_tabular, (X_tr, X_va, X_te))

scores = {"RandomForest": [], "LSTM": []}
for seed in range(5):
    rf = train_random_forest(Xtr_tab, y_tr, seed=seed)
    scores["RandomForest"].append((rmse(y_va, rf.predict(Xva_tab)), rmse(y_te, rf.predict(Xte_tab))))
    lstm, _ = train_lstm(X_tr, y_tr, X_va, y_va, seed=seed, y_scale=float(RUL_CAP), verbose=False)
    scores["LSTM"].append((rmse(y_va, predict(lstm, X_va)), rmse(y_te, predict(lstm, X_te))))
    print(f"seed {seed}  RF val/test={scores['RandomForest'][-1][0]:.2f}/{scores['RandomForest'][-1][1]:.2f}"
          f"  LSTM val/test={scores['LSTM'][-1][0]:.2f}/{scores['LSTM'][-1][1]:.2f}", flush=True)

print()
for name, s in scores.items():
    s = np.array(s)
    print(f"{name:<13} val {s[:, 0].mean():.2f} ± {s[:, 0].std():.2f}   test {s[:, 1].mean():.2f} ± {s[:, 1].std():.2f}")
