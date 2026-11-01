"""test 엔진 수 n에 따라 1·2등 RMSE 차이의 흔들림이 1/sqrt(n)으로 줄어드는지 실측 (delta method 검증).

실행: python scripts/n_scaling.py [--subsets FD002 FD004] [--draws 4000]
train 엔진을 2-fold로 나눠 각 엔진에 '그 엔진을 안 본 모델'의 예측을 만든다. 엔진마다 test처럼
20개 지점에서 잘라 마지막 윈도우를 예측해 두고, n대를 비복원으로 뽑아(엔진당 자른 지점 하나) RMSE 차이를 잰다.
비교: 실측 표준편차 vs delta method SE(뽑을 때마다 계산한 값의 평균).
유한모집단 보정 sqrt((N-n)/(N-1))도 함께 기록하지만, 자르는 지점을 매번 새로 뽑으므로 모집단이 사실상 무한이라 적용되지 않는다.
"""
import argparse
import json

import numpy as np
import torch

from cmapss.config import load_config
from cmapss.data import FD001_DEAD_SENSORS, SENSOR_COLS, load_data
from cmapss.features import (add_rul_column, apply_regime_scaler, create_sequences, fit_regime_scaler)
from cmapss.lstm import predict, train_lstm
from cmapss.models import sequences_to_tabular, train_hist_gb, train_random_forest
from cmapss.protocol import delta_rmse_diff, rmse_np, truncated_windows

FEATURES = [c for c in SENSOR_COLS if c not in FD001_DEAD_SENSORS]
CUTS, WINDOW = 20, 30


def oof_predictions(subset, cfg, seed=0):
    """엔진별 (cut 20개) 정답과 out-of-fold 예측. 반환: y (N, CUTS), {model: (N, CUTS)}"""
    cap = cfg["preprocess"]["rul_cap"]
    df = add_rul_column(load_data(cfg["paths"]["raw_dir"] / f"train_{subset}.txt"), cap)
    units = np.random.default_rng(seed).permutation(df.unit.unique())
    folds = [units[: len(units) // 2], units[len(units) // 2:]]
    Y, P = {}, {"HistGB": {}, "RandomForest": {}, "LSTM": {}}
    for k in range(2):
        tr, ho = df[df.unit.isin(folds[1 - k])], df[df.unit.isin(folds[k])]
        sc = fit_regime_scaler(tr, FEATURES)
        tr, ho = apply_regime_scaler(tr, sc, FEATURES), apply_regime_scaler(ho, sc, FEATURES)
        X_tr, y_tr, _ = create_sequences(tr, FEATURES, WINDOW)
        X_ho, y_ho, u_ho = truncated_windows(ho, FEATURES, WINDOW, CUTS, np.random.default_rng(100 + k))
        tab_tr, tab_ho = sequences_to_tabular(X_tr), sequences_to_tabular(X_ho)
        p = {"HistGB": train_hist_gb(tab_tr, y_tr, seed=seed).predict(tab_ho),
             "RandomForest": train_random_forest(tab_tr, y_tr, seed=seed).predict(tab_ho)}
        # LSTM early stopping용 val: train fold 엔진의 20% (held-out fold는 쓰지 않는다)
        tu = np.random.default_rng(seed).permutation(tr.unit.unique())
        nv = len(tu) // 5
        X_a, y_a, _ = create_sequences(tr[tr.unit.isin(tu[nv:])], FEATURES, WINDOW)
        X_v, y_v, _ = create_sequences(tr[tr.unit.isin(tu[:nv])], FEATURES, WINDOW)
        net, _ = train_lstm(X_a, y_a, X_v, y_v, seed=seed, y_scale=float(cap), verbose=False)
        p["LSTM"] = predict(net, X_ho)
        for u in np.unique(u_ho):
            m = u_ho == u
            Y[u] = y_ho[m]
            for name in P:
                P[name][u] = p[name][m]
    us = sorted(Y)
    return np.array([Y[u] for u in us]), {n: np.array([P[n][u] for u in us]) for n in P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD002", "FD004"])
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config()
    out = cfg["paths"]["results_dir"] / "n_scaling"
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for subset in args.subsets:
        Y, P = oof_predictions(subset, cfg)
        N = len(Y)
        np.savez_compressed(out / f"{subset}_oof.npz", y=Y, **P)
        overall = {m: rmse_np(Y.ravel(), p.ravel()) for m, p in P.items()}
        rng = np.random.default_rng(0)
        result[subset] = {"N_engines": N, "overall_rmse": {m: round(v, 2) for m, v in overall.items()}, "pairs": {}}
        for a, b in [("HistGB", "RandomForest"), ("HistGB", "LSTM"), ("RandomForest", "LSTM")]:
            rows = []
            for n in [n for n in (25, 50, 75, 100, 150, 200) if n < N]:
                diffs, ses = [], []
                for _ in range(args.draws):
                    eng = rng.choice(N, n, replace=False)
                    cut = rng.integers(0, CUTS, n)
                    y, pa, pb = Y[eng, cut], P[a][eng, cut], P[b][eng, cut]
                    r = delta_rmse_diff(y, pa, pb)
                    diffs.append(r["diff"]); ses.append(r["se"])
                fpc = np.sqrt((N - n) / (N - 1))
                rows.append({"n": n, "empirical_sd": round(float(np.std(diffs)), 4),
                             "delta_se": round(float(np.mean(ses)), 4), "delta_se_fpc": round(float(np.mean(ses) * fpc), 4),
                             "mean_diff": round(float(np.mean(diffs)), 3)})
            ns, sd = np.array([r["n"] for r in rows]), np.array([r["empirical_sd"] for r in rows])
            ses = np.array([r["delta_se"] for r in rows])
            slope = float(np.polyfit(np.log(ns), np.log(ses), 1)[0])   # delta SE의 n 의존성 (기대 -0.5)
            result[subset]["pairs"][f"{a} vs {b}"] = {"rows": rows, "loglog_slope_delta_se": round(slope, 3),
                                                     "sd_over_fpc_se_median": round(float(np.median(
                                                         sd / np.array([r["delta_se_fpc"] for r in rows]))), 3)}
        json.dump(result, open(out / "summary.json", "w"), indent=2)
        print(subset, json.dumps(result[subset]), flush=True)


if __name__ == "__main__":
    main()
