"""판정 실험: 검증 방식이 모델 선택을 맞히는가, test가 모델 차이를 구분할 수 있는가.

실행:
  python scripts/judgement_experiment.py --quick                  # FD001, seed 2개, 동작 확인 (수 분)
  python scripts/judgement_experiment.py --subsets FD001 FD002 FD003 FD004 --seeds 10   # 본 실험

각 (서브셋, seed)마다 엔진 단위로 train/val을 나누고 모델 4개를 한 번씩 학습한 뒤,
같은 모델을 두 가지 검증 방식과 test로 채점한다. 판정은 scripts/judgement_analysis.py.

고정한 프로토콜 (모든 서브셋 동일):
  - 센서 14개(표준 세트), 윈도우 30, 학습 라벨 RUL cap 125
  - 운영조건이 여러 개면 조건별 MinMax(train 엔진에서만 fit), 하나면 전역 MinMax
  - LSTM early stopping은 '모든 윈도우' 검증으로 고정 (threat to validity로 보고)
  - test는 엔진당 마지막 윈도우, 지표는 정답 cap 기준과 원본 기준 둘 다 저장
"""
import argparse
import json
import time

import numpy as np

from cmapss.config import load_config
from cmapss.data import FD001_DEAD_SENSORS, SENSOR_COLS, load_data, load_rul_labels
from cmapss.features import (add_rul_column, apply_regime_scaler, create_sequences,
                             create_test_sequences, fit_regime_scaler, regime_keys, split_units)
from cmapss.lstm import predict, train_lstm
from cmapss.models import sequences_to_tabular, train_hist_gb, train_random_forest, train_ridge
from cmapss.protocol import rmse_np, truncated_windows

FEATURES = [c for c in SENSOR_COLS if c not in FD001_DEAD_SENSORS]   # 표준 14개


def run_one(subset, seed, cfg, quick, out_preds):
    raw = cfg["paths"]["raw_dir"]
    pp = cfg["preprocess"]
    cap, window = pp["rul_cap"], pp["window_size"]
    for f in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
        if not (raw / f).exists():
            raise FileNotFoundError(f"{raw / f} 없음 — CMAPSSData.zip의 {subset} 파일을 data/raw/에 넣으세요")

    train = add_rul_column(load_data(raw / f"train_{subset}.txt"), cap)
    test = load_data(raw / f"test_{subset}.txt")
    y_raw = load_rul_labels(raw / f"RUL_{subset}.txt").astype(float)
    y_cap = np.minimum(y_raw, cap)
    n_regimes = regime_keys(train).nunique()

    tr, va, _, _ = split_units(train, pp["val_ratio"], seed)
    scalers = fit_regime_scaler(tr, FEATURES)
    tr_s, va_s, te_s = (apply_regime_scaler(d, scalers, FEATURES) for d in (tr, va, test))

    X_tr, y_tr, _ = create_sequences(tr_s, FEATURES, window)
    X_va, y_va, _ = create_sequences(va_s, FEATURES, window)
    X_vt, y_vt, _ = truncated_windows(va_s, FEATURES, window, n_per_engine=5,
                                      rng=np.random.default_rng(seed))
    X_te, _ = create_test_sequences(te_s, FEATURES, window)

    tabs = [sequences_to_tabular(x) for x in (X_tr, X_va, X_vt, X_te)]
    fits = {
        "Ridge": lambda: train_ridge(tabs[0], y_tr),
        "RandomForest": lambda: train_random_forest(tabs[0], y_tr, seed=seed,
                                                    **({"n_estimators": 50} if quick else {})),
        "HistGB": lambda: train_hist_gb(tabs[0], y_tr, seed=seed),
    }
    rows = []
    for name, fit in fits.items():
        t0 = time.time()
        m = fit()
        preds = [m.predict(t) for t in tabs[1:]]
        rows.append((name, preds, time.time() - t0, None))

    t0 = time.time()
    lstm, info = train_lstm(X_tr, y_tr, X_va, y_va, seed=seed, y_scale=float(cap), verbose=False,
                            **({"max_epochs": 8, "patience": 3} if quick else {}))
    rows.append(("LSTM", [predict(lstm, x) for x in (X_va, X_vt, X_te)], time.time() - t0,
                 info["epochs_run"]))

    records = []
    for name, (p_va, p_vt, p_te), sec, epochs in rows:
        records.append({
            "subset": subset, "seed": seed, "model": name, "n_regimes": int(n_regimes),
            "val_all": rmse_np(y_va, p_va), "val_trunc": rmse_np(y_vt, p_vt),
            "test_cap": rmse_np(y_cap, p_te), "test_raw": rmse_np(y_raw, p_te),
            "n_val_all": len(y_va), "n_val_trunc": len(y_vt), "n_test": len(y_cap),
            "sec": round(sec, 1), "epochs": epochs,
        })
        out_preds[f"{subset}|{seed}|{name}"] = p_te
    out_preds[f"{subset}|y_cap"], out_preds[f"{subset}|y_raw"] = y_cap, y_raw
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="FD001·seed 2개·짧은 학습으로 동작 확인")
    args = ap.parse_args()
    if args.quick:
        args.subsets, args.seeds = ["FD001"], 2

    cfg = load_config()
    out_dir = cfg["paths"]["results_dir"] / ("judgement_quick" if args.quick else "judgement")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    with open(out_dir / "runs.jsonl", "w") as f:
        for subset in args.subsets:
            for seed in range(args.seeds):
                t0 = time.time()
                for r in run_one(subset, seed, cfg, args.quick, preds):
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"{subset} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(out_dir / "test_preds.npz", **preds)
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
