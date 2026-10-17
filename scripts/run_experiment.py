"""베이스라인(Ridge/RF) vs LSTM vs MC Dropout 비교 실험.

실행: python scripts/run_experiment.py   (먼저 02_preprocessing.ipynb로 data/processed 생성)

test 지표는 두 가지로 보고한다.
  - capped: 정답 RUL도 rul_cap(125)으로 자른 값. 학습 라벨과 같은 기준이며 문헌 비교용.
  - raw   : RUL_FD001.txt 원본 그대로. 125를 넘는 엔진(100대 중 11대)에서 구조적으로 손해.
"""
import json
import time

import numpy as np

from cmapss.config import load_config
from cmapss.evaluate import coverage, naive_baseline_rmse, phm_score, rmse
from cmapss.lstm import mc_dropout_predict, predict, train_lstm
from cmapss.models import sequences_to_tabular, train_random_forest, train_ridge

cfg = load_config()
P = cfg["paths"]["processed_dir"]
RUL_CAP = cfg["preprocess"]["rul_cap"]
SEED = cfg["preprocess"]["seed"]

load = lambda n: np.load(P / f"{n}.npy")
X_tr, y_tr = load("X_train"), load("y_train")
X_va, y_va = load("X_val"), load("y_val")
X_te, y_te_raw = load("X_test"), load("y_test")
y_te = np.minimum(y_te_raw, RUL_CAP)
print(f"train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")
print(f"test 정답 중 rul_cap({RUL_CAP}) 초과: {(y_te_raw > RUL_CAP).sum()}대\n")


def test_metrics(pred):
    return {"rmse": rmse(y_te, pred), "phm": phm_score(y_te, pred),
            "rmse_raw": rmse(y_te_raw, pred), "phm_raw": phm_score(y_te_raw, pred)}


def report(name, r):
    val = f"val RMSE={r['val_rmse']:.2f}  " if "val_rmse" in r else ""
    print(f"[{name}] {val}test RMSE={r['rmse']:.2f} (raw {r['rmse_raw']:.2f})  "
          f"PHM={r['phm']:.0f} (raw {r['phm_raw']:.0f})")


results = {"protocol": {"rul_cap": RUL_CAP, "seed": SEED,
                        "test_capped_note": "rmse/phm = capped truth, *_raw = original truth"}}

# ---------- 기준선 ----------
results["naive"] = {"rmse": naive_baseline_rmse(y_tr, y_te),
                    "rmse_raw": naive_baseline_rmse(y_tr, y_te_raw)}
print(f"[naive] train RUL 평균으로 전부 예측 → test RMSE={results['naive']['rmse']:.2f} "
      f"(raw {results['naive']['rmse_raw']:.2f})\n")

# ---------- 베이스라인 ----------
Xtr_tab, Xva_tab, Xte_tab = map(sequences_to_tabular, (X_tr, X_va, X_te))
for name, fit in [("Ridge", lambda: train_ridge(Xtr_tab, y_tr)),
                  ("RandomForest", lambda: train_random_forest(Xtr_tab, y_tr, seed=SEED))]:
    t0 = time.time()
    m = fit()
    results[name] = {"val_rmse": rmse(y_va, m.predict(Xva_tab)), **test_metrics(m.predict(Xte_tab)),
                     "sec": round(time.time() - t0, 1)}
    report(name, results[name])

# ---------- LSTM ----------
print("\n[LSTM] 학습 시작")
t0 = time.time()
model, info = train_lstm(X_tr, y_tr, X_va, y_va, seed=SEED, y_scale=float(RUL_CAP))
pt = predict(model, X_te)
results["LSTM"] = {"val_rmse": info["best_val_rmse"], **test_metrics(pt),
                   "epochs": info["epochs_run"], "sec": round(time.time() - t0, 1)}
report("LSTM", results["LSTM"])
print(f"  ({results['LSTM']['epochs']} epochs, {results['LSTM']['sec']}s)")

# ---------- MC Dropout ----------
print("\n[MC Dropout] 100회 샘플링")
mean, std, _ = mc_dropout_predict(model, X_te, n_samples=100, seed=SEED)
z = 1.2816                                   # 80% 구간 (정규 근사)
lo, hi = mean - z * std, mean + z * std
results["MC_Dropout"] = {**test_metrics(mean),
                         "coverage80": coverage(y_te, lo, hi),
                         "mean_width": float((hi - lo).mean()),
                         "mean_std": float(std.mean()),
                         # 불확실성이 실제 오차와 같이 움직이는지 — 불확실성 추정이 쓸모 있는지의 핵심 검증
                         "corr_std_vs_abserr": float(np.corrcoef(std, np.abs(mean - y_te))[0, 1])}
r = results["MC_Dropout"]
report("MC Dropout", r)
print(f"  80% 예측구간 실제 포함률(coverage) = {r['coverage80']:.1%}   (목표 80%)")
print(f"  평균 구간 폭 = {r['mean_width']:.1f} cycles,  평균 표준편차 = {r['mean_std']:.2f}")
print(f"  불확실성(std)과 실제 절대오차의 상관 = {r['corr_std_vs_abserr']:.3f}")

# ---------- RUL 구간별 오차 (열화 말기일수록 중요) ----------
print("\n[구간별 LSTM 오차, capped 정답 기준]")
results["LSTM_by_rul"] = {}
for lo_, hi_ in [(0, 50), (50, 100), (100, RUL_CAP + 1)]:
    m_ = (y_te >= lo_) & (y_te < hi_)
    if m_.sum():
        e = rmse(y_te[m_], pt[m_])
        results["LSTM_by_rul"][f"{lo_}-{hi_}"] = {"n": int(m_.sum()), "rmse": e}
        print(f"  실제 RUL {lo_:>3}–{hi_:<3}: n={m_.sum():>3}  RMSE={e:.2f}")

out = cfg["paths"]["results_dir"]
out.mkdir(exist_ok=True)
json.dump(results, open(out / "experiment_results.json", "w"), indent=2, ensure_ascii=False)
print(f"\n저장: {out / 'experiment_results.json'}")
