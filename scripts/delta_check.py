"""delta method 검산: 판정 실험의 test 1·2등 비교에서 닫힌 식 반폭이 bootstrap 반폭과 맞는지.

실행: python scripts/delta_check.py   (먼저 judgement_experiment.py 결과가 있어야 함)

서브셋·seed마다 test 1·2등 모델에 대해
  bootstrap 반폭 / delta method 반폭 / 정규 근사 반폭 1.96*sigma*sqrt((1-rho^2)/n)
을 비교하고, 반폭을 0.5와 1.0 RMSE로 줄이려면 test 엔진이 몇 대 필요한지 역산한다.
정규 근사는 오차 꼬리가 두꺼우면(첨도 > 3) 반폭을 작게 잡으므로 역산은 delta 반폭으로 한다.
seed 사이 차이의 표준편차(논문들이 흔히 보고하는 '± std')도 같이 출력해 크기를 비교한다.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from cmapss.config import load_config
from cmapss.protocol import delta_rmse_diff, engine_bootstrap_diff, gaussian_half_width


def main():
    d = load_config()["paths"]["results_dir"] / "judgement"
    runs = pd.read_json(d / "runs.jsonl", lines=True)
    preds = np.load(d / "test_preds.npz")
    out = {}
    for subset, g in runs.groupby("subset"):
        y = preds[f"{subset}|y_cap"]
        rows = []
        for seed, s in g.groupby("seed"):
            test = dict(zip(s.model, s.test_cap))
            first, second = sorted(test, key=test.get)[:2]
            pa, pb = preds[f"{subset}|{seed}|{second}"], preds[f"{subset}|{seed}|{first}"]
            ea, eb = pa - y, pb - y
            sigma = float(np.sqrt((np.mean(ea ** 2) + np.mean(eb ** 2)) / 2))
            rho = float(np.corrcoef(ea, eb)[0, 1])
            rows.append({
                "boot": engine_bootstrap_diff(y, pa, pb, seed=int(seed))["half_width"],
                "delta": delta_rmse_diff(y, pa, pb)["half_width"],
                "gauss": gaussian_half_width(sigma, rho, len(y)),
                "diff": test[second] - test[first], "sigma": sigma, "rho": rho,
                "kurt": float(np.mean([kurtosis(ea, fisher=False), kurtosis(eb, fisher=False)])),
            })
        r = pd.DataFrame(rows)
        sig, rho = r.sigma.median(), r.rho.median()
        out[subset] = {
            "n_test": int(len(y)),
            "half_width_bootstrap": round(r.boot.median(), 2),
            "half_width_delta": round(r.delta.median(), 2),
            "half_width_gaussian": round(r.gauss.median(), 2),
            "sigma": round(sig, 2), "rho": round(rho, 3), "error_kurtosis": round(r["kurt"].median(), 2),
            # SE는 1/sqrt(n)에 비례하므로 실측 delta 반폭에서 역산 (분포 가정 없음)
            "engines_for_hw_1.0": int(np.ceil(len(y) * (r.delta.median() / 1.0) ** 2)),
            "engines_for_hw_0.5": int(np.ceil(len(y) * (r.delta.median() / 0.5) ** 2)),
            "top2_gap_mean": round(r["diff"].mean(), 2),
            "top2_gap_seed_std": round(r["diff"].std(ddof=1), 2),
        }
    print(pd.DataFrame(out).T.to_string())
    json.dump(out, open(d / "delta_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
