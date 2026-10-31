"""판정 실험 분석. 판정 기준은 실험 전에 여기 고정한다 (결과를 보고 바꾸지 않는다).

실행: python scripts/judgement_analysis.py [--dir results/judgement] [--threshold 1.0]

기준 A (test 해상도): 서브셋마다, seed별로 test 1·2등 모델의 RMSE 차이에 엔진 단위
  bootstrap 95% 구간을 구한다. 반폭(half-width)의 중앙값이 --threshold(기본 1.0 RMSE) 이상이면
  "이 test 세트는 그보다 작은 개선을 구분하지 못한다". threshold는 최근 논문들이 보고한
  개선폭으로 정한다 (조사 전 임시값 1.0).
기준 B (선택 신뢰도): seed별 regret(고른 모델의 test RMSE - test 최적 RMSE)을 두 검증 방식으로
  비교. 차이 평균 > 0 이고 2 x 표준오차보다 크면 "test처럼 자른 검증이 선택을 개선한다".
판정: A 또는 B가 서브셋 2개 이상에서 성립하면 GO(논문화), 아니면 NO-GO(기술 보고서로 정리).
지표는 정답 cap 기준 test RMSE. 원본 기준은 참고로 함께 출력.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cmapss.config import load_config
from cmapss.protocol import engine_bootstrap_diff, selection_summary


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(cfg["paths"]["results_dir"] / "judgement"))
    ap.add_argument("--threshold", type=float, default=1.0)
    args = ap.parse_args()
    d = Path(args.dir)
    runs = pd.read_json(d / "runs.jsonl", lines=True)
    preds = np.load(d / "test_preds.npz")

    summary, a_ok, b_ok = {}, [], []
    for subset, g in runs.groupby("subset"):
        sel = {"val_all": [], "val_trunc": []}
        half, zero_in = [], []
        for seed, s in g.groupby("seed"):
            test = dict(zip(s.model, s.test_cap))
            for proto in sel:
                sel[proto].append(selection_summary(dict(zip(s.model, s[proto])), test))
            first, second = sorted(test, key=test.get)[:2]
            y = preds[f"{subset}|y_cap"]
            bs = engine_bootstrap_diff(y, preds[f"{subset}|{seed}|{second}"],
                                       preds[f"{subset}|{seed}|{first}"], seed=int(seed))
            half.append(bs["half_width"]); zero_in.append(bs["lo"] <= 0)

        reg = {p: np.array([r["regret"] for r in v]) for p, v in sel.items()}
        gain = reg["val_all"] - reg["val_trunc"]
        se = gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else float("nan")
        A = float(np.median(half)) >= args.threshold
        B = bool(gain.mean() > 0 and gain.mean() > 2 * se)
        a_ok.append(A); b_ok.append(B)
        summary[subset] = {
            "seeds": int(g.seed.nunique()),
            "top1_val_all": float(np.mean([r["top1"] for r in sel["val_all"]])),
            "top1_val_trunc": float(np.mean([r["top1"] for r in sel["val_trunc"]])),
            "regret_val_all": float(reg["val_all"].mean()),
            "regret_val_trunc": float(reg["val_trunc"].mean()),
            "regret_gain_mean": float(gain.mean()), "regret_gain_se": float(se),
            "tau_val_all": float(np.nanmean([r["kendall_tau"] for r in sel["val_all"]])),
            "tau_val_trunc": float(np.nanmean([r["kendall_tau"] for r in sel["val_trunc"]])),
            "top2_ci_halfwidth_median": float(np.median(half)),
            "top2_ci_contains0_rate": float(np.mean(zero_in)),
            "criterion_A": A, "criterion_B": B,
            "mean_test_cap": g.groupby("model").test_cap.mean().round(2).to_dict(),
            "mean_test_raw": g.groupby("model").test_raw.mean().round(2).to_dict(),
        }

    for subset, s in summary.items():
        print(f"\n== {subset} ({s['seeds']} seeds)")
        print(f"  top-1 맞힘: all-windows {s['top1_val_all']:.0%} / truncated {s['top1_val_trunc']:.0%}")
        print(f"  평균 regret: all-windows {s['regret_val_all']:.2f} / truncated {s['regret_val_trunc']:.2f}"
              f"  (개선 {s['regret_gain_mean']:.2f} ± SE {s['regret_gain_se']:.2f})")
        print(f"  Kendall tau: all-windows {s['tau_val_all']:.2f} / truncated {s['tau_val_trunc']:.2f}")
        print(f"  test 1·2등 차이 95% CI 반폭 중앙값 {s['top2_ci_halfwidth_median']:.2f},"
              f" 0 포함 비율 {s['top2_ci_contains0_rate']:.0%}")
        print(f"  평균 test RMSE (cap): {s['mean_test_cap']}")
        print(f"  기준 A {'O' if s['criterion_A'] else 'X'} / 기준 B {'O' if s['criterion_B'] else 'X'}")

    n = len(summary)
    go = sum(a_ok) >= 2 or sum(b_ok) >= 2
    note = "" if n >= 4 else f"  (서브셋 {n}개만 실행 — 잠정 판정)"
    print(f"\n판정: {'GO — 논문화' if go else 'NO-GO — 기술 보고서로 정리'}"
          f"  [A 성립 {sum(a_ok)}/{n}, B 성립 {sum(b_ok)}/{n}, threshold {args.threshold}]{note}")
    json.dump({"threshold": args.threshold, "go": go, "subsets": summary},
              open(d / "summary.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
