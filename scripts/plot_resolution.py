"""README 그림: 서브셋별 test 1·2등 RMSE 차이와 95% 구간 (seed 10개).

실행: python scripts/plot_resolution.py  → results/figures/test_resolution.png
구간은 delta method (protocol.delta_rmse_diff). 구간이 0을 넘으면 1·2등을 구분할 수 없다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmapss.config import load_config
from cmapss.protocol import delta_rmse_diff

INK, MUTED, GRID, BLUE, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"


def main():
    d = load_config()["paths"]["results_dir"]
    runs = pd.read_json(d / "judgement" / "runs.jsonl", lines=True)
    preds = np.load(d / "judgement" / "test_preds.npz")
    subsets = sorted(runs.subset.unique())

    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    yticks = []
    for k, subset in enumerate(subsets):
        g = runs[runs.subset == subset]
        y_true = preds[f"{subset}|y_cap"]
        base = k * 12
        for seed, s in g.groupby("seed"):
            test = dict(zip(s.model, s.test_cap))
            first, second = sorted(test, key=test.get)[:2]
            r = delta_rmse_diff(y_true, preds[f"{subset}|{seed}|{second}"], preds[f"{subset}|{seed}|{first}"])
            y = base + seed
            ax.plot([r["lo"], r["hi"]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(r["diff"], y, "o", ms=6, color=BLUE, mec=SURFACE, mew=1.5)
        yticks.append((base + 4.5, f"{subset}\n(n={len(y_true)})"))
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([t for t, _ in yticks], [l for _, l in yticks], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE(2nd) − RMSE(1st) on the official test set, 95% CI (delta method)", color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Best vs runner-up model: intervals almost always cross zero\n"
                 "one line = one seed (10 per subset), 5 models", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = d / "figures" / "test_resolution.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
