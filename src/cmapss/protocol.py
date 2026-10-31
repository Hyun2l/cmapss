"""검증 프로토콜과 통계 비교 (판정 실험용).

- truncated_windows: 학습에 쓰지 않은 엔진을 test처럼 임의 지점에서 잘라 마지막 윈도우만 평가
- engine_bootstrap_diff: 엔진 단위 paired bootstrap으로 두 모델의 RMSE 차이 구간
- selection_summary: 검증 순위로 고른 모델의 test 성적(top-1 정확도, regret, Kendall tau)
- delta_rmse_diff, gaussian_half_width, engines_needed: 같은 구간을 닫힌 식(delta method)으로
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def truncated_windows(df: pd.DataFrame, feature_cols: list, window_size: int, n_per_engine: int,
                      rng: np.random.Generator, frac=(0.2, 0.97), label_col: str = "RUL"):
    """엔진마다 관측 비율 f~U(frac)로 잘라 마지막 윈도우 하나를 만든다 (n_per_engine회).

    길이가 window_size보다 짧으면 create_test_sequences와 같이 첫 행 반복으로 앞을 채운다.
    라벨은 잘린 지점(윈도우 마지막 사이클)의 RUL.
    """
    X, y, units = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        data, labels = g[feature_cols].to_numpy(), g[label_col].to_numpy()
        n = len(g)
        for f in rng.uniform(frac[0], frac[1], n_per_engine):
            end = max(1, int(round(f * n)))
            w = data[max(0, end - window_size):end]
            if len(w) < window_size:
                w = np.vstack([np.repeat(w[:1], window_size - len(w), axis=0), w])
            X.append(w); y.append(labels[end - 1]); units.append(unit)
    return np.array(X), np.array(y), np.array(units)


def rmse_np(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def engine_bootstrap_diff(y, p_a, p_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """RMSE(a) - RMSE(b)의 엔진 단위 paired bootstrap. 엔진당 예측이 하나인 test용."""
    y, p_a, p_b = (np.asarray(v, float) for v in (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = rmse_np(y[i], p_a[i]) - rmse_np(y[i], p_b[i])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": rmse_np(y, p_a) - rmse_np(y, p_b), "lo": float(lo), "hi": float(hi),
            "half_width": float((hi - lo) / 2)}


def selection_summary(val_scores: dict, test_scores: dict) -> dict:
    """검증 점수(낮을수록 좋음)로 고른 모델이 test에서 얼마나 손해인지."""
    models = sorted(val_scores)
    pick = min(models, key=val_scores.get)
    best = min(models, key=test_scores.get)
    tau = kendalltau([val_scores[m] for m in models], [test_scores[m] for m in models]).statistic
    return {"picked": pick, "test_best": best, "top1": pick == best,
            "regret": test_scores[pick] - test_scores[best], "kendall_tau": float(tau)}


def delta_rmse_diff(y, p_a, p_b, z: float = 1.96) -> dict:
    """RMSE(a) - RMSE(b)의 정규근사 구간 (delta method). bootstrap 없이 닫힌 식으로 같은 걸 구한다.

    a_i = e_ai^2, b_i = e_bi^2 라 하면 RMSE = sqrt(mean(a)). g(u, v) = sqrt(u) - sqrt(v)를
    (mean(a), mean(b)) 근처에서 1차 근사하면 기울기는 (1/(2 R_a), -1/(2 R_b))이고
        Var(diff) ~= Var(a_i / (2 R_a) - b_i / (2 R_b)) / n.
    엔진끼리 독립이라는 가정만 쓴다 (오차 분포 가정 없음).
    """
    y, p_a, p_b = (np.asarray(v, float) for v in (y, p_a, p_b))
    a, b = (y - p_a) ** 2, (y - p_b) ** 2
    r_a, r_b = np.sqrt(a.mean()), np.sqrt(b.mean())
    se = float(np.std(a / (2 * r_a) - b / (2 * r_b), ddof=1) / np.sqrt(len(y)))
    diff = float(r_a - r_b)
    return {"diff": diff, "se": se, "lo": diff - z * se, "hi": diff + z * se, "half_width": z * se}


def gaussian_half_width(sigma: float, rho: float, n: int, z: float = 1.96) -> float:
    """두 모델 오차가 평균 0, 표준편차 sigma, 상관 rho인 정규분포일 때 delta method 반폭.

    Var(e^2) = 2 sigma^4, Cov(e_a^2, e_b^2) = 2 rho^2 sigma^4 를 delta_rmse_diff의 분산식에 넣으면
        SE = sigma * sqrt((1 - rho^2) / n).
    반폭은 오차 크기에 비례하고, 엔진 수의 제곱근과 두 모델 오차의 '다름' sqrt(1 - rho^2)에 좌우된다.
    """
    return z * sigma * np.sqrt((1 - rho ** 2) / n)


def engines_needed(sigma: float, rho: float, half_width: float, z: float = 1.96) -> int:
    """gaussian_half_width를 n에 대해 푼 것: 반폭을 half_width까지 줄이려면 test 엔진이 몇 대 필요한가."""
    return int(np.ceil((z * sigma) ** 2 * (1 - rho ** 2) / half_width ** 2))
