import numpy as np
import pandas as pd

from cmapss.features import (add_rul_column, apply_regime_scaler, apply_scaler, fit_regime_scaler,
                             fit_scaler, regime_keys)
from cmapss.protocol import (delta_rmse_diff, engine_bootstrap_diff, engines_needed,
                             gaussian_half_width, rmse_np, selection_summary, truncated_windows)


def _df(lengths=(40, 50), regimes=1, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n in enumerate(lengths, start=1):
        for c in range(1, n + 1):
            r = c % regimes
            rows.append({"unit": unit, "cycle": c, "op_setting1": 10.0 * r, "op_setting2": 0.25 * r,
                         "op_setting3": 100.0, "s": 100.0 * r + rng.normal()})
    return pd.DataFrame(rows)


def test_regime_keys_count():
    assert regime_keys(_df(regimes=1)).nunique() == 1
    assert regime_keys(_df(regimes=3)).nunique() == 3


def test_single_regime_scaler_equals_global():
    """조건이 하나면 조건별 정규화 = 전역 정규화 (구현 검증용 unit test)."""
    df = _df(regimes=1)
    a = apply_regime_scaler(df, fit_regime_scaler(df, ["s"]), ["s"])
    b = apply_scaler(df, fit_scaler(df, ["s"]), ["s"])
    np.testing.assert_allclose(a["s"], b["s"])


def test_regime_scaler_removes_condition_offset():
    df = _df(regimes=2)
    out = apply_regime_scaler(df, fit_regime_scaler(df, ["s"]), ["s"])
    for _, g in out.groupby(regime_keys(df)):
        assert g["s"].min() == 0.0 and g["s"].max() == 1.0


def test_truncated_windows_label_matches_cut_point():
    df = add_rul_column(_df(lengths=(40,)), rul_cap=1000)
    df["t"] = df["cycle"].astype(float)
    X, y, units = truncated_windows(df, ["t"], window_size=10, n_per_engine=20,
                                    rng=np.random.default_rng(1))
    assert X.shape == (20, 10, 1)
    # 마지막 윈도우의 마지막 cycle이 t, 그 시점 RUL = 40 - t
    np.testing.assert_allclose(y, 40 - X[:, -1, 0])


def test_bootstrap_zero_when_identical():
    y = np.arange(50.0)
    r = engine_bootstrap_diff(y, y + 1, y + 1, n_boot=200)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_selection_summary_regret():
    s = selection_summary({"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 2.5})
    assert s["picked"] == "a" and s["test_best"] == "b" and not s["top1"]
    assert s["regret"] == 0.5


def _paired_errors(n, sigma, rho, seed):
    rng = np.random.default_rng(seed)
    e = rng.multivariate_normal([0, 0], sigma ** 2 * np.array([[1, rho], [rho, 1]]), n)
    y = rng.uniform(0, 125, n)
    return y, y + e[:, 0], y + e[:, 1]


def test_delta_matches_bootstrap():
    """정규 오차에서 delta method 반폭이 bootstrap 반폭과 10% 안으로 맞는다."""
    y, pa, pb = _paired_errors(400, 12.0, 0.8, seed=1)
    d = delta_rmse_diff(y, pa, pb)
    bs = engine_bootstrap_diff(y, pa, pb, n_boot=4000, seed=1)
    assert abs(d["diff"] - bs["diff"]) < 1e-12
    assert abs(d["half_width"] / bs["half_width"] - 1) < 0.10


def test_gaussian_half_width_matches_simulation():
    """닫힌 식 sigma*sqrt((1-rho^2)/n)이 반복 시뮬레이션의 RMSE 차이 표준편차와 맞는다."""
    n, sigma, rho = 100, 12.0, 0.9
    diffs = []
    for s in range(2000):
        y, pa, pb = _paired_errors(n, sigma, rho, seed=s)
        diffs.append(rmse_np(y, pa) - rmse_np(y, pb))
    expected_se = gaussian_half_width(sigma, rho, n) / 1.96
    assert abs(np.std(diffs) / expected_se - 1) < 0.08


def test_engines_needed_inverts_half_width():
    n = engines_needed(12.0, 0.9, 0.5)
    assert gaussian_half_width(12.0, 0.9, n) <= 0.5 < gaussian_half_width(12.0, 0.9, n - 1)
