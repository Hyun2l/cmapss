import numpy as np
import pandas as pd

from cmapss.evaluate import phm_score, rmse
from cmapss.features import add_rul_column, create_sequences, create_test_sequences, split_units
from cmapss.models import sequences_to_tabular


def _toy_df(lengths=(5, 8, 6)):
    rows = []
    for unit, n in enumerate(lengths, start=1):
        for cycle in range(1, n + 1):
            rows.append({"unit": unit, "cycle": cycle, "s": float(cycle)})
    return pd.DataFrame(rows)


def test_add_rul_column_caps_and_ends_at_zero():
    df = add_rul_column(_toy_df(), rul_cap=4)
    assert df["RUL"].max() == 4
    assert (df.groupby("unit")["RUL"].last() == 0).all()


def test_split_units_has_no_overlap():
    df = _toy_df(lengths=(5,) * 10)
    tr, val, tr_u, val_u = split_units(df, val_ratio=0.2, seed=0)
    assert not set(tr_u) & set(val_u)
    assert len(val_u) == 2
    assert len(tr) + len(val) == len(df)


def test_create_sequences_shapes_and_labels():
    df = add_rul_column(_toy_df(), rul_cap=100)
    X, y, units = create_sequences(df, ["s"], window_size=5)
    # 윈도우 수 = sum(n - w + 1)
    assert X.shape == ((5 - 4) + (8 - 4) + (6 - 4), 5, 1)
    assert y[0] == 0  # unit1 유일한 윈도우는 마지막 cycle에서 끝남


def test_create_test_sequences_pads_short_units():
    X, units = create_test_sequences(_toy_df(), ["s"], window_size=7)
    assert X.shape == (3, 7, 1)
    assert X[0, 0, 0] == X[0, 2, 0] == 1.0  # unit1(길이 5)은 첫 행으로 패딩


def test_sequences_to_tabular_slope():
    X = np.arange(10, dtype=float).reshape(1, 10, 1)
    tab = sequences_to_tabular(X)
    np.testing.assert_allclose(tab[0], [9.0, 4.5, 1.0])


def test_metrics():
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)
    assert phm_score([10], [10]) == 0.0
    assert phm_score([10], [20]) > phm_score([20], [10])  # 늦은 예측이 더 큰 페널티
