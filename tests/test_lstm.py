import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch는 선택 의존성(.[dl]) — 없으면 건너뜀

from cmapss.lstm import RULLSTM, mc_dropout_predict, predict, train_lstm


def _toy_xy(n=64, window=10, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, window, n_features)).astype(np.float32)
    y = (X[:, -1, 0] * 100).astype(np.float32)      # 마지막 시점 첫 피처에 비례하는 타깃
    return X, y


def test_rullstm_output_shape():
    model = RULLSTM(n_features=3)
    out = model(torch.randn(4, 10, 3))
    assert out.shape == (4,)


def test_predict_restores_target_scale():
    """y_scale로 나눠 학습해도 predict는 원래 단위(사이클)로 되돌려야 한다."""
    X, y = _toy_xy()
    model, _ = train_lstm(X, y, X, y, max_epochs=3, patience=3, y_scale=100.0, verbose=False)
    pred = predict(model, X)
    raw = model.eval()(torch.tensor(X)).detach().numpy()
    np.testing.assert_allclose(pred, raw * 100.0, rtol=1e-5)


def test_mc_dropout_gives_spread_and_is_seeded():
    X, y = _toy_xy()
    model, _ = train_lstm(X, y, X, y, max_epochs=2, patience=2, dropout=0.3, verbose=False)
    mean1, std1, samples = mc_dropout_predict(model, X, n_samples=20, seed=1)
    mean2, _, _ = mc_dropout_predict(model, X, n_samples=20, seed=1)
    assert samples.shape == (20, len(X))
    assert (std1 > 0).all()                 # dropout이 켜져 있으면 샘플마다 값이 달라야 함
    np.testing.assert_allclose(mean1, mean2)  # 같은 seed면 재현
