import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch는 선택 의존성(.[dl]) — 없으면 건너뜀

from cmapss.cnn import RULCNN, train_cnn
from cmapss.lstm import predict


def test_rulcnn_output_shape():
    assert RULCNN(n_features=3, window=10)(torch.randn(4, 10, 3)).shape == (4,)


def test_train_cnn_learns_and_is_seeded():
    rng = np.random.default_rng(0)
    X = rng.random((256, 10, 3)).astype(np.float32)
    y = (X[:, -1, 0] * 100).astype(np.float32)
    m1, info = train_cnn(X, y, X, y, max_epochs=30, patience=30, batch=32, seed=3, verbose=False)
    m2, _ = train_cnn(X, y, X, y, max_epochs=30, patience=30, batch=32, seed=3, verbose=False)
    np.testing.assert_allclose(predict(m1, X), predict(m2, X), rtol=1e-5)
    assert info["best_val_rmse"] < y.std() * 0.5      # 평균 예측보다 확실히 나아야 함
