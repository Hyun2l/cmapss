"""
베이스라인 모델 (sklearn).

(N, window, features) 시퀀스를 트리/선형 모델에 넣기 위해 tabular로 변환한 뒤 학습.
LSTM과 MC Dropout은 lstm.py에 있다. Attention은 아직 구현하지 않았다.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def sequences_to_tabular(X: np.ndarray) -> np.ndarray:
    """윈도우를 요약 통계로 펼침: 마지막 시점값 + 윈도우 평균 + 윈도우 기울기.

    (N, window, F) -> (N, 3F). 마지막 값은 '현재 상태', 평균은 노이즈 완화,
    기울기는 열화 속도를 반영.
    """
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    slope = np.einsum("t,ntf->nf", t, X) / np.sum(t ** 2)
    return np.hstack([last, mean, slope])


def train_ridge(X_tab: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X_tab, y)
    return model


def train_random_forest(X_tab: np.ndarray, y: np.ndarray, seed: int = 42, **kwargs) -> RandomForestRegressor:
    params = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5, "n_jobs": -1}
    params.update(kwargs)
    model = RandomForestRegressor(random_state=seed, **params)
    model.fit(X_tab, y)
    return model


def train_hist_gb(X_tab: np.ndarray, y: np.ndarray, seed: int = 42, **kwargs):
    """HistGradientBoosting: 큰 데이터(FD002/004)에서도 빠른 boosting 베이스라인."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31}
    params.update(kwargs)
    model = HistGradientBoostingRegressor(random_state=seed, **params)
    model.fit(X_tab, y)
    return model
