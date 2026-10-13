# RMSE
import numpy as np

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# PHM score

def phm_score(y_true, y_pred, a_early: float = 13.0, a_late: float = 10.0) -> float:
    """
    d = 예측 - 실제
    d < 0 (조기 예측, 상대적으로 안전): 분모 a_early=13, 완만한 페널티
    d >= 0 (지연 예측, 위험): 분모 a_late=10, 훨씬 가파른 페널티
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / a_early) - 1, np.exp(d / a_late) - 1)
    return float(np.sum(scores))
