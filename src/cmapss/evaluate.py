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


# Baseline / summary

def naive_baseline_rmse(y_train, y_true) -> float:
    """train RUL 평균값 하나로 전부 예측했을 때의 RMSE. 모델이 이것보다는 나아야 의미 있음."""
    y_true = np.asarray(y_true, dtype=float)
    return rmse(y_true, np.full_like(y_true, np.mean(y_train)))


def coverage(y_true, lower, upper) -> float:
    """예측 구간 [lower, upper] 안에 실제값이 들어간 비율 (불확실성 모델 평가용)."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def evaluate(y_true, y_pred, name: str = "model") -> dict:
    """RMSE + PHM score를 한 번에 계산해서 dict로 반환."""
    result = {"model": name, "rmse": rmse(y_true, y_pred), "phm_score": phm_score(y_true, y_pred)}
    print(f"[{name}] RMSE={result['rmse']:.2f}  PHM={result['phm_score']:.1f}")
    return result
