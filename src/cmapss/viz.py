"""시각화 유틸."""
import matplotlib.pyplot as plt
import numpy as np

# 그래프에서 마이너스(-) 기호가 깨지는 문제
plt.rcParams["axes.unicode_minus"] = False


def plot_prediction_vs_actual(y_true, y_pred, title: str = "Predicted vs Actual RUL", save_path: str = None):
    """엔진별 실제 RUL(정렬) 대비 예측 RUL."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true[order], label="Actual", linewidth=2)
    ax.plot(y_pred[order], "o", markersize=3, alpha=0.7, label="Predicted")
    ax.set_xlabel("Engine (sorted by actual RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig, ax
