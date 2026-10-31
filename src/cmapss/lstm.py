"""
LSTM 기반 RUL 예측 + MC Dropout 불확실성 추정.

베이스라인(Ridge, RandomForest)과 동일한 (N, window, F) 입력을 쓰되,
시퀀스를 평탄화하지 않고 시간 구조를 그대로 LSTM에 넣는다.

MC Dropout: 추론 시에도 dropout을 켠 채 여러 번 예측해 그 분포의
평균을 예측값, 표준편차를 불확실성으로 사용한다(근사 베이지안 추론).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class RULLSTM(nn.Module):
    """2층 LSTM + dropout + 선형 헤드.

    마지막 시점의 은닉 상태만 사용한다. 윈도우 전체를 요약한 표현이
    그 시점의 열화 상태를 담고 있다고 보기 때문.
    """

    def __init__(self, n_features: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)          # MC Dropout 대상
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]                     # 마지막 시점의 은닉 상태
        return self.head(self.drop(last)).squeeze(-1)


def _loader(X, y, batch, shuffle, seed=42):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch,
                                       shuffle=shuffle, generator=g if shuffle else None)


def train_lstm(X_tr, y_tr, X_val, y_val, *, hidden=64, layers=2, dropout=0.2,
               lr=3e-3, batch=256, max_epochs=60, patience=8, seed=42,
               y_scale=125.0, verbose=True):
    """val RMSE 기준 early stopping. 가장 좋은 가중치를 되돌려 반환.

    y_scale: 타깃을 이 값으로 나눠 학습한다(RUL 상한과 동일하게 두면 [0,1] 범위).
      ※ 원본 타깃(평균 약 81)을 쓰면 0 근처에서 출발한 출력을 Adam이 스텝당 약 lr씩만
        끌어올릴 수 있어 매우 느리고, 그동안 은닉 상태가 포화되어 입력과 무관한 상수를
        낸다(예측 std=0). 출력 bias를 평균으로 초기화해도 풀리므로 원인은 출력 크기 불일치.
        재현: scripts/diag_target_scaling.py
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RULLSTM(X_tr.shape[2], hidden, layers, dropout)
    return fit_sequence_model(model, X_tr, y_tr, X_val, y_val, lr=lr, batch=batch,
                              max_epochs=max_epochs, patience=patience, seed=seed,
                              y_scale=y_scale, verbose=verbose)


def fit_sequence_model(model, X_tr, y_tr, X_val, y_val, *, lr=3e-3, batch=256, max_epochs=60,
                       patience=8, seed=42, y_scale=125.0, verbose=True):
    """(N, window, F)를 받는 모델 공용 학습 루프 (LSTM, CNN). 모델 초기화 전에 seed를 고정할 것."""
    model.y_scale = y_scale
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    tr = _loader(X_tr, y_tr / y_scale, batch, True, seed)

    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    best, best_state, bad, history = float("inf"), None, 0, []
    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v = float(torch.sqrt(((model(Xv) * y_scale - yv) ** 2).mean()))
        history.append(v)

        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  epoch {ep:>2}  val RMSE = {v:.3f}")
        if bad >= patience:
            if verbose:
                print(f"  early stopping at epoch {ep} (best val RMSE {best:.3f})")
            break

    model.load_state_dict(best_state)
    return model, {"best_val_rmse": best, "epochs_run": len(history), "history": history}


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32)).numpy() * getattr(model, "y_scale", 1.0)


@torch.no_grad()
def mc_dropout_predict(model, X, n_samples: int = 100, seed: int = 42):
    """추론 시 dropout을 켠 채 n_samples번 예측.

    반환: (평균, 표준편차, 전체 샘플 (n_samples, N))
    """
    torch.manual_seed(seed)
    model.train()                                # dropout 활성 유지
    for m in model.modules():                    # BatchNorm 등은 eval로 두는 것이 원칙
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    xb = torch.tensor(X, dtype=torch.float32)
    sc = getattr(model, "y_scale", 1.0)
    samples = np.stack([model(xb).numpy() * sc for _ in range(n_samples)])
    return samples.mean(0), samples.std(0), samples
