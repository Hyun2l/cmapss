"""
1D-CNN 기반 RUL 예측 (Li et al., 2018 DCNN 계열: 시간축 합성곱 + flatten).

LSTM과 같은 (N, window, F) 입력, 같은 학습 루프(lstm.fit_sequence_model)와 타깃 스케일을 쓴다.
합성곱은 시간축으로 국소 패턴(짧은 구간의 기울기, 급변)을 잡고, flatten은 윈도우 안의 위치
정보(마지막 시점이 가장 중요)를 버리지 않기 위해 global pooling 대신 사용한다.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from cmapss.lstm import fit_sequence_model


class RULCNN(nn.Module):
    def __init__(self, n_features: int, window: int, channels=(32, 64, 64), kernel: int = 5,
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        layers, c_in = [], n_features
        for c in channels:
            layers += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2), nn.ReLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in * window, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):                        # x: (N, window, F)
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


def train_cnn(X_tr, y_tr, X_val, y_val, *, dropout=0.2, lr=1e-3, batch=256, max_epochs=60,
              patience=8, seed=42, y_scale=125.0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RULCNN(X_tr.shape[2], X_tr.shape[1], dropout=dropout)
    return fit_sequence_model(model, X_tr, y_tr, X_val, y_val, lr=lr, batch=batch,
                              max_epochs=max_epochs, patience=patience, seed=seed,
                              y_scale=y_scale, verbose=verbose)
