"""
LSTM 학습 실패 진단: 타깃 스케일링이 왜 필요한가.

증상
----
원본 RUL(평균 약 81, 최대 125)을 그대로 타깃으로 쓰면 LSTM의 예측 표준편차가 0이다.
즉 입력과 무관하게 모든 샘플에 같은 값을 낸다. 그 값은 0 근처에서 출발해 epoch마다
조금씩 올라가며, 40 epoch쯤 되어서야 평균값 예측(naive) 수준의 RMSE에 도달한다.

원인
----
1) 출력 크기 불일치: 초기화 직후 출력은 0 근처인데 정답 평균은 약 81이다.
   Adam은 한 스텝에 파라미터를 대략 lr(1e-3)만큼만 움직이므로, bias 하나로 81을
   채우려면 수만 스텝이 걸린다.
2) 포화: 그래서 모델은 은닉 상태를 tanh 포화(±1) 쪽으로 밀고 출력층 가중치를 키워
   출력을 끌어올린다. 은닉 상태가 포화되면 입력이 달라도 같은 값이 나오고(예측 std=0),
   포화 구간은 기울기도 작아서 입력 정보를 다시 배우기 어렵다.

검증 (각 설정 8 epoch, 아래 실행 결과)
----
  A. 원본 타깃                     → 예측 평균이 0에서 천천히 상승, std = 0
  B. 원본 타깃 + 출력 bias = 평균  → 바로 학습됨  (출력 크기 문제라는 증거)
  C. 타깃 / rul_cap               → 가장 빠르고 안정적 (출력·손실·기울기 크기가 모두 1 근처)

해결
----
타깃을 rul_cap(125)으로 나눠 [0, 1] 범위로 학습하고, 예측값에 다시 곱해 되돌린다.
cmapss.lstm.train_lstm의 y_scale 인자가 이 역할을 한다.

실행: python scripts/diag_target_scaling.py
"""
import time

import numpy as np
import torch
import torch.nn as nn

from cmapss.config import load_config
from cmapss.lstm import RULLSTM

cfg = load_config()
P = cfg["paths"]["processed_dir"]
CAP = float(cfg["preprocess"]["rul_cap"])
SEED = cfg["preprocess"]["seed"]

X = np.load(P / "X_train.npy"); y = np.load(P / "y_train.npy").astype(np.float32)
Xv = np.load(P / "X_val.npy"); yv = np.load(P / "y_val.npy").astype(np.float32)


def run(tag, scale=1.0, init_bias=None, epochs=8, lr=1e-3, dropout=0.3):
    torch.manual_seed(SEED)
    m = RULLSTM(X.shape[2], 64, 2, dropout)
    if init_bias is not None:
        with torch.no_grad():
            m.head.bias.fill_(init_bias / scale)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.MSELoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y / scale)),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    t0 = time.time()
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(Xv, dtype=torch.float32)).numpy() * scale
    rmse = float(np.sqrt(((p - yv) ** 2).mean()))
    print(f"  {tag:<34} val RMSE={rmse:6.2f}  예측 평균={p.mean():6.2f}  예측 std={p.std():6.2f}"
          f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    print(f"[기준] 정답 평균={y.mean():.2f}, val 정답 std={yv.std():.2f}"
          f"  →  평균으로 전부 예측하면 val RMSE≈{yv.std():.2f}\n")
    print("각 설정 8 epoch:")
    run("A. 원본 타깃", scale=1.0)
    run("B. 원본 타깃 + 출력 bias=평균", scale=1.0, init_bias=float(y.mean()))
    run(f"C. 타깃/{CAP:.0f}", scale=CAP)
    print("\nA: 예측 std≈0, 예측 평균이 정답 평균(약 81)보다 한참 낮다 → 출력이 아직 정답 크기에 못 미침.")
    print("B: 출력 시작점만 맞춰도 학습된다 → 원인은 '출력 크기 불일치'.")
