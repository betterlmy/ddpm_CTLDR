import torch
import torch.nn as nn

# ema测试
model = nn.Sequential(nn.Linear(1, 1))

opt = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练数据
inputs = torch.randn(64, 1)
targets = torch.sin(inputs)
alpha = 0.9
smoothed_loss = 0

loss_fn = nn.MSELoss()

for i in range(100):
    # 前向传播
    preds = model(inputs)
    loss = loss_fn(preds, targets)

    opt.zero_grad()
    loss.backward()
    opt.step()

    smoothed_loss = alpha * smoothed_loss + (1 - alpha) * loss.item()

    if i % 10 == 0:
        print(f"iter {i},loss{loss.item():.5f},smooth{smoothed_loss}")
