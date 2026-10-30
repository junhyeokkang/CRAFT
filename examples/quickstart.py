"""Train CRAFT on a synthetic multivariate series.

Replace `series` with your own (T, C) array, scaled per channel.
"""
import numpy as np
import torch

from craft import CRAFT, build_memory

LOOKBACK, HORIZON = 336, 96
rng = np.random.default_rng(0)

# Six channels with different periods, plus noise.
steps = np.arange(4000)[:, None]
periods = np.array([24, 24, 48, 168, 168, 12])
series = np.sin(2 * np.pi * steps / periods) + 0.1 * rng.normal(size=(4000, 6))
series = ((series - series.mean(0)) / series.std(0)).astype(np.float32)
train, test = series[:3000], series[3000 - LOOKBACK:]

# The memory and the training windows both come from the training split only.
keys, values, starts = build_memory(train, LOOKBACK, HORIZON, stride=10)
model = CRAFT(LOOKBACK, HORIZON)
model.build_memory(keys, values, starts)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
inputs, targets, input_starts = build_memory(train, LOOKBACK, HORIZON, stride=1)
for epoch in range(3):
    order = torch.randperm(len(inputs))
    epoch_losses = []
    for start in range(0, len(order), 32):
        batch = order[start:start + 32]
        # Training windows overlap the memory, so pass their starts to exclude
        # memory samples that would reveal their own future.
        forecast = model(inputs[batch], query_starts=input_starts[batch])
        loss = torch.nn.functional.mse_loss(forecast, targets[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    print(f"epoch {epoch + 1}: train MSE {np.mean(epoch_losses):.4f}")

test_inputs, test_targets, _ = build_memory(test, LOOKBACK, HORIZON, stride=HORIZON)
with torch.no_grad():
    test_mse = torch.nn.functional.mse_loss(model(test_inputs), test_targets).item()
print(f"test MSE {test_mse:.4f}")
