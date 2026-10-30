"""CRAFT forecaster: a direct forecast refined by channel-wise retrieved references.

    Y_hat = Y_direct + alpha * Y_retrieval

Shape letters: B batch size, L lookback length, H horizon, C channels,
R retrieved references per channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .retrieval import ChannelWiseRetrieval


class LastValueMLP(nn.Module):
    """Two-layer MLP over the time axis with NLinear-style last-value normalization.

    The last value of the lookback window is subtracted from the input and
    added back to the output, which alleviates distribution shift.
    """

    def __init__(self, lookback_length: int, horizon: int, hidden_size: int | None = None):
        super().__init__()
        hidden_size = hidden_size or horizon
        self.layers = nn.Sequential(
            nn.Linear(lookback_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B, H, C)."""
        last_value = x[:, -1:, :]
        forecast = self.layers((x - last_value).transpose(1, 2)).transpose(1, 2)
        return forecast + last_value


class CRAFT(nn.Module):
    """Channel-wise Retrieval-Augmented Forecasting.

    Args:
        lookback_length: L, length of the input window.
        horizon: H, number of future steps.
        alpha: fixed weight of the retrieval-based forecast.
        num_relations: M, related channels kept per channel in the relation graph.
        num_references: R, references retrieved per channel.
        num_frequencies: F, low-frequency FFT components used for similarity.
        include_self: whether a channel counts among its own related channels.
        direct_forecaster: any module mapping (B, L, C) to (B, H, C). Defaults to
            a two-layer MLP with last-value normalization.
    """

    def __init__(self, lookback_length: int, horizon: int, alpha: float = 0.001,
                 num_relations: int = 3, num_references: int = 1, num_frequencies: int = 36,
                 include_self: bool = True, direct_forecaster: nn.Module | None = None):
        super().__init__()
        self.alpha = alpha
        self.retrieval = ChannelWiseRetrieval(num_relations, num_references, num_frequencies,
                                              include_self)
        self.direct_forecaster = direct_forecaster or LastValueMLP(lookback_length, horizon)
        self.reference_projection = nn.Linear(horizon, horizon)   # f_retrieved

    def build_memory(self, keys: torch.Tensor, values: torch.Tensor,
                     starts: torch.Tensor | None = None) -> None:
        """Index the training memory. Call once before training or inference.

        Args:
            keys: (N, L, C) past lookback windows.
            values: (N, H, C) the horizons that followed them.
            starts: (N,) their start positions, used to exclude overlaps in training.
        """
        self.retrieval.fit(keys, values, starts)

    def retrieval_forecast(self, x: torch.Tensor,
                           query_starts: torch.Tensor | None = None) -> torch.Tensor:
        """Average of the projected references for every channel: (B, L, C) -> (B, H, C)."""
        _, references = self.retrieval(x, query_starts)            # (C, B, R, H)
        last_value = references[..., -1:]
        projected = self.reference_projection(references - last_value) + last_value
        return projected.mean(dim=2).permute(1, 2, 0)

    def forward(self, x: torch.Tensor, query_starts: torch.Tensor | None = None) -> torch.Tensor:
        """(B, L, C) lookback windows -> (B, H, C) forecasts.

        Pass `query_starts` (B,) for training windows cut from the memory series,
        so that a window never retrieves its own future.
        """
        return self.direct_forecaster(x) + self.alpha * self.retrieval_forecast(x, query_starts)
