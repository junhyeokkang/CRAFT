"""Channel-wise retrieval: sparse relation graph + spectral similarity.

Shape letters: N memory samples, L lookback length, H horizon, C channels,
M related channels per channel, F kept frequency components, B batch size,
R retrieved references per channel.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_memory(series: np.ndarray, lookback_length: int, horizon: int,
                 stride: int = 10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut a series into (lookback, future) pairs.

    Use it for the retrieval memory (training split only) and, with stride=1,
    for training samples.

    Args:
        series: (T, C) series, already scaled the same way as the model input.
        stride: step between consecutive samples.

    Returns:
        keys: (N, L, C) lookback windows.
        values: (N, H, C) the horizons that followed them.
        starts: (N,) start position of every window in `series`.
    """
    starts = np.arange(0, len(series) - lookback_length - horizon + 1, stride)
    keys = np.stack([series[start:start + lookback_length] for start in starts])
    values = np.stack([series[start + lookback_length:start + lookback_length + horizon]
                       for start in starts])
    return (torch.tensor(keys, dtype=torch.float32), torch.tensor(values, dtype=torch.float32),
            torch.tensor(starts, dtype=torch.long))


def cosine_similarity_matrix(rows: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between every pair of rows of a (C, D) matrix."""
    normalized = rows / (torch.linalg.norm(rows, dim=1, keepdim=True) + epsilon)
    return normalized @ normalized.T


class ChannelWiseRetrieval(nn.Module):
    """Retrieve variable-specific references from a historical memory.

    1. Relation graph: each channel is linked to its `num_relations` most similar
       channels (cosine similarity of their concatenated memory trajectories).
       The channel itself is one of them unless `include_self=False`.
    2. Knowledge base: the first `num_frequencies` FFT components of every
       candidate key, grouped per channel by the relation graph.
    3. Retrieval: each query channel is compared with its candidates by the
       normalized complex inner product of spectra, and the values of the
       `num_references` best keys are returned.
    """

    def __init__(self, num_relations: int = 3, num_references: int = 1,
                 num_frequencies: int = 36, include_self: bool = True, epsilon: float = 1e-8):
        super().__init__()
        self.num_relations = num_relations
        self.include_self = include_self
        self.num_references = num_references
        self.num_frequencies = num_frequencies
        self.epsilon = epsilon

    @torch.no_grad()
    def fit(self, keys: torch.Tensor, values: torch.Tensor,
            starts: torch.Tensor | None = None) -> None:
        """Build the relation graph and the channel-wise knowledge base.

        Args:
            keys: (N, L, C) memory lookback windows.
            values: (N, H, C) memory horizons.
            starts: (N,) start positions of the memory windows in the training
                series. Needed only to exclude overlapping memory during training.
        """
        num_channels = keys.shape[2]
        trajectories = keys.permute(2, 0, 1).reshape(num_channels, -1)       # (C, N*L)
        similarity = cosine_similarity_matrix(trajectories, self.epsilon)    # (C, C)
        # The channel itself is always the first neighbour, or never one.
        similarity.fill_diagonal_(float("inf") if self.include_self else float("-inf"))
        related_channels = torch.topk(similarity, k=self.num_relations, dim=-1).indices  # (C, M)

        key_spectra = torch.fft.rfft(keys, dim=1)[:, :self.num_frequencies]  # (N, F, C)
        candidate_spectra = key_spectra[:, :, related_channels]              # (N, F, C, M)
        # (C, F, N*M): candidate index n*M + m is memory sample n of related channel m.
        candidate_spectra = candidate_spectra.permute(2, 1, 0, 3).flatten(start_dim=2)

        self.register_buffer("keys", keys, persistent=False)
        self.register_buffer("values", values, persistent=False)
        self.register_buffer("related_channels", related_channels, persistent=False)
        self.register_buffer("candidate_spectra", candidate_spectra, persistent=False)
        self.register_buffer("starts", starts, persistent=False)

    def spectral_similarity(self, query_spectra: torch.Tensor) -> torch.Tensor:
        """Normalized complex inner product between queries and candidate keys.

        Args:
            query_spectra: (C, B, F) complex spectra of the query windows.

        Returns:
            (C, B, N*M) similarity of every query channel to each of its candidates.
        """
        inner = torch.matmul(query_spectra, torch.conj(self.candidate_spectra)).real
        query_norm = torch.linalg.norm(query_spectra, dim=-1)                # (C, B)
        candidate_norm = torch.linalg.norm(self.candidate_spectra, dim=-2)   # (C, N*M)
        return inner / (query_norm.unsqueeze(-1) * candidate_norm.unsqueeze(-2) + self.epsilon)

    def mask_overlapping_memory(self, similarity: torch.Tensor,
                                query_starts: torch.Tensor) -> torch.Tensor:
        """Hide memory samples whose horizon overlaps the query's own horizon.

        When training queries come from the same series as the memory, such a
        sample would hand the model its own target.
        """
        horizon = self.values.shape[1]
        candidate_starts = self.starts.repeat_interleave(self.num_relations)  # (N*M,)
        overlaps = (candidate_starts[None, :] - query_starts[:, None]).abs() < horizon  # (B, N*M)
        return similarity.masked_fill(overlaps.unsqueeze(0), float("-inf"))

    @torch.no_grad()
    def forward(self, query: torch.Tensor,
                query_starts: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve references for every channel of every query window.

        Args:
            query: (B, L, C) lookback windows.
            query_starts: (B,) start positions of the queries in the training series.
                Pass them during training so overlapping memory is excluded; leave
                None for validation and test queries.

        Returns:
            retrieved_keys: (C, B, R, L)
            retrieved_values: (C, B, R, H)
        """
        query_spectra = torch.fft.rfft(query, dim=1)[:, :self.num_frequencies].permute(2, 0, 1)
        similarity = self.spectral_similarity(query_spectra)                 # (C, B, N*M)
        if query_starts is not None:
            if self.starts is None:
                raise ValueError("fit() was called without starts; cannot exclude overlaps")
            similarity = self.mask_overlapping_memory(similarity, query_starts)
        best = torch.topk(similarity, k=self.num_references, dim=2).indices  # (C, B, R)

        sample_index = best // self.num_relations
        relation_slot = best % self.num_relations
        # Map the relation slot of channel c back to the channel id it points to.
        channel_index = torch.gather(
            self.related_channels.unsqueeze(1).expand(-1, best.shape[1], -1), 2, relation_slot)

        retrieved_keys = self.keys[sample_index, :, channel_index]           # (C, B, R, L)
        retrieved_values = self.values[sample_index, :, channel_index]       # (C, B, R, H)
        return retrieved_keys, retrieved_values
