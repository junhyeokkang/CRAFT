import numpy as np
import torch

from craft import CRAFT, ChannelWiseRetrieval, build_memory

LOOKBACK, HORIZON, CHANNELS = 96, 24, 6


def random_memory(num_samples=30, seed=0):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(num_samples, LOOKBACK, CHANNELS, generator=generator).cumsum(dim=1)
    values = torch.randn(num_samples, HORIZON, CHANNELS, generator=generator)
    return keys, values


def test_relation_graph_includes_the_channel_itself_by_default():
    keys, values = random_memory()
    retrieval = ChannelWiseRetrieval(num_relations=3, num_frequencies=8)
    retrieval.fit(keys, values)
    for channel, neighbours in enumerate(retrieval.related_channels.tolist()):
        assert neighbours[0] == channel
        assert len(set(neighbours)) == 3


def test_relation_graph_can_exclude_the_channel_itself():
    keys, values = random_memory()
    retrieval = ChannelWiseRetrieval(num_relations=3, num_frequencies=8, include_self=False)
    retrieval.fit(keys, values)
    for channel, neighbours in enumerate(retrieval.related_channels.tolist()):
        assert channel not in neighbours
        assert len(set(neighbours)) == 3


def test_references_come_from_the_matched_neighbour_channel():
    keys, values = random_memory()
    retrieval = ChannelWiseRetrieval(num_relations=3, num_references=1, num_frequencies=8)
    retrieval.fit(keys, values)
    retrieved_keys, retrieved_values = retrieval(keys[:4])
    for channel in range(CHANNELS):
        for batch in range(4):
            matched = [(sample, neighbour)
                       for sample in range(len(keys))
                       for neighbour in retrieval.related_channels[channel].tolist()
                       if torch.equal(retrieved_keys[channel, batch, 0], keys[sample, :, neighbour])]
            assert len(matched) == 1
            sample, neighbour = matched[0]
            assert torch.equal(retrieved_values[channel, batch, 0], values[sample, :, neighbour])


def test_build_memory_pairs_windows_with_their_future():
    series = np.arange(300 * CHANNELS, dtype=np.float32).reshape(300, CHANNELS)
    keys, values, starts = build_memory(series, LOOKBACK, HORIZON, stride=10)
    assert keys.shape[1:] == (LOOKBACK, CHANNELS) and values.shape[1:] == (HORIZON, CHANNELS)
    assert torch.equal(values[0, 0], keys[0, -1] + CHANNELS)
    assert starts[:3].tolist() == [0, 10, 20]


def test_training_queries_never_retrieve_their_own_future():
    rng = np.random.default_rng(1)
    series = rng.normal(size=(600, CHANNELS)).cumsum(axis=0).astype(np.float32)
    keys, values, starts = build_memory(series, LOOKBACK, HORIZON, stride=10)
    retrieval = ChannelWiseRetrieval(num_relations=3, num_frequencies=8)
    retrieval.fit(keys, values, starts)

    _, unmasked_values = retrieval(keys)
    own_future_found = any(torch.equal(unmasked_values[channel, sample, 0], values[sample, :, channel])
                           for channel in range(CHANNELS) for sample in range(len(keys)))
    assert own_future_found   # without masking a memory window finds itself

    _, masked_values = retrieval(keys, query_starts=starts)
    for channel in range(CHANNELS):
        for sample in range(len(keys)):
            for other in range(len(keys)):
                if abs(int(starts[other]) - int(starts[sample])) < HORIZON:
                    for neighbour in range(CHANNELS):
                        assert not torch.equal(masked_values[channel, sample, 0], values[other, :, neighbour])


def test_forward_shape_and_gradients():
    keys, values = random_memory()
    model = CRAFT(LOOKBACK, HORIZON, num_frequencies=8)
    model.build_memory(keys, values)
    forecast = model(keys[:5])
    assert forecast.shape == (5, HORIZON, CHANNELS)
    forecast.pow(2).mean().backward()
    assert model.reference_projection.weight.grad is not None
