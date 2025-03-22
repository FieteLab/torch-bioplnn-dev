import torch

from bioplnn.models.spatially_embedded import (
    SpatiallyEmbeddedArea,
    SpatiallyEmbeddedAreaConfig,
    SpatiallyEmbeddedRNN,
)


def test_spatially_embedded_area_initialization():
    # Test initialization with default parameters
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    area = SpatiallyEmbeddedArea(config)
    assert area is not None


def test_spatially_embedded_rnn_initialization():
    # Test initialization with default parameters
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    rnn = SpatiallyEmbeddedRNN(num_areas=1, area_configs=[config])
    assert rnn is not None


def test_spatially_embedded_area_forward():
    # Test forward pass
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    area = SpatiallyEmbeddedArea(config)
    input_tensor = torch.rand(1, 3, 32, 32)
    neuron_state = area.init_neuron_state(1)
    output, new_neuron_state = area.forward(input_tensor, neuron_state)
    assert output.shape == (1, 16, 32, 32)


def test_spatially_embedded_rnn_forward():
    # Test forward pass
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    rnn = SpatiallyEmbeddedRNN(num_areas=1, area_configs=[config])
    input_tensor = torch.rand(5, 1, 3, 32, 32)
    output_states, neuron_states, feedback_states = rnn.forward(input_tensor)
    assert output_states[0].shape == (5, 1, 16, 32, 32)


# TODO: More extensive testing needs to be done for the SpatiallyEmbeddedRNN, SpatiallyEmbeddedArea, and SpatiallyEmbeddedAreaConfig
