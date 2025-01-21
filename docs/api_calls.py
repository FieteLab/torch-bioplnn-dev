import torch
from torchvision.datasets import MNIST

from bioplnn import Conv2dEIRNN

test_dataset = MNIST(root="./data", train=False, download=True)
img, label = test_dataset[0]


"""
Conv2dEIRNN
Implements a deep convolutional recurrent neural network with excitatory-inhibitory
neurons (EIRNN).

Args:
    in_size (tuple[int, int]): Size of the input data (height, width).
    in_channels (int): Number of input channels.
    h_pyr_channels (int | list[int]): Number of channels in the pyramidal cell \
        hidden state for each layer.
    h_inter_channels (list[int] | list[list[int]]): Number of channels in the
        interneuron hidden state for each layer.
    fb_channels (int | list[int]): Number of channels for the feedback input for \
        each layer.
    exc_kernel_size (tuple[int, int] | tuple[tuple[int, int]]): Kernel size for
        excitatory convolutions in each layer.
    inh_kernel_size (tuple[int, int] | tuple[tuple[int, int]]): Kernel size for
        inhibitory convolutions in each layer.
    fb_kernel_size (tuple[int, int] | tuple[tuple[int, int]]): Kernel size for
        feedback convolutions in each layer.
    use_three_compartments (bool): Whether to use a three-compartment model for
        pyramidal cells.
    immediate_inhibition (bool): Whether interneurons provide immediate inhibition.
    num_layers (int): Number of layers in the RNN.
    inter_mode (str): Mode for handling interneuron size relative to input size
        ('half' or 'same').
    layer_time_delay (bool): Whether to introduce a time delay between layers.
    exc_rectify bool: Activation function for excitatory weights (e.g., \
        'relu').
    inh_rectify bool: Activation function for inhibitory weights (e.g., \
        'relu').
    hidden_init_mode (str): Initialization mode for hidden states ('zeros' or
        'normal').
    fb_init_mode (str): Initialization mode for feedback input ('zeros' or \
        'normal').
    out_init_mode (str): Initialization mode for output ('zeros' or 'normal').
    fb_adjacency (Optional[torch.Tensor]): Adjacency matrix for feedback \
        connections.
    pool_kernel_size (tuple[int, int] | tuple[tuple[int, int]]): Kernel size for \
        pooling in each layer.
    pool_stride (tuple[int, int] | tuple[tuple[int, int]]): Stride for pooling in \
        each layer.
    pool_global (bool | tuple[int, int]): Whether to use global pooling in each layer.
    pre_inh_activation (Optional[str]): Activation function applied before \
        inhibition.
    post_inh_activation (Optional[str]): Activation function applied after \
        inhibition (to pyramidal cell).
    post_integration_activation (Optional[str]): Activation function applied \
        after integration (to pyramidal and interneurons).
    tau_mode (Optional[str]): Mode for handling tau values ('channel', 'spatial', 'channel_spatial').
    fb_activation (Optional[str]): Activation function for feedback connections.
    bias (bool | tuple[bool]): Whether to add bias for convolutions in each layer.
    layer_time_delay (bool): Whether to introduce a time delay between layers.
    fb_adjacency (Optional[tuple[tuple[int | bool]] | torch.Tensor] = None): \
        Adjacency matrix for feedback connections.
    hidden_init_mode (str): Initialization mode for hidden states ('zeros' or
        'normal').
    fb_init_mode (str): Initialization mode for feedback input ('zeros' or \
        'normal').
    out_init_mode (str): Initialization mode for output ('zeros' or 'normal').
    batch_first (bool): Whether the input tensor has batch dimension as the \
        first dimension.
"""


################################################################################

# One layer one excitatory neuron type (channel)

rnn = Conv2dEIRNN(
    in_size=(28, 28),
    in_channels=1,
    h_pyr_channels=1,
    exc_kernel_size=(3, 3),
    # Optional
    exc_rectify=False,
    pool_kernel_size=(3, 3),
    pool_stride=(2, 2),
    # or pool_global=True,
    tau_mode="channel",
    bias=True,
    hidden_init_mode="zeros",
    out_init_mode="zeros",
    batch_first=False,
)

outs, h_pyrs, h_inters, fbs = rnn(
    x=img.unsqueeze(0),  # (1, 1, 28, 28)
    # Optional
    num_steps=10,
    out_0=torch.ones(1, 1, 14, 14),  # (1, 1, 1, 1) when global_pool=True
    h_pyr_0=torch.randn(1, 1, 28, 28),
)

################################################################################

# One layer multiple (16) neuron types

rnn = Conv2dEIRNN(
    # As above, but
    h_pyr_channels=16,
)

outs, h_pyrs, h_inters, fbs = rnn(
    # As above
)

################################################################################

# One layer EI neuron type distinction

rnn = Conv2dEIRNN(
    # As above, and
    h_inter_channels=4,
    # Optional: As above, and
    h_inter_kernel_size=(3, 3),
    inter_mode="same",
    immediate_inhibition=False,
    inh_rectify=False,
)


outs, h_pyrs, h_inters, fbs = rnn(
    # As above, and
    h_inter_0=torch.randn(1, 16, 28, 28),
)

################################################################################

# Circuit Motifs (Brainstorming)
# fmt: off
rnn = Conv2dEIRNN(
    # As above, but
    h_pyr_channels=(16, 16),
    h_inter_channels=(4, 4, 4),
    fb_adjacency=(
        (1, 1, 1, 0, 1),
        (1, 1, 0, 1, 1),
        (1, 0, 1, 0, 0),
        (0, 1, 0, 1, 0),
        (1, 1, 1, 1, 1)
    ),
    post_integration_activation="tanh",
)
# fmt: on

outs, h_pyrs, h_inters, fbs = rnn(
    # As above
)

################################################################################

# Multiple layers

rnn = Conv2dEIRNN(
    # As above, but
    h_pyr_channels=((16, 16), (32, 32)),
    h_inter_channels=((4, 4, 4), (8, 8, 8)),
    num_layers=2,
    # Optional: As above, and
    exc_kernel_size=((5, 5), (3, 3)),
    inh_kernel_size=((5, 5), (3, 3)),
    layer_time_delay=False,
    pool_kernel_size=((3, 3), None),
    pool_stride=((2, 2), None),
    pool_global=(False, True),
)

outs, h_pyrs, h_inters, fbs = rnn(
    # Optional: As above, and
    return_all_layers_out=True,
)

################################################################################

# Multiple layers with feedback

rnn = Conv2dEIRNN(
    # As above, but
    fb_channels=(16, 32),
    fb_adjacency=((0, 0), (1, 0)),
    # Optional: As above, and
    fb_kernel_size=((5, 5), (3, 3)),
    fb_activation="relu",
    fb_init_mode="zeros",
)

outs, h_pyrs, h_inters, fbs = rnn(
    # As above
)
