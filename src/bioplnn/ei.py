from math import prod
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Variable

from bioplnn.utils import get_activation_class


class LinearExc(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight = torch.relu(self.weight)
        return super().forward(x)


class LinearInh(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight = -torch.relu(-self.weight)
        return super().forward(x)


class Conv2dPositive(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight = torch.relu(self.weight)
        if self.bias is not None:
            self.bias = torch.relu(self.bias)
        return super().forward(x)


class Conv2dEIRNNCell(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        h_pyr_dim: int,
        h_int_1_dim: int,
        h_int_2_dim: int,
        exc_kernel_size: tuple[int, int],
        inh_kernel_size: tuple[int, int],
        use_fb: bool = False,
        pool_kernel_size: tuple[int, int] = (5, 5),
        pool_stride: tuple[int, int] = (2, 2),
        bias: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize the ConvRNNEICell.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            prev_pyr_dim (int): Number of channels of previous excitatory tensor.
            prev_inh_dim (int): Number of channels of previous inhibitory tensor.
            cur_pyr_dim (int): Number of channels of current excitatory tensor.
            cur_inh_dim (int): Number of channels of current inhibitory tensor.
            fb_pyr_dim (int): Number of channels of fb excitatory tensor.
            fb_inh_dim (int): Number of channels of fb inhibitory tensor.
            exc_kernel_size (tuple[int, int]): Size of the kernel for excitatory convolution.
            inhib_kernel_sizes (list[tuple[int, int]]): Sizes of the kernels for inhibitory convolutions.
            bias (bool, optional): Whether or not to add the bias. Default is True.
            euler (bool, optional): Whether to use Euler updates for the cell state. Default is True.
            dt (int, optional): Time step for Euler updates. Default is 1.
            activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported. Default is "tanh".
        """
        super().__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.h_pyr_dim = h_pyr_dim
        self.h_int_1_dim = h_int_1_dim
        self.h_int_2_dim = h_int_2_dim
        self.use_fb = use_fb
        self.pool_stride = pool_stride
        self.activation = get_activation_class(activation)()

        # Learnable membrane time constants for excitatory and inhibitory cell populations
        self.tau_pyr = nn.Parameter(
            torch.randn((1, h_pyr_dim, *input_size), requires_grad=True)
        )
        self.tau_int_1 = nn.Parameter(
            torch.randn((1, h_int_1_dim, *input_size), requires_grad=True)
            + 0.5
        )
        self.tau_int_2 = nn.Parameter(
            torch.randn((1, h_int_2_dim, *input_size), requires_grad=True)
            + 0.5
        )

        # Initialize excitatory convolutional layers
        self.conv_exc = Conv2dPositive(
            in_channels=input_dim + h_pyr_dim,
            out_channels=h_pyr_dim + h_int_1_dim + h_int_2_dim,
            kernel_size=exc_kernel_size,
            padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
            bias=bias,
        )

        if use_fb:
            self.conv_exc_fb = Conv2dPositive(
                in_channels=h_pyr_dim,
                out_channels=h_pyr_dim + h_int_1_dim + h_int_2_dim,
                kernel_size=exc_kernel_size,
                padding=(
                    exc_kernel_size[0] // 2,
                    exc_kernel_size[1] // 2,
                ),
                bias=bias,
            )

        # Initialize inhibitory convolutional layers
        self.conv_inh_1 = Conv2dPositive(
            in_channels=h_int_1_dim,
            out_channels=h_pyr_dim + h_int_1_dim + h_int_2_dim,
            kernel_size=inh_kernel_size,
            stride=1,
            padding=(inh_kernel_size[0] // 2, inh_kernel_size[1] // 2),
            bias=False,
        )
        self.conv_inh_2 = Conv2dPositive(
            in_channels=h_int_1_dim,
            out_channels=h_pyr_dim + h_int_1_dim + h_int_2_dim,
            kernel_size=inh_kernel_size,
            stride=1,
            padding=(inh_kernel_size[0] // 2, inh_kernel_size[1] // 2),
            bias=False,
        )

        # Initialize output pooling layer
        self.out_pool = nn.AvgPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=(pool_kernel_size[0] // 2, pool_kernel_size[1] // 2),
        )

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.

        Returns:
            torch.Tensor: The initialized excitatory hidden state tensor.
            torch.Tensor: The initialized inhibitory hidden state tensor.
        """
        return (
            torch.zeros(batch_size, self.h_pyr_dim, *self.input_size),
            torch.zeros(batch_size, self.h_int_1_dim, *self.input_size),
            torch.zeros(batch_size, self.h_int_2_dim, *self.input_size),
        )

    def init_fb(self, batch_size):
        """
        Initializes the output tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.

        Returns:
            torch.Tensor: The initialized output tensor.
        """
        return torch.zeros(batch_size, self.h_pyr_dim, *self.input_size)

    def out_shape(self):
        return (
            1,
            self.h_pyr_dim,
            (self.input_size[0] // self.pool_stride[0]),
            (self.input_size[1] // self.pool_stride[1]),
        )

    def forward(
        self,
        input: torch.Tensor,
        h_pyr: torch.Tensor,
        h_int_1: torch.Tensor,
        h_int_2: torch.Tensor,
        fb: torch.Tensor = None,
    ):
        """
        Performs forward pass of the cRNN_EI model.

        Args:
            input (torch.Tensor): Input tensor of shape (b, c, h, w).
                The input is actually the target_model.
            h (torch.Tensor): Current hidden and cell states respectively
                of shape (b, c_hidden, h, w).

        Returns:
            torch.Tensor: Next hidden state of shape (b, c_hidden*2, h, w).
            torch.Tensor: Output tensor after pooling of shape (b, c_hidden*2, h', w').
        """
        # Compute the excitations
        cnm_exc = torch.cat([input, h_pyr], dim=1)
        cnm_exc = self.conv_exc(cnm_exc)
        cnm_exc_pyr_b, cnm_exc_int_1, cnm_exc_int_2 = torch.split(
            cnm_exc,
            [self.h_pyr_dim, self.h_int_1_dim, self.h_int_2_dim],
            dim=1,
        )

        if self.use_fb:
            if fb is None:
                raise ValueError("If use_fb is True, fb_exc must be provided.")
            cnm_exc_fb = self.conv_exc_fb(fb)
            cnm_exc_pyr_a, cnm_exc_fb_int_1, cnm_exc_fb_int_2 = torch.split(
                cnm_exc_fb,
                [self.h_pyr_dim, self.h_int_1_dim, self.h_int_2_dim],
                dim=1,
            )

        # Compute the inhibitions
        cnm_inh_1 = self.conv_inh_1(h_int_1)
        cnm_inh_2 = self.conv_inh_2(h_int_2)
        (
            cnm_inh_pyr_a,
            cnm_inh_1_int_1,
            cnm_inh_1_int_2,
        ) = torch.split(
            cnm_inh_1,
            [self.h_pyr_dim, self.h_int_1_dim, self.h_int_2_dim],
            dim=1,
        )
        (
            cnm_inh_pyr_b,
            cnm_inh_2_int_1,
            cnm_inh_2_int_2,
        ) = torch.split(
            cnm_inh_2,
            [self.h_pyr_dim, self.h_int_1_dim, self.h_int_2_dim],
            dim=1,
        )

        # Computer candidate neural memory (cnm) states
        cnm_pyr_b = self.activation(cnm_exc_pyr_b - cnm_inh_pyr_b)
        if self.use_fb:
            cnm_pyr_a = self.activation(cnm_exc_pyr_a - cnm_inh_pyr_a)
            cnm_pyr = cnm_pyr_a + cnm_pyr_b
            cnm_int_1 = self.activation(
                cnm_exc_int_1
                + cnm_exc_fb_int_1
                - cnm_inh_1_int_1
                - cnm_inh_2_int_1
            )
            cnm_int_2 = self.activation(
                cnm_exc_int_2
                + cnm_exc_fb_int_2
                - cnm_inh_1_int_2
                - cnm_inh_2_int_2
            )
        else:
            cnm_pyr = cnm_pyr_b
            cnm_int_1 = self.activation(
                cnm_exc_int_1 - cnm_inh_1_int_1 - cnm_inh_2_int_1
            )
            cnm_int_2 = self.activation(
                cnm_exc_int_2 - cnm_inh_1_int_2 - cnm_inh_2_int_2
            )

        # Euler update for the cell state
        self.tau_pyr = torch.sigmoid(self.tau_pyr)
        h_next_pyr = (1 - self.tau_pyr) * h_pyr + self.tau_pyr * cnm_pyr

        self.tau_int_1 = torch.sigmoid(self.tau_int_1)
        h_next_int_1 = (
            1 - self.tau_int_1
        ) * h_int_1 + self.tau_int_1 * cnm_int_1

        self.tau_int_2 = torch.sigmoid(self.tau_int_2)
        h_next_int_2 = (
            1 - self.tau_int_2
        ) * h_int_2 + self.tau_int_2 * cnm_int_2

        # Pool the output
        out = self.out_pool(h_next_pyr)

        return h_next_pyr, h_next_int_1, h_next_int_2, out


class Conv2dEIRNN(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        pyr_dim: int | list[int],
        int_1_dim: int | list[int],
        int_2_dim: int | list[int],
        exc_kernel_size: tuple[int, int] | list[tuple[int, int]],
        inh_kernel_size: tuple[int, int] | list[tuple[int, int]],
        num_layers: int,
        num_steps: int,
        num_classes: int,
        use_fb: bool = False,
        fb_adjacency: Optional[torch.Tensor] = None,
        pool_kernel_size: tuple[int, int] | list[tuple[int, int]] = (5, 5),
        pool_stride: tuple[int, int] | list[tuple[int, int]] = (2, 2),
        bias: bool | list[bool] = True,
        activation: str = "relu",
        fc_dim: int = 1024,
    ):
        """
        Initialize the Conv2dEIRNN.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int): Number of channels of hidden tensor.
            num_layers (int): Number of layers in the RNN.
            num_iterations (int): Number of iterations to perform in each layer.
            exc_kernel_size (tuple[int, int]): Size of the kernel for excitatory convolution.
            inhib_kernel_sizes (list[tuple[int, int]]): Sizes of the kernels for inhibitory convolutions.
            use_h_prev (bool, optional): Whether to use previous hidden states as input. Default is False.
            use_fb (bool, optional): Whether to use fb from previous layers as input. Default is False.
            pool_kernel_size (tuple[int, int], optional): Size of the kernel for pooling. Default is (5, 5).
            pool_stride (tuple[int, int], optional): Stride of the pooling operation. Default is (2, 2).
            bias (bool, optional): Whether or not to add the bias. Default is True.
            euler (bool, optional): Whether to use Euler updates for the cell state. Default is True.
            dt (int, optional): Time step for Euler updates. Default is 1.
            activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported. Default is "tanh".
        """
        super().__init__()
        self.input_size = input_size
        self.pyr_dims = self._extend_for_multilayer(pyr_dim, num_layers)
        self.int_1_dims = self._extend_for_multilayer(int_1_dim, num_layers)
        self.int_2_dims = self._extend_for_multilayer(int_2_dim, num_layers)
        self.exc_kernel_sizes = self._extend_for_multilayer(
            exc_kernel_size, num_layers
        )
        self.inh_kernel_sizes = self._extend_for_multilayer(
            inh_kernel_size, num_layers
        )
        self.num_steps = num_steps
        self.pool_kernel_sizes = self._extend_for_multilayer(
            pool_kernel_size, num_layers
        )
        self.pool_strides = self._extend_for_multilayer(
            pool_stride, num_layers
        )
        self.biases = self._extend_for_multilayer(bias, num_layers)

        activation_class = get_activation_class(activation)
        self.activation = activation_class()

        if use_fb:
            if (
                fb_adjacency.dim() != 2
                or fb_adjacency.shape[0] != num_layers
                or fb_adjacency.shape[1] != num_layers
            ):
                raise ValueError(
                    "The the dimensions of fb_adjacency must match number of layers."
                )
            self.fb_adjacency = []
            self.fb_convs = dict()
            for row in fb_adjacency:
                row = row.nonzero().squeeze().tolist()
                self.fb_adjacency.append(row)
                for j in row:
                    upsample = nn.Upsample(
                        size=self.layers[j].input_size, mode="bilinear"
                    )
                    conv_exc = Conv2dPositive(
                        in_channels=self.layers[i].h_pyr_dim,
                        out_channels=self.layers[j].h_pyr_dim,
                        kernel_size=1,
                        bias=True,
                    )
                    self.fb_convs[(i, j)] = nn.Sequential(upsample, conv_exc)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                Conv2dEIRNNCell(
                    input_size=input_size,
                    input_dim=(input_dim if i == 0 else self.pyr_dims[i - 1]),
                    h_pyr_dim=self.pyr_dims[i],
                    h_int_1_dim=self.int_1_dims[i],
                    h_int_2_dim=self.int_2_dims[i],
                    exc_kernel_size=self.exc_kernel_sizes[i],
                    inh_kernel_size=self.inh_kernel_sizes[i],
                    use_fb=use_fb,
                    pool_kernel_size=self.pool_kernel_sizes[i],
                    pool_stride=self.pool_strides[i],
                    bias=self.biases[i],
                    activation=activation,
                )
            )
            input_size = (
                input_size[0] // self.pool_strides[i][0],
                input_size[1] // self.pool_strides[i][1],
            )

        self.out_layer = nn.Sequential(
            nn.Linear(
                prod(self.layers[-1].out_shape()[1:]),
                fc_dim,
            ),
            activation_class(),
            nn.Dropout(),
            nn.Linear(fc_dim, num_classes),
        )

    def _init_hidden(self, batch_size):
        h_pyrs = []
        h_int_1s = []
        h_int_2s = []
        for layer in self.layers:
            h_pyr, h_int_1, h_int_2 = layer.init_hidden(batch_size)
            h_pyrs.append(h_pyr)
            h_int_1s.append(h_int_1)
            h_int_2s.append(h_int_2)
        return h_pyrs, h_int_1s, h_int_2s

    def _init_fb(self, batch_size):
        h_fbs = []
        for layer in self.layers:
            h_fb = layer.init_fb(batch_size)
            h_fbs.append(h_fb)
        return h_fbs

    @staticmethod
    def _extend_for_multilayer(param, num_layers, depth=0):
        inner = param
        for _ in range(depth):
            if not isinstance(inner, list):
                raise ValueError("depth exceeds the depth of param.")
            inner = inner[0]

        if not isinstance(inner, list):
            param = [param] * num_layers
        return param

    def forward(self, cue: torch.Tensor, mixture: torch.Tensor):
        """
        Performs forward pass of the Conv2dEIRNN.

        Args:
            input (torch.Tensor): Input tensor of shape (b, c, h, w).

        Returns:
            torch.Tensor: Output tensor after pooling of shape (b, hidden_dim*2, h', w').
        """
        batch_size = cue.shape[0]
        h_pyrs, h_int_1s, h_int_2s = self._init_hidden(batch_size)

        for input in (cue, mixture):
            outs = [None] * len(self.layers)
            fbs_prev = self._init_fb(batch_size)
            fbs = self._init_fb(batch_size)
            for t in range(self.num_steps):
                upper = min(t, len(self.layers) - 1)
                lower = 0
                # lower = max(len(self.layers) - self.num_steps + t, 0)
                for i in range(upper, lower - 1, -1):
                    layer = self.layers[i]
                    (h_pyrs[i], h_int_1s[i], h_int_2s[i], outs[i]) = layer(
                        input=input if i == 0 else outs[i - 1],
                        h_pyr=h_pyrs[i],
                        h_int_1=h_int_1s[i],
                        h_int_2=h_int_2s[i],
                        fb=fbs_prev[i] if self.use_fb else None,
                    )
                    for j in self.fb_adjacency[i]:
                        fbs[j] += self.fb_convs[(i, j)](outs[i])
                fbs_prev = fbs
                fbs = self._init_fb(batch_size)

        out = self.out_layer(outs[-1].flatten(1))
        return out
