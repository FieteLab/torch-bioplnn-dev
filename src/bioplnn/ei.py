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
        input_exc_dim: int,
        input_inh_dim: int,
        h_exc_dim: int,
        h_inh_dim: int,
        exc_kernel_size: tuple[int, int],
        inh_kernel_sizes: list[tuple[int, int]],
        use_fb: bool = False,
        pool_kernel_size: tuple[int, int] = (5, 5),
        pool_stride: tuple[int, int] = (2, 2),
        bias: bool = True,
        activation: str = "tanh",
    ):
        """
        Initialize the ConvRNNEICell.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            prev_exc_dim (int): Number of channels of previous excitatory tensor.
            prev_inh_dim (int): Number of channels of previous inhibitory tensor.
            cur_exc_dim (int): Number of channels of current excitatory tensor.
            cur_inh_dim (int): Number of channels of current inhibitory tensor.
            fb_exc_dim (int): Number of channels of fb excitatory tensor.
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
        self.input_exc_dim = input_exc_dim
        self.input_inh_dim = input_inh_dim
        self.h_exc_dim = h_exc_dim
        self.h_inh_dim = h_inh_dim
        self.use_fb = use_fb
        self.activation = get_activation_class(activation)()

        # Learnable membrane time constants for excitatory and inhibitory cell populations
        self.tau_exc = nn.Parameter(
            torch.randn((1, h_exc_dim, *input_size), requires_grad=True)
        )
        self.tau_inh = nn.Parameter(
            torch.randn((1, h_inh_dim, *input_size), requires_grad=True) + 0.5
        )

        # Initialize excitatory convolutional layers
        self.conv_exc = Conv2dPositive(
            in_channels=input_exc_dim + h_exc_dim,
            out_channels=h_exc_dim + h_inh_dim,
            kernel_size=exc_kernel_size,
            padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
            bias=bias,
        )

        if use_fb:
            self.fb_conv_exc = Conv2dPositive(
                in_channels=input_exc_dim,
                out_channels=h_exc_dim + h_inh_dim,
                kernel_size=exc_kernel_size,
                padding=(
                    exc_kernel_size[0] // 2,
                    exc_kernel_size[1] // 2,
                ),
                bias=bias,
            )

        # Initialize inhibitory convolutional layers with different kernel sizes
        self.convs_inh = nn.ModuleList()
        for kernel_size in inh_kernel_sizes:
            self.convs_inh.append(
                Conv2dPositive(
                    in_channels=input_inh_dim + h_inh_dim,
                    out_channels=h_exc_dim + h_inh_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    bias=False,
                )
            )

        if use_fb:
            self.fb_convs_inh = nn.ModuleList()
            for kernel_size in inh_kernel_sizes:
                self.fb_convs_inh.append(
                    Conv2dPositive(
                        in_channels=input_inh_dim,
                        out_channels=h_exc_dim + h_inh_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                        bias=False,
                    )
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
            torch.zeros(batch_size, (self.h_exc_dim), *self.input_size),
            torch.zeros(batch_size, (self.h_inh_dim), *self.input_size),
        )

    def init_output(self, batch_size):
        """
        Initializes the output tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.

        Returns:
            torch.Tensor: The initialized output tensor.
        """
        return (
            torch.zeros(
                batch_size,
                self.h_exc_dim,
                self.input_size[0] // 2,
                self.input_size[1] // 2,
            ),
            torch.zeros(
                batch_size,
                self.h_inh_dim,
                self.input_size[0] // 2,
                self.input_size[1] // 2,
            ),
        )

    def out_exc_dim_flat(self):
        return self.h_exc_dim * (self.input_size[0] // 2) * (self.input_size[1] // 2)

    def forward(
        self,
        input_exc: torch.Tensor,
        input_inh: torch.Tensor,
        h_exc: torch.Tensor,
        h_inh: torch.Tensor,
        fb_exc: torch.Tensor | None = None,
        fb_inh: torch.Tensor | None = None,
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
        # Compute the excitatory convolutions
        exc_input = torch.cat([input_exc, h_exc], dim=1)
        cnm = self.activation(self.conv_exc(exc_input))

        # Compute the feedback excitatory convolutions
        if self.use_fb:
            if fb_exc is None:
                raise ValueError("If use_fb is True, fb_exc must be provided.")
            cnm += self.activation(self.fb_conv_exc(fb_exc))

        # Compute the inhibitory convolutions
        inh_input = torch.cat([input_inh, h_inh], dim=1)
        inhibitions = torch.zeros_like(cnm)
        for conv in self.convs_inh:
            inhibitions += self.activation(conv(inh_input))

        # Compute the feedback inhibitory convolutions
        if self.use_fb:
            if fb_inh is None:
                raise ValueError("If use_fb is True, fb_inh must be provided.")
            for conv in self.fb_convs_inh:
                inhibitions += self.activation(conv(fb_inh))

        # Subtract contribution of inhibitory conv's from the cnm
        cnm_with_inh = cnm - inhibitions
        cnm_exc_with_inh, cnm_inh_with_inh = torch.split(
            cnm_with_inh, [self.h_exc_dim, self.h_inh_dim], dim=1
        )

        # Euler update for the cell state
        self.tau_exc = torch.sigmoid(self.tau_exc)
        h_next_exc = (1 - self.tau_exc) * h_exc + (self.tau_exc) * cnm_exc_with_inh

        self.tau_inh = torch.sigmoid(self.tau_inh)
        h_next_inh = (1 - self.tau_inh) * h_inh + (self.tau_inh) * cnm_inh_with_inh

        # Pool the output
        out = self.out_pool(torch.cat([h_next_exc, h_next_inh], dim=1))
        out_exc, out_inh = torch.split(out, [self.h_exc_dim, self.h_inh_dim], dim=1)

        return h_next_exc, h_next_inh, out_exc, out_inh


class Conv2dEIRNN(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        exc_dim: int | list[int],
        inh_dim: int | list[int],
        exc_kernel_size: tuple[int, int] | list[tuple[int, int]],
        inh_kernel_sizes: list[tuple[int, int]] | list[list[tuple[int, int]]],
        inh_scale_factors: list[int] | list[list[int]],
        num_layers: int,
        num_steps: int,
        num_classes: int,
        use_fb: bool = False,
        fb_exc_kernel_size: Optional[tuple[int, int] | list[tuple[int, int]]] = None,
        fb_inh_kernel_sizes: Optional[
            list[tuple[int, int]] | list[list[tuple[int, int]]]
        ] = None,
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
        self.exc_dims = self._extend_for_multilayer(exc_dim, num_layers)
        self.inh_dims = self._extend_for_multilayer(inh_dim, num_layers)
        self.exc_kernel_sizes = self._extend_for_multilayer(exc_kernel_size, num_layers)
        self.inh_kernel_sizes = self._extend_for_multilayer(
            inh_kernel_sizes, num_layers, depth=1
        )
        self.inh_scale_factors = self._extend_for_multilayer(
            inh_scale_factors, num_layers, depth=1
        )
        self.num_steps = num_steps
        self.pool_kernel_sizes = self._extend_for_multilayer(
            pool_kernel_size, num_layers
        )
        self.pool_strides = self._extend_for_multilayer(pool_stride, num_layers)
        self.biases = self._extend_for_multilayer(bias, num_layers)

        activation_class = get_activation_class(activation)
        self.activation = activation_class()

        if use_fb:
            self.fb_exc_kernel_sizes = self._extend_for_multilayer(
                fb_exc_kernel_size, num_layers
            )
            self.fb_inh_kernel_sizes = self._extend_for_multilayer(
                fb_inh_kernel_sizes, num_layers, depth=1
            )
            if (
                fb_adjacency.dim() != 2
                or fb_adjacency.shape[0] != num_layers
                or fb_adjacency.shape[1] != num_layers
            ):
                raise ValueError(
                    "The number of layers must match the first dimension of fb_adjacency."
                )
            self.fb_adjacency = []
            self.fb_exc_convs = dict()
            self.fb_inh_convs = dict()
            for row in fb_adjacency:
                row = row.nonzero().squeeze().tolist()
                self.fb_adjacency.append(row)
                for j in row:
                    upsample = nn.Upsample(
                        size=self.layers[j].input_size, mode="bilinear"
                    )
                    conv_exc = Conv2dPositive(
                        in_channels=self.layers[i].h_exc_dim,
                        out_channels=self.layers[j].input_exc_dim,
                        kernel_size=1,
                        bias=True,
                    )
                    conv_inh = Conv2dPositive(
                        in_channels=self.layers[i].h_inh_dim,
                        out_channels=self.layers[j].input_inh_dim,
                        kernel_size=1,
                        bias=True,
                    )
                    self.fb_exc_convs[(i, j)] = nn.Sequential(upsample, conv_exc)
                    self.fb_exc_convs[(i, j)] = nn.Sequential(upsample, conv_inh)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                Conv2dEIRNNCell(
                    input_size=input_size,
                    input_exc_dim=(input_dim if i == 0 else self.exc_dims[i - 1]),
                    input_inh_dim=(0 if i == 0 else self.inh_dims[i - 1]),
                    h_exc_dim=self.exc_dims[i],
                    h_inh_dim=self.inh_dims[i],
                    exc_kernel_size=self.exc_kernel_sizes[i],
                    inh_kernel_sizes=self.inh_kernel_sizes[i],
                    use_fb=use_fb,
                    pool_kernel_size=self.pool_kernel_sizes[i],
                    pool_stride=self.pool_strides[i],
                    bias=self.biases[i],
                    activation=activation,
                )
            )
            input_size = (input_size[0] // 2, input_size[1] // 2)

        self.out_layer = nn.Sequential(
            nn.Linear(
                self.layers[-1].out_exc_dim_flat(),
                fc_dim,
            ),
            activation_class(),
            nn.Dropout(),
            nn.Linear(fc_dim, num_classes),
        )

    def _init_hidden(self, batch_size):
        init_excs = []
        init_inhs = []
        for layer in self.layers:
            init_exc, init_inh = layer.init_hidden(batch_size)
            init_excs.append(init_exc)
            init_inhs.append(init_inh)
        return init_excs, init_inhs

    def _init_fb(self, batch_size):
        return self._init_hidden(batch_size)

    def _init_output(self, batch_size):
        init_excs = []
        init_inhs = []
        for layer in self.layers:
            init_exc, init_inh = layer.init_output(batch_size)
            init_excs.append(init_exc)
            init_inhs.append(init_inh)
        return init_excs, init_inhs

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

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
        h_excs, h_inhs = self._init_hidden(batch_size)

        for input in (cue, mixture):
            out_excs, out_inhs = self._init_output(batch_size)
            fb_prev_excs, fb_prev_inhs = self._init_fb(batch_size)
            fb_excs, fb_inhs = self._init_fb(batch_size)
            for t in range(self.num_steps):
                upper = min(t, len(self.layers) - 1)
                lower = 0
                # lower = max(len(self.layers) - self.num_steps + t, 0)
                for i in range(upper, lower - 1, -1):
                    layer = self.layers[i]
                    (
                        h_excs[i],
                        h_inhs[i],
                        out_excs[i],
                        out_inhs[i],
                    ) = layer(
                        input_exc=input if i == 0 else out_excs[i - 1],
                        input_inh=(
                            torch.zeros_like(input) if i == 0 else out_inhs[i - 1]
                        ),
                        h_exc=h_excs[i],
                        h_inh=h_inhs[i],
                        fb_exc=fb_prev_excs[i] if self.use_fb else None,
                        fb_inh=fb_prev_inhs[i] if self.use_fb else None,
                    )
                    for j in self.fb_adjacency[i]:
                        fb_excs[j] += self.fb_exc_convs[(i, j)](out_excs[i])
                        fb_inhs[j] += self.fb_inh_convs[(i, j)](out_inhs[i])
                fb_prev_excs = fb_excs
                fb_prev_inhs = fb_inhs
                fb_excs, fb_inhs = self._init_fb(batch_size)

        out = self.out_layer(out_excs[-1].flatten(1))
        return out
