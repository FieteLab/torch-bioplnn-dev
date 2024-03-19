from math import prod, ceil
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Variable

from bioplnn.utils import get_activation_class


class Conv2dPositive(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.relu(self.weight.data)
        if self.bias is not None:
            self.bias.data = torch.relu(self.bias.data)
        return super().forward(x)


class ConvTranspose2dPositive(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.weight.data = torch.relu(self.weight.data)
        if self.bias is not None:
            self.bias.data = torch.relu(self.bias.data)
        return super().forward(*args, **kwargs)


class Conv2dEIRNNCell(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        h_pyr_dim: int = 4,
        h_inter_dims: tuple[int] = (4),
        fb_dim: int = 0,
        exc_kernel_size: tuple[int, int] = (5, 5),
        inh_kernel_size: tuple[int, int] = (5, 5),
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
        self.inter_size = (ceil(input_size[0] / 2), ceil(input_size[1] / 2))
        self.input_dim = input_dim
        self.h_pyr_dim = h_pyr_dim
        self.h_inter_dims = h_inter_dims
        self.num_inter = len(h_inter_dims)
        if self.num_inter not in (1, 2, 4):
            raise ValueError(
                "The length of h_inter_dims must be 1, 2, or 4. "
                f"Got {self.num_inter}."
            )
        self.h_inter_dims_sum = sum(h_inter_dims)
        self.fb_dim = fb_dim
        self.use_fb = fb_dim > 0
        self.pool_stride = pool_stride
        self.activation = get_activation_class(activation)()
        self.out_shape = (
            1,
            h_pyr_dim,
            ceil(input_size[0] / 2),
            ceil(input_size[1] / 2),
        )

        # Learnable membrane time constants for excitatory and inhibitory cell populations
        self.tau_pyr = nn.Parameter(
            torch.randn((1, h_pyr_dim, *input_size), requires_grad=True)
        )
        self.tau_inter = nn.Parameter(
            torch.randn(
                (1, self.h_inter_dims_sum, *self.inter_size),
                requires_grad=True,
            )
            + 0.5
        )

        # Initialize excitatory convolutional layers
        self.conv_exc_pyr = Conv2dPositive(
            in_channels=input_dim + h_pyr_dim,
            out_channels=h_pyr_dim,
            kernel_size=exc_kernel_size,
            padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
            bias=bias,
        )

        self.conv_exc_inter = Conv2dPositive(
            in_channels=input_dim + h_pyr_dim,
            out_channels=self.h_inter_dims_sum,
            kernel_size=exc_kernel_size,
            stride=2,
            padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
            bias=bias,
        )

        if self.use_fb:
            self.conv_exc_pyr_fb = Conv2dPositive(
                in_channels=fb_dim,
                out_channels=h_pyr_dim,
                kernel_size=exc_kernel_size,
                padding=(
                    exc_kernel_size[0] // 2,
                    exc_kernel_size[1] // 2,
                ),
                bias=bias,
            )
            self.conv_exc_inter_fb = Conv2dPositive(
                in_channels=fb_dim,
                out_channels=self.h_inter_dims_sum,
                kernel_size=exc_kernel_size,
                stride=2,
                padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                bias=bias,
            )

        # Initialize inhibitory convolutional layers
        self.convs_inh = nn.ModuleList()
        self.inh_out_dims = [
            h_pyr_dim,
            self.h_inter_dims_sum,
            h_pyr_dim,
            h_pyr_dim,
        ]
        for i, h_inter_dim in enumerate(h_inter_dims):
            conv = ConvTranspose2dPositive(
                in_channels=h_inter_dim,
                out_channels=self.inh_out_dims[i],
                kernel_size=inh_kernel_size,
                stride=1 if i == 1 else 2,
                padding=(inh_kernel_size[0] // 2, inh_kernel_size[1] // 2),
                bias=bias,
            )
            self.convs_inh.append(conv)

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
            torch.zeros(batch_size, self.h_inter_dims_sum, *self.inter_size),
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

    def forward(
        self,
        input: torch.Tensor,
        h_pyr: torch.Tensor,
        h_inter: torch.Tensor,
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
        # TODO: Add flag for toggling input to interneurons
        batch_size = input.shape[0]
        exc = torch.cat([input, h_pyr], dim=1)
        exc_pyr_basal = self.conv_exc_pyr(exc)
        exc_inter = self.conv_exc_inter(exc)

        if self.use_fb:
            if fb is None:
                raise ValueError("If use_fb is True, fb_exc must be provided.")
            exc_pyr_apical = self.conv_exc_pyr_fb(fb)
            exc_fb_inter = self.conv_exc_inter_fb(fb)
        else:
            exc_pyr_apical = 0
            exc_fb_inter = 0

        # Compute the inhibitions
        h_inters = torch.split(h_inter, self.h_inter_dims, dim=1)
        inhs = [0] * 4
        for i in range(self.num_inter):
            inhs[i] = self.convs_inh[i](
                h_inters[i],
                output_size=(
                    batch_size,
                    self.inh_out_dims[i],
                    self.inter_size[0] if i == 1 else self.input_size[0],
                    self.inter_size[1] if i == 1 else self.input_size[1],
                ),
            )
        inh_pyr_soma = inhs[0]
        inh_inter = 0
        inh_pyr_basal = 0
        inh_pyr_apical = 0
        if self.num_inter >= 2:
            inh_inter = inhs[1]
        if self.num_inter == 4:
            inh_pyr_basal = inhs[2]
            inh_pyr_apical = inhs[3]
        # TODO: Have less
        # Computer candidate neural memory (cnm) states
        pyr_basal = torch.relu(exc_pyr_basal - inh_pyr_basal)
        try:
            pyr_apical = torch.relu(exc_pyr_apical - inh_pyr_apical)
        except ValueError:
            pyr_apical = 0
        cnm_pyr = torch.relu(
            self.activation(pyr_apical + pyr_basal) - inh_pyr_soma
        )
        cnm_inter = self.activation(exc_inter + exc_fb_inter - inh_inter)

        # Euler update for the cell state
        tau_pyr = torch.sigmoid(self.tau_pyr)
        h_next_pyr = (1 - tau_pyr) * h_pyr + tau_pyr * cnm_pyr

        tau_inter = torch.sigmoid(self.tau_inter)
        h_next_inter = (1 - tau_inter) * h_inter + tau_inter * cnm_inter

        # Pool the output
        out = self.out_pool(h_next_pyr)

        return h_next_pyr, h_next_inter, out


class Conv2dEIRNN(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        h_pyr_dim: int | list[int],
        h_inter_dims: list[int] | list[list[int]],
        fb_dim: int | list[int],
        exc_kernel_size: list[int, int] | list[list[int, int]],
        inh_kernel_size: list[int, int] | list[list[int, int]],
        num_layers: int,
        num_steps: int,
        num_classes: int,
        fb_adjacency: Optional[torch.Tensor] = None,
        pool_kernel_size: list[int, int] | list[list[int, int]] = (5, 5),
        pool_stride: list[int, int] | list[list[int, int]] = (2, 2),
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
        self.h_pyr_dims = self._extend_for_multilayer(h_pyr_dim, num_layers)
        self.h_inter_dims = self._extend_for_multilayer(
            h_inter_dims, num_layers, depth=1
        )
        self.fb_dims = self._extend_for_multilayer(fb_dim, num_layers)
        self.exc_kernel_sizes = self._extend_for_multilayer(
            exc_kernel_size, num_layers, depth=1
        )
        self.inh_kernel_sizes = self._extend_for_multilayer(
            inh_kernel_size, num_layers, depth=1
        )
        self.num_steps = num_steps
        self.pool_kernel_sizes = self._extend_for_multilayer(
            pool_kernel_size, num_layers, depth=1
        )
        self.pool_strides = self._extend_for_multilayer(
            pool_stride, num_layers, depth=1
        )
        self.biases = self._extend_for_multilayer(bias, num_layers)

        self.input_sizes = [input_size]
        for i in range(num_layers - 1):
            self.input_sizes.append(
                (
                    ceil(self.input_sizes[i][0] / self.pool_strides[i][0]),
                    ceil(self.input_sizes[i][1] / self.pool_strides[i][1]),
                )
            )

        activation_class = get_activation_class(activation)
        self.activation = activation_class()

        self.use_fb = [False] * num_layers
        if fb_adjacency is not None:
            try:
                fb_adjacency = torch.load(fb_adjacency)
            except:
                fb_adjacency = torch.tensor(fb_adjacency)
            if (
                fb_adjacency.dim() != 2
                or fb_adjacency.shape[0] != num_layers
                or fb_adjacency.shape[1] != num_layers
            ):
                raise ValueError(
                    "The the dimensions of fb_adjacency must match number of layers."
                )
            if fb_adjacency.count_nonzero() == 0:
                raise ValueError(
                    "fb_adjacency must be a non-zero tensor if provided."
                )

            self.fb_adjacency = []
            self.fb_convs = dict()
            for i, row in enumerate(fb_adjacency):
                row = row.nonzero().squeeze(1).tolist()
                self.fb_adjacency.append(row)
                for j in row:
                    self.use_fb[j] = True
                    upsample = nn.Upsample(
                        size=self.input_sizes[j], mode="bilinear"
                    )
                    conv_exc = Conv2dPositive(
                        in_channels=self.h_pyr_dims[i],
                        out_channels=self.fb_dims[j],
                        kernel_size=1,
                        bias=True,
                    )
                    self.fb_convs[(i, j)] = nn.Sequential(upsample, conv_exc)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                Conv2dEIRNNCell(
                    input_size=self.input_sizes[i],
                    input_dim=(
                        input_dim if i == 0 else self.h_pyr_dims[i - 1]
                    ),
                    h_pyr_dim=self.h_pyr_dims[i],
                    h_inter_dims=self.h_inter_dims[i],
                    fb_dim=self.fb_dims[i] if self.use_fb[i] else 0,
                    exc_kernel_size=self.exc_kernel_sizes[i],
                    inh_kernel_size=self.inh_kernel_sizes[i],
                    pool_kernel_size=self.pool_kernel_sizes[i],
                    pool_stride=self.pool_strides[i],
                    bias=self.biases[i],
                    activation=activation,
                )
            )

        self.out_layer = nn.Sequential(
            nn.Linear(
                prod(self.layers[-1].out_shape[1:]),
                fc_dim,
            ),
            activation_class(),
            nn.Dropout(),
            nn.Linear(fc_dim, num_classes),
        )

    def _init_hidden(self, batch_size):
        h_pyrs = []
        h_inters = []
        for layer in self.layers:
            h_pyr, h_inter = layer.init_hidden(batch_size)
            h_pyrs.append(h_pyr)
            h_inters.append(h_inter)
        return h_pyrs, h_inters

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
            if not isinstance(inner, (list, tuple)):
                raise ValueError("depth exceeds the depth of param.")
            inner = inner[0]

        if not isinstance(inner, (list, tuple)):
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
        h_pyrs, h_inters = self._init_hidden(batch_size)
        fbs_prev = self._init_fb(batch_size)
        fbs = self._init_fb(batch_size)

        for input in (cue, mixture):
            outs = [None] * len(self.layers)
            for t in range(self.num_steps):
                upper = min(t, len(self.layers) - 1)
                lower = 0
                # lower = max(len(self.layers) - self.num_steps + t, 0)
                for i in range(upper, lower - 1, -1):
                    layer = self.layers[i]
                    (h_pyrs[i], h_inters[i], outs[i]) = layer(
                        input=input if i == 0 else outs[i - 1],
                        h_pyr=h_pyrs[i],
                        h_inter=h_inters[i],
                        fb=fbs_prev[i] if self.use_fb[i] else None,
                    )
                    for j in self.fb_adjacency[i]:
                        fbs[j] += self.fb_convs[(i, j)](outs[i])
                fbs_prev = fbs
                fbs = self._init_fb(batch_size)

        out = self.out_layer(outs[-1].flatten(1))
        return out
