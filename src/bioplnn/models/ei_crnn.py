import warnings
from math import ceil
from typing import Optional

import torch
import torch.nn as nn

from bioplnn.utils import (
    expand_list,
    get_activation_class,
)


class Conv2dPositive(nn.Conv2d):
    """
    A convolutional layer that ensures positive weights and biases.

    Args:
        *args: Positional arguments passed to the base `nn.Conv2d` class.
        **kwargs: Keyword arguments passed to the base `nn.Conv2d` class.
    """

    def forward(self, *args, **kwargs):
        """
        Forward pass of the layer.

        Args:
            *args: Positional arguments passed to the base `nn.Conv2d` class.
            **kwargs: Keyword arguments passed to the base `nn.Conv2d` class.

        Returns:
            torch.Tensor: The output tensor.
        """
        self.weight.data = torch.relu(self.weight.data)
        if self.bias is not None:
            self.bias.data = torch.relu(self.bias.data)
        return super().forward(*args, **kwargs)


class Conv2dEIRNNCell(nn.Module):
    """
    A convolutional recurrent neural network (RNN) cell with excitatory and inhibitory neurons.

    Args:
        input_size (tuple[int, int]): Height and width of the input tensor.
        input_dim (int): Number of channels of the input tensor.
        h_pyr_dim (int, optional): Number of channels of the hidden state of the excitatory pyramidal tensor. Default is 16.
        h_inter_dims (list[int], optional): Number of channels for the hidden state tensors of each of the interneuron classes.
            The length of this list specifies the number of interneuron classes. For example, [4, 8, 2, 4] instantiates
            interneuron classes 1, 2, 3, and 4, with sizes 4, 8, 2, and 4, respectively (Default is [4]).
        fb_dim (int, optional): Number of channels of the feedback excitatory tensor. Default is 0.
        inter_mode (str, optional): Mode for interneuron processing. Can be 'half' or 'same'. Default is 'half'.
        exc_kernel_size (tuple[int, int], optional): Size of the kernel for excitatory convolution. Default is (5, 5).
        inh_kernel_size (tuple[int, int], optional): Size of the kernel for inhibitory convolution. Default is (5, 5).
        use_three_compartments (bool, optional): Whether to use three compartments for pyramidal neurons (soma, basal, apical). Default is False.
        immediate_inhibition (bool, optional): Whether to use immediate inhibition. Default is False.
        pool_kernel_size (tuple[int, int], optional): Size of the kernel for pooling. Default is (5, 5).
        pool_stride (tuple[int, int], optional): Stride for pooling. Default is (2, 2).
        bias (bool, optional): Whether to add bias. Default is True.
        pre_inh_activation (Optional[str], optional): Activation function before inhibitory convolution. Default is "tanh".
        post_inh_activation (Optional[str], optional): Activation function after inhibitory convolution. Default is None.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        h_pyr_dim: int = 16,
        h_inter_dims: list[int] = [4],
        fb_dim: int = 0,
        inter_mode: str = "half",
        exc_kernel_size: tuple[int, int] = (5, 5),
        inh_kernel_size: tuple[int, int] = (5, 5),
        use_three_compartments: bool = False,
        immediate_inhibition: bool = False,
        exc_rectify: bool = False,
        inh_rectify: bool = True,
        pool_kernel_size: tuple[int, int] = (5, 5),
        pool_stride: tuple[int, int] = (2, 2),
        bias: bool = True,
        pre_inh_activation: Optional[str] = "tanh",
        post_inh_activation: Optional[str] = None,
        post_integration_activation: Optional[str] = None,
    ):
        super().__init__()
        # Save the parameters
        self.input_size = input_size
        if inter_mode == "half":
            self.inter_size = (ceil(input_size[0] / 2), ceil(input_size[1] / 2))
        elif inter_mode == "same":
            self.inter_size = input_size
        else:
            raise ValueError("inter_mode must be 'half' or 'same'.")
        self.out_size = (
            ceil(input_size[0] / pool_stride[0]),
            ceil(input_size[1] / pool_stride[1]),
        )
        self.input_dim = input_dim
        self.h_pyr_dim = h_pyr_dim
        self.out_dim = h_pyr_dim
        self.h_inter_dims = h_inter_dims if h_inter_dims is not None else []
        self.use_h_inter = self.h_inter_dims and not immediate_inhibition
        self.fb_dim = fb_dim
        self.use_fb = fb_dim > 0
        self.immediate_inhibition = immediate_inhibition
        self.use_three_compartments = use_three_compartments
        self.pool_stride = pool_stride
        try:
            self.pre_inh_activation = get_activation_class(pre_inh_activation)()
        except ValueError:
            self.pre_inh_activation = nn.Sequential(
                *(
                    get_activation_class(activation)()
                    for activation in pre_inh_activation
                )
            )
        try:
            self.post_inh_activation = get_activation_class(post_inh_activation)()
        except ValueError:
            self.post_inh_activation = nn.Sequential(
                *(
                    get_activation_class(activation)()
                    for activation in post_inh_activation
                )
            )
        try:
            self.post_integration_activation = get_activation_class(
                post_integration_activation
            )()
        except Exception:
            self.post_integration_activation = nn.Sequential(
                *(
                    get_activation_class(activation)()
                    for activation in post_integration_activation
                )
            )

        # Value check
        if len(self.h_inter_dims) not in (0, 1, 2, 4):
            raise ValueError(
                "h_inter_dims must be None or a list/tuple of length 0, 1, 2, or 4."
                f"Got length {len(self.h_inter_dims)}."
            )
        if len(self.h_inter_dims) == 4:
            if not self.use_fb:
                warnings.warn(
                    """
                    If h_inter_dims has length 4, fb_dim must be greater than 0. 
                    Disabling interneuron types 2 and 3.
                    """
                )
                self.h_inter_dims = self.h_inter_dims[:2]
            if not use_three_compartments:
                raise ValueError(
                    "If h_inter_dims has length 4, use_three_compartments must be True."
                )

        self.h_inter_dims_sum = sum(self.h_inter_dims)

        if self.immediate_inhibition and len(self.h_inter_dims) not in (0, 1):
            raise ValueError(
                "If immediate_inhibition is True, h_inter_dims must be None or a list/tuple of length 0 or 1."
            )

        # Learnable membrane time constants for excitatory and inhibitory cell populations
        self.tau_pyr = nn.Parameter(torch.randn((1, h_pyr_dim, *input_size)))
        if self.use_h_inter:
            self.tau_inter = nn.Parameter(
                torch.randn((1, self.h_inter_dims_sum, *self.inter_size))
            )

        # Rectify the weights and biases if specified
        if exc_rectify:
            Conv2dExc = Conv2dPositive
        else:
            Conv2dExc = nn.Conv2d

        # Initialize excitatory convolutional layers
        if use_three_compartments:
            conv_exc_pyr_dim = h_pyr_dim
            self.conv_exc_pyr_input = Conv2dExc(
                in_channels=input_dim,
                out_channels=h_pyr_dim,
                kernel_size=exc_kernel_size,
                padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                bias=bias,
            )
            if self.use_fb:
                self.conv_exc_pyr_fb = Conv2dExc(
                    in_channels=fb_dim,
                    out_channels=h_pyr_dim,
                    kernel_size=exc_kernel_size,
                    padding=(
                        exc_kernel_size[0] // 2,
                        exc_kernel_size[1] // 2,
                    ),
                    bias=bias,
                )
        else:
            conv_exc_pyr_dim = h_pyr_dim + input_dim + fb_dim
        self.conv_exc_pyr = Conv2dExc(
            in_channels=conv_exc_pyr_dim,
            out_channels=h_pyr_dim,
            kernel_size=exc_kernel_size,
            padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
            bias=bias,
        )

        if self.h_inter_dims:
            if len(self.h_inter_dims) == 4:
                self.conv_exc_input_inter = Conv2dExc(
                    in_channels=input_dim,
                    out_channels=self.h_inter_dims[2],
                    kernel_size=exc_kernel_size,
                    stride=2 if inter_mode == "half" else 1,
                    padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                    bias=bias,
                )
                self.conv_exc_inter_fb = Conv2dExc(
                    in_channels=fb_dim,
                    out_channels=self.h_inter_dims[3],
                    kernel_size=exc_kernel_size,
                    stride=2 if inter_mode == "half" else 1,
                    padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                    bias=bias,
                )

            self.conv_exc_inter = Conv2dExc(
                in_channels=h_pyr_dim,
                out_channels=self.h_inter_dims_sum,
                kernel_size=exc_kernel_size,
                stride=2 if inter_mode == "half" else 1,
                padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                bias=bias,
            )

            if inh_rectify:
                Conv2dInh = Conv2dPositive
            else:
                Conv2dInh = nn.Conv2d

            self.convs_inh = nn.ModuleList()
            self.inh_out_dims = [
                h_pyr_dim,
                self.h_inter_dims_sum,
                h_pyr_dim,
                h_pyr_dim,
            ]
            for i, h_inter_dim in enumerate(self.h_inter_dims):
                conv = Conv2dInh(
                    in_channels=h_inter_dim,
                    out_channels=self.inh_out_dims[i],
                    kernel_size=inh_kernel_size,
                    padding=(inh_kernel_size[0] // 2, inh_kernel_size[1] // 2),
                    bias=bias,
                )
                if inter_mode == "half" and i != 1:
                    conv = nn.Sequential(
                        nn.Upsample(size=input_size, mode="bilinear"), conv
                    )
                if i in (2, 3):
                    conv2 = Conv2dInh(
                        in_channels=h_inter_dim,
                        out_channels=(self.h_inter_dims[3 if i == 2 else 2]),
                        kernel_size=inh_kernel_size,
                        padding=(inh_kernel_size[0] // 2, inh_kernel_size[1] // 2),
                        bias=bias,
                    )
                    conv_mod = nn.Module()
                    conv_mod.conv1 = conv
                    conv_mod.conv2 = conv2
                    conv = conv_mod
                self.convs_inh.append(conv)

        # Initialize output pooling layer
        self.out_pool = nn.AvgPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=(pool_kernel_size[0] // 2, pool_kernel_size[1] // 2),
        )

    def init_hidden(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the hidden state tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.
            device (torch.device, optional): The device to initialize the tensor on. Default is None.
            init_mode (str, optional): The initialization mode. Can be "zeros" or "normal". Default is "zeros".

        Returns:
            torch.Tensor: The initialized excitatory hidden state tensor.
            torch.Tensor: The initialized inhibitory hidden state tensor.
        """

        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")
        return (
            func(batch_size, self.h_pyr_dim, *self.input_size, device=device),
            (
                func(
                    batch_size,
                    self.h_inter_dims_sum,
                    *self.inter_size,
                    device=device,
                )
                if self.use_h_inter
                else None
            ),
        )

    def init_fb(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the output tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.
            device (torch.device, optional): The device to initialize the tensor on. Default is None.
            init_mode (str, optional): The initialization mode. Can be "zeros" or "normal". Default is "zeros".

        Returns:
            torch.Tensor: The initialized output tensor.
        """
        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")
        return (
            func(batch_size, self.fb_dim, *self.input_size, device=device)
            if self.use_fb
            else None
        )

    def init_out(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the output tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.
            device (torch.device, optional): The device to initialize the tensor on. Default is None.
            init_mode (str, optional): The initialization mode. Can be "zeros" or "normal". Default is "zeros".

        Returns:
            torch.Tensor: The initialized output tensor.
        """
        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")
        return func(batch_size, self.out_dim, *self.out_size, device=device)

    def forward(
        self,
        input: torch.Tensor,
        h_pyr: torch.Tensor,
        h_inter: Optional[torch.Tensor] = None,
        fb: Optional[torch.Tensor] = None,
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
        if self.use_fb and fb is None:
            raise ValueError("If use_fb is True, fb must be provided.")

        # Compute the excitations for pyramidal cells
        if self.use_three_compartments:
            exc_pyr_soma = self.pre_inh_activation(self.conv_exc_pyr(h_pyr))
            exc_pyr_basal = self.pre_inh_activation(self.conv_exc_pyr_input(input))
            if self.use_fb:
                exc_pyr_apical = self.pre_inh_activation(self.conv_exc_pyr_fb(fb))
        else:
            exc_cat = [h_pyr, input, fb] if self.use_fb else [h_pyr, input]
            exc_pyr_soma = self.pre_inh_activation(
                self.conv_exc_pyr(torch.cat(exc_cat, dim=1))
            )

        inhs = [0] * 4
        if self.h_inter_dims:
            # Compute the excitations for interneurons
            inter_activation = (
                nn.Tanh() if self.immediate_inhibition else self.pre_inh_activation
            )
            exc_inter = inter_activation(self.conv_exc_inter(h_pyr))
            if len(self.h_inter_dims) == 4:
                exc_input_inter = self.pre_inh_activation(
                    self.conv_exc_input_inter(input)
                )
                exc_fb_inter = self.pre_inh_activation(self.conv_exc_inter_fb(fb))

            # Compute the inhibitions for all neurons
            h_inters = h_inter
            if self.immediate_inhibition:
                if h_inter is not None:
                    raise ValueError(
                        "If h_inter_dims is provided and immediate_inhibition is True, h_inter must not be provided."
                    )
                h_inters = exc_inter
            elif h_inter is None:
                raise ValueError(
                    "If h_inter_dims is provided and immediate_inhibition is False, h_inter must be provided."
                )

            h_inters = torch.split(
                h_inters,
                self.h_inter_dims,
                dim=1,
            )
            inh_inter_2 = inh_inter_3 = 0
            for i in range(len(self.h_inter_dims)):
                conv = self.convs_inh[i]
                if i in (2, 3):
                    conv, conv2 = conv.conv1, conv.conv2
                    if i == 2:
                        inh_inter_3 = self.pre_inh_activation(conv2(h_inters[i]))
                    else:
                        inh_inter_2 = self.pre_inh_activation(conv2(h_inters[i]))
                inhs[i] = self.pre_inh_activation(conv(h_inters[i]))
        inh_pyr_soma, inh_inter, inh_pyr_basal, inh_pyr_apical = inhs

        # Compute the new pyramidal cell hidden state
        pyr_soma = self.post_inh_activation(exc_pyr_soma - inh_pyr_soma)
        if self.use_three_compartments:
            pyr_basal = self.post_inh_activation(exc_pyr_basal - inh_pyr_basal)
            pyr_apical = (
                self.post_inh_activation(exc_pyr_apical - inh_pyr_apical)
                if self.use_fb
                else 0
            )
            pyr_soma = pyr_soma + pyr_apical + pyr_basal
            pyr_soma /= 3
        h_pyr_new = self.post_integration_activation(pyr_soma)

        # Compute the Euler update for the pyramidal cell hidden state
        tau_pyr = torch.sigmoid(self.tau_pyr)
        h_pyr = (1 - tau_pyr) * h_pyr + tau_pyr * h_pyr_new

        # Compute the new interneuron cell hidden state
        if self.use_h_inter:
            h_inter_new = exc_inter - inh_inter
            if len(self.h_inter_dims) == 4:
                # Add excitations and inhibitions to interneuron 2
                start = sum(self.h_inter_dims[:2])
                end = start + self.h_inter_dims[2]
                h_inter_new[:, start:end, ...] = (
                    h_inter_new[:, start:end, ...] + exc_input_inter - inh_inter_2
                )
                # Add excitations and inhibitions to interneuron 3
                start = sum(self.h_inter_dims[:3])
                h_inter_new[:, start:, ...] = (
                    h_inter_new[:, start:, ...] + exc_fb_inter - inh_inter_3
                )
            h_inter_new = self.post_integration_activation(
                self.post_inh_activation(h_inter_new)
            )
            # Compute the Euler update for the interneuron cell hidden state
            tau_inter = torch.sigmoid(self.tau_inter)
            h_inter = (1 - tau_inter) * h_inter + tau_inter * h_inter_new

        # Pool the output
        out = self.out_pool(h_pyr)

        return out, h_pyr, h_inter


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
        use_three_compartments: bool,
        immediate_inhibition: bool,
        num_layers: int,
        inter_mode: str,
        layer_time_delay: bool,
        exc_rectify: Optional[str],
        inh_rectify: Optional[str],
        hidden_init_mode: str,
        fb_init_mode: str,
        out_init_mode: str,
        fb_adjacency: Optional[torch.Tensor],
        pool_kernel_size: list[int, int] | list[list[int, int]],
        pool_stride: list[int, int] | list[list[int, int]],
        bias: bool | list[bool],
        pre_inh_activation: Optional[str],
        post_inh_activation: Optional[str],
        post_integration_activation: Optional[str],
        batch_first: bool,
    ):
        """
        Initialize the Conv2dEIRNN.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            h_pyr_dim (int | list[int]): Number of channels of the pyramidal neurons or a list of number of channels for each layer.
            h_inter_dims (list[int] | list[list[int]]): Number of channels of the interneurons or a list of number of channels for each layer.
            fb_dim (int | list[int]): Number of channels of the feedback activationsor a list of number of channels for each layer.
            exc_kernel_size (list[int, int] | list[list[int, int]]): Size of the kernel for excitatory convolutions or a list of kernel sizes for each layer.
            inh_kernel_size (list[int, int] | list[list[int, int]]): Size of the kernel for inhibitory convolutions or a list of kernel sizes for each layer.
            num_layers (int): Number of layers in the RNN.
            num_steps (int): Number of iterations to perform in each layer.
            num_classes (int): Number of output classes. If None, the activations of the final layer at the last time step will be output.
            fb_adjacency (Optional[torch.Tensor], optional): Adjacency matrix for feedback connections.
            pool_kernel_size (list[int, int] | list[list[int, int]], optional): Size of the kernel for pooling or a list of kernel sizes for each layer.
            pool_stride (list[int, int] | list[list[int, int]], optional): Stride of the pooling operation or a list of strides for each layer.
            bias (bool | list[bool], optional): Whether or not to add the bias or a list of booleans indicating whether to add bias for each layer.
            activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported.
            fc_dim (int, optional): Dimension of the fully connected layer.
        """
        super().__init__()
        self.h_pyr_dims = expand_list(h_pyr_dim, num_layers)
        self.h_inter_dims = expand_list(
            h_inter_dims, num_layers, depth=0 if h_inter_dims is None else 1
        )
        self.fb_dims = expand_list(fb_dim, num_layers)
        self.fb_adjacency = fb_adjacency
        self.exc_kernel_sizes = expand_list(exc_kernel_size, num_layers, depth=1)
        self.inh_kernel_sizes = expand_list(inh_kernel_size, num_layers, depth=1)
        self.immediate_inhibition = immediate_inhibition
        self.layer_time_delay = layer_time_delay
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.hidden_init_mode = hidden_init_mode
        self.fb_init_mode = fb_init_mode
        self.out_init_mode = out_init_mode
        self.pool_kernel_sizes = expand_list(pool_kernel_size, num_layers, depth=1)
        self.pool_strides = expand_list(pool_stride, num_layers, depth=1)
        self.biases = expand_list(bias, num_layers)

        self.input_sizes = [input_size]
        for i in range(num_layers - 1):
            self.input_sizes.append(
                (
                    ceil(self.input_sizes[i][0] / self.pool_strides[i][0]),
                    ceil(self.input_sizes[i][1] / self.pool_strides[i][1]),
                )
            )

        self.receives_fb = [False] * num_layers
        self.fb_adjacency = [[]] * num_layers
        if fb_adjacency is not None:
            if not any(self.fb_dims):
                raise ValueError(
                    "fb_adjacency must be provided if and only if fb_dim is provided for at least one layer."
                )
            try:
                fb_adjacency = torch.load(fb_adjacency)
            except AttributeError:
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
                raise ValueError("fb_adjacency must be a non-zero tensor if provided.")

            if exc_rectify:
                Conv2dFb = Conv2dPositive
            else:
                Conv2dFb = nn.Conv2d
            self.fb_convs = nn.ModuleDict()
            for i, row in enumerate(fb_adjacency):
                row = row.nonzero().squeeze(1).tolist()
                self.fb_adjacency[i] = row
                for j in row:
                    self.receives_fb[j] = True
                    upsample = nn.Upsample(size=self.input_sizes[j], mode="bilinear")
                    conv_exc = Conv2dFb(
                        in_channels=self.h_pyr_dims[i],
                        out_channels=self.fb_dims[j],
                        kernel_size=1,
                        bias=True,
                    )
                    self.fb_convs[f"fb_conv_{i}_{j}"] = nn.Sequential(
                        upsample, conv_exc
                    )
        else:
            if any(self.fb_dims):
                raise ValueError(
                    "fb_adjacency must be provided if and only if fb_dim is provided for at least one layer."
                )

        self.layers = nn.ModuleList()
        self.pertubations = nn.ModuleList()
        self.pertubations_inter = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                Conv2dEIRNNCell(
                    input_size=self.input_sizes[i],
                    input_dim=(input_dim if i == 0 else self.h_pyr_dims[i - 1]),
                    h_pyr_dim=self.h_pyr_dims[i],
                    h_inter_dims=self.h_inter_dims[i],
                    fb_dim=self.fb_dims[i] if self.receives_fb[i] else 0,
                    inter_mode=inter_mode,
                    exc_kernel_size=self.exc_kernel_sizes[i],
                    inh_kernel_size=self.inh_kernel_sizes[i],
                    use_three_compartments=use_three_compartments,
                    immediate_inhibition=immediate_inhibition,
                    exc_rectify=exc_rectify,
                    inh_rectify=inh_rectify,
                    pool_kernel_size=self.pool_kernel_sizes[i],
                    pool_stride=self.pool_strides[i],
                    bias=self.biases[i],
                    pre_inh_activation=pre_inh_activation,
                    post_inh_activation=post_inh_activation,
                    post_integration_activation=post_integration_activation,
                )
            )

    def _init_hidden(self, batch_size, init_mode="zeros", device=None):
        h_pyrs = []
        h_inters = []
        for layer in self.layers:
            h_pyr, h_inter = layer.init_hidden(
                batch_size, init_mode=init_mode, device=device
            )
            h_pyrs.append(h_pyr)
            h_inters.append(h_inter)
        return h_pyrs, h_inters

    def _init_fb(self, batch_size, init_mode="zeros", device=None):
        fbs = []
        for layer in self.layers:
            fb = layer.init_fb(batch_size, init_mode=init_mode, device=device)
            fbs.append(fb)
        return fbs

    def _init_out(self, batch_size, init_mode="zeros", device=None):
        outs = []
        for layer in self.layers:
            out = layer.init_out(batch_size, init_mode=init_mode, device=device)
            outs.append(out)
        return outs

    def _init_state(
        self,
        out_0,
        h_pyr_0,
        h_inter_0,
        fb_0,
        num_steps,
        batch_size,
        device=None,
    ):
        outs = [[None] * num_steps for _ in self.layers]
        h_pyrs = [[None] * num_steps for _ in self.layers]
        h_inters = [[None] * num_steps for _ in self.layers]
        fbs = [
            [0 if self.receives_fb[i] else None] * num_steps
            for i in range(self.num_layers)
        ]

        if out_0 is None:
            out_0 = [None] * self.num_layers
        if h_pyr_0 is None:
            h_pyr_0 = [None] * self.num_layers
        if h_inter_0 is None:
            h_inter_0 = [None] * self.num_layers
        if fb_0 is None:
            fb_0 = [None] * self.num_layers

        if all(x is None for x in out_0):
            out_0 = self._init_out(
                batch_size, init_mode=self.out_init_mode, device=device
            )
        if all(x is None for x in h_pyr_0 or all(x is None for x in h_inter_0)):
            h_pyr_tmp, h_inter_tmp = self._init_hidden(
                batch_size, init_mode=self.hidden_init_mode, device=device
            )
            h_pyr_0 = h_pyr_tmp if all(x is None for x in h_pyr_0) else h_pyr_0
            h_inter_0 = h_inter_tmp if all(x is None for x in h_inter_0) else h_inter_0
        if all(x is None for x in fb_0):
            fb_0 = self._init_fb(batch_size, init_mode=self.fb_init_mode, device=device)

        for i in range(self.num_layers):
            h_pyrs[i][-1] = h_pyr_0[i]
            h_inters[i][-1] = h_inter_0[i]
            fbs[i][-1] = fb_0[i]
            outs[i][-1] = out_0[i]

        return outs, h_pyrs, h_inters, fbs

    def _format_x(self, x, num_steps):
        # Check if the input is consistent with the number of steps
        if x.dim() == 4:
            if num_steps is None or num_steps < 1:
                raise ValueError(
                    "If x is 4D, num_steps must be provided and greater than 0"
                )
            x = x.unsqueeze(0).expand((num_steps, -1, -1, -1, -1))
        elif x.dim() == 5:
            if self.batch_first:
                x = x.transpose(0, 1)
            if num_steps is not None and num_steps != x.shape[0]:
                raise ValueError(
                    "If x is 5D and num_steps is provided, it must match the sequence length."
                )
            num_steps = x.shape[0]
        else:
            raise ValueError(
                "The input must be a 4D tensor or a 5D tensor with sequence length."
            )
        return x, num_steps

    @staticmethod
    def _modulation_identity(x, i, t):
        return x

    def _format_modulation_fns(
        self, modulation_out_fn, modulation_pyr_fn, modulation_inter_fn
    ):
        modulation_fns = [modulation_out_fn, modulation_pyr_fn, modulation_inter_fn]
        for i, fn in enumerate(modulation_fns):
            if fn is None:
                modulation_fns[i] = self._modulation_identity
            elif not callable(fn):
                raise ValueError("modulation_fns must be callable or None.")

        return modulation_fns

    def _format_outputs(self, outs, h_pyrs, h_inters, fbs):
        for i in range(self.num_layers):
            outs[i] = torch.stack(outs[i])
            h_pyrs[i] = torch.stack(h_pyrs[i])
            if self.h_inter_dims[i] and not self.immediate_inhibition:
                h_inters[i] = torch.stack(h_inters[i])  # TODO: Check if this is correct
            else:
                assert not self.h_inter_dims[i] or self.immediate_inhibition
                assert all(x is None for x in h_inters[i])
                h_inters[i] = None
            if self.receives_fb[i]:
                fbs[i] = torch.stack(fbs[i])
            else:
                assert all(x is None for x in fbs[i])
                fbs[i] = None
            if self.batch_first:
                outs[i] = outs[i].transpose(0, 1)
                h_pyrs[i] = h_pyrs[i].transpose(0, 1)
                if self.h_inter_dims[i] and not self.immediate_inhibition:
                    h_inters[i] = h_inters[i].transpose(0, 1)
                if self.receives_fb[i]:
                    fbs[i] = fbs[i].transpose(0, 1)
        if all(x is None for x in h_inters):
            h_inters = None
        if all(x is None for x in fbs):
            fbs = None

        return outs, h_pyrs, h_inters, fbs

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        out_0: Optional[list[torch.Tensor]] = None,
        h_pyr_0: Optional[list[torch.Tensor]] = None,
        h_inter_0: Optional[list[torch.Tensor]] = None,
        fb_0: Optional[list[torch.Tensor]] = None,
        modulation_out_fn: Optional[torch.Tensor] = None,
        modulation_pyr_fn: Optional[torch.Tensor] = None,
        modulation_inter_fn: Optional[torch.Tensor] = None,
        return_all_layers_out: bool = False,
    ):
        """
        Performs forward pass of the Conv2dEIRNN.

        Args:

        Returns:
            torch.Tensor: Output tensor after pooling of shape (b, n), where n is the number of classes.
        """
        device = x.device

        x, num_steps = self._format_x(x, num_steps)

        batch_size = x.shape[1]

        outs, h_pyrs, h_inters, fbs = self._init_state(
            out_0, h_pyr_0, h_inter_0, fb_0, num_steps, batch_size, device=device
        )

        modulation_out_fn, modulation_pyr_fn, modulation_inter_fn = (
            self._format_modulation_fns(
                modulation_out_fn, modulation_pyr_fn, modulation_inter_fn
            )
        )

        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                # Apply additive modulation
                outs[i][t] = modulation_out_fn(outs[i][t], i, t)
                h_pyrs[i][t] = modulation_pyr_fn(h_pyrs[i][t], i, t)
                h_inters[i][t] = modulation_inter_fn(h_inters[i][t], i, t)

                # Compute layer update and output
                outs[i][t], h_pyrs[i][t], h_inters[i][t] = layer(
                    input=(
                        x[t]
                        if i == 0
                        else (
                            outs[i - 1][t - 1]
                            if self.layer_time_delay
                            else outs[i - 1][t]
                        )
                    ),
                    h_pyr=h_pyrs[i][t - 1],
                    h_inter=h_inters[i][t - 1],
                    fb=fbs[i][t - 1],
                )
                # Apply feedback
                for j in self.fb_adjacency[i]:
                    fbs[j][t] = fbs[j][t] + self.fb_convs[f"fb_conv_{i}_{j}"](
                        outs[i][t]
                    )

        outs, h_pyrs, h_inters, fbs = self._format_outputs(outs, h_pyrs, h_inters, fbs)

        if not return_all_layers_out:
            outs = outs[-1]

        return outs, h_pyrs, h_inters, fbs
