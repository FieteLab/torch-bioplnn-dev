import warnings
from math import ceil
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from bioplnn.utils import (
    expand_list,
    get_activation,
)


class Conv2dRectify(nn.Conv2d):
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


def circuit_connectivity_df(
    num_exc: int,
    num_inh: int,
    use_fb: bool,
    circuit_connectivity: Optional[list[list[int]]] = None,
):
    row_labels = (
        ["input"]
        + (["fb"] if use_fb else [])
        + [f"exc_{i}" for i in range(num_exc)]
        + [f"inh_{i}" for i in range(num_inh)]
    )
    column_labels = (
        [f"exc_{i}" for i in range(num_exc)]
        + [f"inh_{i}" for i in range(num_inh)]
        + ["output"]
    )

    if circuit_connectivity is None:
        circuit_connectivity = torch.full(
            (len(row_labels), len(column_labels)), False, dtype=torch.bool
        )

    df = pd.DataFrame(
        circuit_connectivity, index=row_labels, columns=column_labels
    )
    df.index.name = "from"
    df.columns.name = "to"

    return df


class Conv2dEIRNNCell(nn.Module):
    """
    Implements a 2D convolutional recurrent neural network cell with excitatory \
    and inhibitory neurons.

    Args:
        in_size (tuple[int, int]): Size of the input data (height, width).
        in_channels (int): Number of input channels.
        h_exc_channels (int, optional): Number of channels in the excitatory cell \
            hidden state. Defaults to 16.
        h_inh_channels (list[int], optional): List of number of channels for \
            each inhibitory neuron type. Defaults to [16]. The length can be 0, 1, 2, \
            or 4, indicating the number of inhibitory neuron types. A length of 0 \
            indicates no inhibitory neurons.
        fb_channels (int, optional): Number of channels for the feedback input. \
            Defaults to 0.
        inh_mode (str, optional): Mode for handling inhibitory neuron size relative \
            to input size. Must be 'half' or 'same'. Defaults to 'half'.
        rectify (bool | list[bool], optional): List of booleans indicating whether to \
            rectify the corresponding layer. Defaults to False.
        exc_kernel_size (tuple[int, int], optional): Kernel size for excitatory \
            convolutions. Defaults to (3, 3).
        inh_kernel_size (tuple[int, int], optional): Kernel size for inhibitory \
            convolutions. Defaults to (3, 3).
        pool_kernel_size (tuple[int, int], optional): Kernel size for the output \
            pooling layer. Defaults to (3, 3).
        pool_stride (tuple[int, int], optional): Stride for the output pooling \
            layer. Defaults to (2, 2).
        pre_inh_activation (str | list[str], optional): Activation function \
            applied before inhibition. Defaults to "relu". If a list is \
            provided, the activations are applied sequentially.
        post_inh_activation (str | list[str], optional): Activation function \
            applied after inhibition. Defaults to "tanh". If a list is provided, \
            the activations are applied sequentially.
        post_integration_activation (str | list[str], optional): Activation \
            function applied after integration. Defaults to None. If a list is \
            provided, the activations are applied sequentially.
        tau_mode (str, optional): Mode for handling membrane time constants. \
            Defaults to "channel". Options are None, "channel", "spatial", and "channel_spatial".
        bias (bool, optional): Whether to add a bias term for convolutions. \
            Defaults to True.

    Raises:
        ValueError: If invalid arguments are provided.
    """

    def __init__(
        self,
        in_size: tuple[int, int],
        in_channels: int,
        out_channels: int,
        h_exc_channels: int | list[int] = 16,
        h_inh_channels: Optional[int | list[int]] = None,
        fb_channels: Optional[int] = None,
        circuit_connectivity: list[list[int]] = [
            [1, 0],
            [0, 1],
        ],
        rectify: bool | list[list[bool]] = False,
        kernel_size: tuple[int, int] | list[list[tuple[int, int]]] = (3, 3),
        bias: bool | list[list[bool]] = True,
        out_pool_kernel_size: tuple[int, int] = (3, 3),
        out_pool_stride: Optional[tuple[int, int]] = None,
        out_pool_size: Optional[tuple[int, int]] = None,
        inh_mode: str = "same",
        conv_activation: Optional[str | list[list[str]]] = None,
        post_activation: Optional[str | list[str]] = "tanh",
        tau_mode: Optional[str] = "channel",
    ):
        super().__init__()

        # Store necessary parameters
        self.in_size = in_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fb_channels = fb_channels if fb_channels is not None else 0
        self.use_fb = fb_channels > 0

        # Calculate inhibitory neuron size based on mode
        if inh_mode == "half":
            self.inh_size = (ceil(in_size[0] / 2), ceil(in_size[1] / 2))
        elif inh_mode == "same":
            self.inh_size = in_size
        else:
            raise ValueError("inh_mode must be 'half' or 'same'.")

        # Calculate output size based on pooling
        if out_pool_stride is not None:
            if out_pool_size is not None:
                raise ValueError(
                    "out_pool_stride and out_pool_size cannot both be provided."
                )
            self.out_size = (
                ceil(in_size[0] / out_pool_stride[0]),
                ceil(in_size[1] / out_pool_stride[1]),
            )
        elif out_pool_size is not None:
            self.out_size = out_pool_size
        else:
            self.out_size = in_size

        # Format excitatory neuron channels
        try:
            iter(h_exc_channels)
        except TypeError:
            self.h_exc_channels = [h_exc_channels]
        else:
            self.h_exc_channels = h_exc_channels

        # Format inhibitory neuron channels
        if h_inh_channels is not None:
            try:
                iter(h_inh_channels)
            except TypeError:
                self.h_inh_channels = [h_inh_channels]
            else:
                self.h_inh_channels = h_inh_channels
        else:
            self.h_inh_channels = []
        self.use_inh = bool(self.h_inh_channels)

        # Format circuit connectivity
        self.circuit_connectivity = torch.tensor(circuit_connectivity)
        if (
            self.circuit_connectivity.dim() != 2
            or (
                self.circuit_connectivity.shape[0]
                != len(self.h_exc_channels) + len(self.h_inh_channels) + 1
            )
            or (
                self.circuit_connectivity.shape[1]
                != len(self.h_exc_channels)
                + len(self.h_inh_channels)
                + int(self.use_fb)
                + 1
            )
        ):
            raise ValueError(
                "circuit_connectivity must be an array-like of shape (len(h_exc_channels) + len(h_inh_channels) + 1 [for output], len(h_exc_channels) + len(h_inh_channels) + 1 [for input] + int(use_fb) [for feedback])."
            )

        # Format connectivity variables
        def to_connectivity(
            var: Any, name: str, transform: Callable = lambda x: x
        ):
            try:
                var[0][0]
            except IndexError:
                var = [
                    [var] * self.circuit_connectivity.shape[1]
                    for _ in range(self.circuit_connectivity.shape[0])
                ]

            arr = np.empty(self.circuit_connectivity.shape, dtype=object)

            if len(var) != self.circuit_connectivity.shape[0]:
                raise ValueError(
                    f"{name} must be an array-like of shape (circuit_connectivity.shape[0], circuit_connectivity.shape[1])."
                )
            for i in range(self.circuit_connectivity.shape[0]):
                if len(var[i]) != self.circuit_connectivity.shape[1]:
                    raise ValueError(
                        f"{name} must be an array-like of shape (circuit_connectivity.shape[0], circuit_connectivity.shape[1])."
                    )
                for j in range(self.circuit_connectivity.shape[1]):
                    arr[i, j] = (
                        transform(var[i][j])
                        if self.circuit_connectivity[i, j]
                        else None
                    )

            return arr

        self.kernel_size = to_connectivity(kernel_size, name="kernel_size")
        self.rectify = to_connectivity(rectify, name="rectify")
        self.conv_activations = to_connectivity(
            conv_activation,
            name="conv_activation",
            transform=get_activation,
        )

        # Initialize learnable membrane time constants
        tau_exc_shape_dict = {
            "channel": (1, sum(self.h_exc_channels), 1, 1),
            "spatial": (1, 1, *self.in_size),
            "channel_spatial": (1, sum(self.h_exc_channels), *self.in_size),
            "none": (1, 1, 1, 1),
        }
        tau_inh_shape_dict = {
            "channel": (1, sum(self.h_inh_channels), 1, 1),
            "spatial": (1, 1, *self.inh_size),
            "channel_spatial": (1, sum(self.h_inh_channels), *self.inh_size),
            "none": (1, 1, 1, 1),
        }

        self.tau_mode = tau_mode if tau_mode is not None else "none"
        try:
            self.tau_exc = nn.Parameter(
                torch.randn(tau_exc_shape_dict[self.tau_mode])
            )
            if self.use_inh:
                self.tau_inh = nn.Parameter(
                    torch.randn(tau_inh_shape_dict[self.tau_mode])
                )
        except KeyError:
            raise ValueError(
                "tau_mode must be 'channel', 'spatial', 'channel_spatial', or None."
            )

        # Create convolutional layers
        # Here, we represent the circuit connectivity between neuron classes
        # as an array of convolutions (implemented as a dictionary for efficiency).
        # The convolution self.convs[f"{i}->{j}"] corresponds to the connection
        # from neuron class i to neuron class j.
        self.convs = nn.ModuleDict()
        for i in range(self.circuit_connectivity.shape[0]):
            if i == 0:
                in_type = "input"
                conv_in_channels = self.in_channels
            elif self.use_fb and i == 1:
                in_type = "feedback"
                conv_in_channels = self.fb_channels
            elif i < 1 + int(self.use_fb) + len(self.h_exc_channels):
                in_type = "excitatory"
                conv_in_channels = self.h_exc_channels[
                    i - 1 - int(self.use_fb)
                ]
            else:
                in_type = "inhibitory"
                conv_in_channels = self.h_inh_channels[
                    i - 1 - int(self.use_fb) - len(self.h_exc_channels)
                ]

            for j in range(self.circuit_connectivity.shape[1]):
                if self.circuit_connectivity[i, j] == 1:
                    if self.rectify[i, j]:
                        if in_type == "input":
                            warnings.warn(
                                "Rectification of input neurons may hinder learning."
                            )
                        Conv2d = Conv2dRectify
                    else:
                        Conv2d = nn.Conv2d
                    conv = nn.Sequential()
                    stride = 1
                    if j < len(self.h_exc_channels):
                        # excitatory
                        conv_out_channels = self.h_exc_channels[j]
                        if in_type == "inhibitory" and self.inh_mode == "half":
                            # inh -> exc
                            do_upsample = True
                    elif j < len(self.h_exc_channels) + len(
                        self.h_inh_channels
                    ):
                        # inhibitory neuron
                        conv_out_channels = self.h_inh_channels[
                            j - len(self.h_exc_channels)
                        ]
                        if in_type == "inhibitory":
                            # inh -> inh
                            do_upsample = False
                        elif self.inh_mode == "half":
                            # exc/input/fb -> inh
                            stride = 2

                    else:
                        conv_out_channels = self.out_channels
                        if in_type == "inhibitory" and self.inh_mode == "half":
                            # inh -> out
                            do_upsample = True

                    if do_upsample:
                        conv.append(
                            nn.Upsample(size=self.in_size, mode="bilinear")
                        )
                    conv.append(
                        Conv2d(
                            in_channels=conv_in_channels,
                            out_channels=conv_out_channels,
                            kernel_size=self.kernel_size[i, j],
                            stride=stride,
                            padding=(
                                self.kernel_size[i, j, 0] // 2,
                                self.kernel_size[i, j, 1] // 2,
                            ),
                            bias=bias,
                        )
                    )
                    self.convs[f"{i}->{j}"] = conv

        # Initialize output pooling layer
        if out_pool_stride is not None:
            self.out_pool = nn.AvgPool2d(
                kernel_size=out_pool_kernel_size,
                stride=out_pool_stride,
                padding=(
                    out_pool_kernel_size[0] // 2,
                    out_pool_kernel_size[1] // 2,
                ),
            )
        elif out_pool_size is not None:
            self.out_pool = nn.AdaptiveAvgPool2d(out_pool_size)
        else:
            self.out_pool = nn.Identity()

    def init_hidden(
        self,
        batch_size: int,
        init_mode: str = "zeros",
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Initializes the hidden state of the cell.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros' or 'normal'. Defaults to 'zeros'.
            device (torch.device, optional): Device to allocate the hidden state on. Defaults to None.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the initialized excitatory cell hidden state and, if inhibitory neurons are used, the initialized inhibitory neuron cell hidden state.
        """

        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "ones":
            func = torch.ones
        elif init_mode == "randn":
            func = torch.randn
        else:
            raise ValueError(
                "Invalid init_mode. Must be 'zeros', 'ones', or 'randn'."
            )

        return (
            func(
                batch_size, self.h_exc_channels, *self.in_size, device=device
            ),
            func(
                batch_size,
                sum(self.h_inh_channels),
                *self.inh_size,
                device=device,
            )
            if self.use_inh
            else None,
        )

    def init_out(
        self,
        batch_size: int,
        init_mode: str = "zeros",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Initializes the output.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros' or 'randn'. Defaults to 'zeros'.
            device (torch.device, optional): Device to allocate the output on. Defaults to None.

        Returns:
            torch.Tensor: The initialized output.
        """

        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "ones":
            func = torch.ones
        elif init_mode == "randn":
            func = torch.randn
        else:
            raise ValueError(
                "Invalid init_mode. Must be 'zeros', 'ones', or 'randn'."
            )

        return func(
            batch_size, self.out_channels, *self.out_size, device=device
        )

    def init_fb(
        self,
        batch_size: int,
        init_mode: str = "zeros",
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Initializes the feedback input.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros' or 'normal'. Defaults to 'zeros'.
            device (torch.device, optional): Device to allocate the feedback input on. Defaults to None.

        Returns:
            Optional[torch.Tensor]: The initialized feedback input if `use_fb` is True, otherwise None.
        """

        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "ones":
            func = torch.ones
        elif init_mode == "randn":
            func = torch.randn
        else:
            raise ValueError(
                "Invalid init_mode. Must be 'zeros', 'ones', or 'randn'."
            )

        return (
            func(batch_size, self.fb_channels, *self.in_size, device=device)
            if self.use_fb
            else None
        )

    def forward(
        self,
        input: torch.Tensor,
        h_exc: torch.Tensor,
        h_inh: Optional[torch.Tensor] = None,
        fb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the cell.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_size[0], in_size[1]).
            h_exc (torch.Tensor): Excitatory cell hidden state of shape (batch_size, h_exc_channels, in_size[0], in_size[1]).
            h_inh (Optional[torch.Tensor]): Inhibitory neuron cell hidden state of shape (batch_size, h_inh_channels_sum, inh_size[0], inh_size[1]).
            fb (Optional[torch.Tensor]): Feedback input of shape (batch_size, fb_channels, in_size[0], in_size[1]).

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing the output, new excitatory cell hidden state, and new inhibitory neuron cell hidden state.
        """

        if self.use_fb and fb is None:
            raise ValueError("If use_fb is True, fb must be provided.")

        excs = [0] * len(self.h_exc_channels)
        inhs = [0] * len(self.h_inh_channels)
        out = 0
        for i in range(self.circuit_connectivity.shape[0]):
            if i == 0:
                in_sign = 1
                in_data = input
            elif self.use_fb and i == 1:
                in_sign = 1
                in_data = fb
            elif i < 1 + int(self.use_fb) + len(self.h_exc_channels):
                in_sign = 1
                in_data = h_exc[i - 1 - int(self.use_fb)]
            else:
                in_sign = -1
                in_data = h_inh[
                    i - 1 - int(self.use_fb) - len(self.h_exc_channels)
                ]

            for j in range(self.circuit_connectivity.shape[1]):
                if self.circuit_connectivity[i, j] == 1:
                    if j < len(self.h_exc_channels):
                        # excitatory
                        excs[j] += in_sign * self.convs[f"{i}->{j}"](in_data)
                    elif j < len(self.h_exc_channels) + len(
                        self.h_inh_channels
                    ):
                        # inhibitory
                        inhs[j - len(self.h_exc_channels)] += (
                            in_sign * self.convs[f"{i}->{j}"](in_data)
                        )
                    else:
                        # output
                        out += in_sign * self.convs[f"{i}->{j}"](in_data)

        # Compute excitations for excitatory cells
        exc_cat = torch.cat(
            [h_exc, input, fb] if self.use_fb else [h_exc, input], dim=1
        )
        if self.use_three_compartments:
            exc_exc_soma = self.pre_inh_activation(self.conv_exc_exc(h_exc))
            exc_exc_basal = self.pre_inh_activation(
                self.conv_exc_exc_input(input)
            )
            if self.use_fb:
                exc_exc_apical = self.pre_inh_activation(
                    self.conv_exc_exc_fb(fb)
                )
        else:
            exc_exc_soma = self.pre_inh_activation(self.conv_exc_exc(exc_cat))

        inhs = [0] * 4
        if self.h_inh_channels:
            # Compute excitations for inhibitory neurons
            if len(self.h_inh_channels) == 4:
                exc_inh = self.pre_inh_activation(self.conv_exc_inh(h_exc))
                exc_input_inh = self.pre_inh_activation(
                    self.conv_exc_input_inh(input)
                )
                exc_fb_inh = self.pre_inh_activation(self.conv_exc_inh_fb(fb))
            else:
                exc_inh = self.pre_inh_activation(self.conv_exc_inh(exc_cat))

            # Compute inhibitions for all neurons
            h_inhs = h_inh
            if self.immediate_inhibition:
                if h_inh is not None:
                    raise ValueError(
                        "If h_inh_channels is provided and immediate_inhibition is True, h_inh must not be provided."
                    )
                h_inhs = exc_inh
            elif h_inh is None:
                raise ValueError(
                    "If h_inh_channels is provided and immediate_inhibition is False, h_inh must be provided."
                )

            h_inhs = torch.split(
                h_inhs,
                self.h_inh_channels,
                dim=1,
            )
            inh_inh_2 = inh_inh_3 = 0
            for i in range(len(self.h_inh_channels)):
                conv = self.convs_inh[i]
                if i in (2, 3):
                    conv, conv2 = conv.conv1, conv.conv2
                    if i == 2:
                        inh_inh_3 = self.pre_inh_activation(conv2(h_inhs[i]))
                    else:
                        inh_inh_2 = self.pre_inh_activation(conv2(h_inhs[i]))
                inhs[i] = self.pre_inh_activation(conv(h_inhs[i]))
        inh_exc_soma, inh_inh, inh_exc_basal, inh_exc_apical = inhs

        # Compute new excitatory cell hidden state
        exc_soma = self.post_inh_activation(exc_exc_soma - inh_exc_soma)
        if self.use_three_compartments:
            exc_basal = self.post_inh_activation(exc_exc_basal - inh_exc_basal)
            exc_apical = (
                self.post_inh_activation(exc_exc_apical - inh_exc_apical)
                if self.use_fb
                else 0
            )
            exc_soma = exc_soma + exc_apical + exc_basal
            exc_soma /= 3
        h_exc_new = self.post_integration_activation(exc_soma)

        # Compute Euler update for excitatory cell hidden state
        tau_exc = torch.sigmoid(self.tau_exc)
        h_exc = tau_exc * h_exc_new + (1 - tau_exc) * h_exc

        # Compute new inhibitory neuron cell hidden state
        if self.use_inh:
            h_inh_new = exc_inh - inh_inh
            if len(self.h_inh_channels) == 4:
                # Add excitations and inhibitions to inhibitory neuron 2
                start = sum(self.h_inh_channels[:2])
                end = start + self.h_inh_channels[2]
                h_inh_new[:, start:end, ...] = (
                    h_inh_new[:, start:end, ...] + exc_input_inh - inh_inh_2
                )
                # Add excitations and inhibitions to inhibitory neuron 3
                start = sum(self.h_inh_channels[:3])
                h_inh_new[:, start:, ...] = (
                    h_inh_new[:, start:, ...] + exc_fb_inh - inh_inh_3
                )
            h_inh_new = self.post_inh_activation(h_inh_new)
            # Compute Euler update for inhibitory neuron cell hidden state
            tau_inh = torch.sigmoid(self.tau_inh)
            h_inh = tau_inh * h_inh_new + (1 - tau_inh) * h_inh

        # Pool the output
        out = self.out_pool(h_exc)

        return out, h_exc, h_inh


class Conv2dEIRNN(nn.Module):
    """
    Implements a deep convolutional recurrent neural network with excitatory-inhibitory
    neurons (EIRNN).

    Args:
        in_size (tuple[int, int]): Size of the input data (height, width).
        in_channels (int): Number of input channels.
        h_exc_channels (int | list[int]): Number of channels in the excitatory cell \
            hidden state for each layer.
        h_inh_channels (list[int] | list[list[int]]): Number of channels in the
            inhibitory neuron hidden state for each layer.
        fb_channels (int | list[int]): Number of channels for the feedback input for \
            each layer.
        exc_kernel_size (tuple[int, int] | tuple[tuple[int, int]]): Kernel size for
            excitatory convolutions in each layer.
        inh_kernel_size (tuple[int, int] | tuple[tuple[int, int]]): Kernel size for
            inhibitory convolutions in each layer.
        fb_kernel_size (tuple[int, int] | tuple[tuple[int, int]]): Kernel size for
            feedback convolutions in each layer.
        use_three_compartments (bool): Whether to use a three-compartment model for
            excitatory cells.
        immediate_inhibition (bool): Whether inhibitory neurons provide immediate inhibition.
        num_layers (int): Number of layers in the RNN.
        inh_mode (str): Mode for handling inhibitory neuron size relative to input size
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
            inhibition (to excitatory cell).
        post_integration_activation (Optional[str]): Activation function applied \
            after integration (to excitatory and inhibitory neurons).
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

    def __init__(
        self,
        in_size: tuple[int, int],
        in_channels: int,
        h_exc_channels: int | tuple[int],
        h_inh_channels: tuple[int] | tuple[tuple[int]] = None,
        fb_channels: Optional[int | tuple[int]] = None,
        num_layers: int = 1,
        inh_mode: str = "same",
        use_three_compartments: bool = False,
        immediate_inhibition: bool = False,
        exc_rectify: bool = False,
        inh_rectify: bool = False,
        exc_kernel_size: tuple[int, int] | tuple[tuple[int, int]] = (3, 3),
        inh_kernel_size: tuple[int, int] | tuple[tuple[int, int]] = (3, 3),
        fb_kernel_size: tuple[int, int] | tuple[tuple[int, int]] = (3, 3),
        pool_kernel_size: Optional[
            tuple[int, int] | tuple[tuple[int, int]]
        ] = None,
        pool_stride: Optional[tuple[int, int] | tuple[tuple[int, int]]] = None,
        pool_global: bool | tuple[bool] = True,
        pre_inh_activation: Optional[str] = None,
        post_inh_activation: Optional[str] = "tanh",
        post_integration_activation: Optional[str] = None,
        tau_mode: Optional[str] = "channel",
        fb_activation: Optional[str] = None,
        bias: bool | tuple[bool] = True,
        layer_time_delay: bool = False,
        fb_adjacency: Optional[tuple[tuple[int | bool]] | torch.Tensor] = None,
        hidden_init_mode: str = "zeros",
        fb_init_mode: str = "zeros",
        out_init_mode: str = "zeros",
        batch_first: bool = True,
    ):
        super().__init__()

        # Expand the layer specific parameters as necessary
        self.num_layers = num_layers
        self.h_exc_channels = expand_list(h_exc_channels, self.num_layers)
        self.h_inh_channels = expand_list(
            h_inh_channels,
            self.num_layers,
            depth=0 if h_inh_channels is None else 1,
        )
        self.fb_channels = expand_list(fb_channels, self.num_layers)
        self.exc_kernel_sizes = expand_list(
            exc_kernel_size, self.num_layers, depth=1
        )
        self.inh_kernel_sizes = expand_list(
            inh_kernel_size, self.num_layers, depth=1
        )
        self.fb_kernel_sizes = expand_list(
            fb_kernel_size, self.num_layers, depth=1
        )
        self.pool_kernel_sizes = expand_list(
            pool_kernel_size, self.num_layers, depth=1
        )
        self.pool_strides = expand_list(pool_stride, self.num_layers, depth=1)
        self.pool_globals = expand_list(pool_global, self.num_layers)
        self.biases = expand_list(bias, self.num_layers)

        # Save layer agnostic parameters
        self.use_three_compartments = use_three_compartments
        self.immediate_inhibition = immediate_inhibition
        self.layer_time_delay = layer_time_delay
        self.batch_first = batch_first
        self.hidden_init_mode = hidden_init_mode
        self.fb_init_mode = fb_init_mode
        self.out_init_mode = out_init_mode
        self.fb_activation = get_activation(fb_activation)

        # Calculate input sizes for each layer
        self.in_sizes = [in_size]
        for i in range(self.num_layers - 1):
            self.in_sizes.append(
                (
                    ceil(self.in_sizes[i][0] / self.pool_strides[i][0]),
                    ceil(self.in_sizes[i][1] / self.pool_strides[i][1]),
                )
            )

        # Initialize feedback connections
        self.receives_fb = [False] * self.num_layers
        self.fb_adjacency = [[]] * self.num_layers
        if fb_adjacency is not None:
            # Check if fb_adjacency and fb_channels are consistent
            if not any(self.fb_channels):
                raise ValueError(
                    "fb_adjacency must be provided if and only if fb_channels is "
                    "provided for at least one layer."
                )

            # Load or create fb_adjacency tensor
            if not isinstance(fb_adjacency, torch.Tensor):
                fb_adjacency = torch.tensor(fb_adjacency)

            # Validate fb_adjacency tensor
            if (
                fb_adjacency.dim() != 2
                or fb_adjacency.shape[0] != self.num_layers
                or fb_adjacency.shape[1] != self.num_layers
            ):
                raise ValueError(
                    "The the dimensions of fb_adjacency must match number of layers."
                )
            if fb_adjacency.count_nonzero() == 0:
                raise ValueError(
                    "fb_adjacency must be a non-zero tensor if provided."
                )

            # Create feedback convolutions
            if exc_rectify:
                Conv2dFb = Conv2dRectify
            else:
                Conv2dFb = nn.Conv2d
            self.fb_convs = nn.ModuleDict()
            for i, row in enumerate(fb_adjacency):
                row = row.nonzero().squeeze(1).tolist()
                self.fb_adjacency[i] = row
                for j in row:
                    self.receives_fb[j] = True
                    self.fb_convs[(i, j)] = nn.Sequential(
                        nn.Upsample(size=self.in_sizes[j], mode="bilinear"),
                        Conv2dFb(
                            in_channels=self.h_exc_channels[i],
                            out_channels=self.fb_channels[j],
                            kernel_size=self.fb_kernel_sizes[j],
                            padding=(
                                self.fb_kernel_sizes[j][0] // 2,
                                self.fb_kernel_sizes[j][1] // 2,
                            ),
                            bias=self.biases[j],
                        ),
                        self.fb_activation,
                    )
        elif any(self.fb_channels):
            raise ValueError(
                "fb_adjacency must be provided if and only if fb_channels is provided "
                "for at least one layer."
            )

        # Create layers and perturbation modules
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(
                Conv2dEIRNNCell(
                    in_size=self.in_sizes[i],
                    in_channels=(
                        in_channels if i == 0 else self.h_exc_channels[i - 1]
                    ),
                    h_exc_channels=self.h_exc_channels[i],
                    h_inh_channels=self.h_inh_channels[i],
                    fb_channels=self.fb_channels[i]
                    if self.receives_fb[i]
                    else 0,
                    inh_mode=inh_mode,
                    exc_kernel_size=self.exc_kernel_sizes[i],
                    inh_kernel_size=self.inh_kernel_sizes[i],
                    use_three_compartments=self.use_three_compartments,
                    immediate_inhibition=self.immediate_inhibition,
                    exc_rectify=exc_rectify,
                    inh_rectify=inh_rectify,
                    pool_kernel_size=self.pool_kernel_sizes[i],
                    pool_stride=self.pool_strides[i],
                    pool_global=self.pool_globals[i],
                    bias=self.biases[i],
                    pre_inh_activation=pre_inh_activation,
                    post_inh_activation=post_inh_activation,
                    post_integration_activation=post_integration_activation,
                    tau_mode=tau_mode,
                )
            )

    def _init_hidden(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the hidden states for all layers.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode ('zeros' or 'normal').
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            tuple(list[torch.Tensor], list[torch.Tensor]): A tuple of lists containing \
                the initialized excitatory and inhibitory neuron hidden states for each layer.
        """
        h_excs = []
        h_inhs = []
        for layer in self.layers:
            h_exc, h_inh = layer.init_hidden(
                batch_size, init_mode=init_mode, device=device
            )
            h_excs.append(h_exc)
            h_inhs.append(h_inh)
        return h_excs, h_inhs

    def _init_fb(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the feedback inputs for all layers.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode ('zeros' or 'normal').
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list of initialized feedback inputs for each layer.
        """
        fbs = []
        for layer in self.layers:
            fb = layer.init_fb(batch_size, init_mode=init_mode, device=device)
            fbs.append(fb)
        return fbs

    def _init_out(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the outputs for all layers.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode ('zeros' or 'normal').
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list of initialized outputs for each layer.
        """
        outs = []
        for layer in self.layers:
            out = layer.init_out(
                batch_size, init_mode=init_mode, device=device
            )
            outs.append(out)
        return outs

    def _init_state(
        self,
        out_0,
        h_exc_0,
        h_inh_0,
        fb_0,
        num_steps,
        batch_size,
        device=None,
    ):
        """
        Initializes the inhnal state of the network.

        Args:
            out_0 (Optional[list[torch.Tensor]]): Initial outputs for each layer.
            h_exc_0 (Optional[list[torch.Tensor]]): Initial excitatory cell hidden states for each layer.
            h_inh_0 (Optional[list[torch.Tensor]]): Initial inhibitory neuron cell hidden states for each layer.
            fb_0 (Optional[list[torch.Tensor]]): Initial feedback inputs for each layer.
            num_steps (int): Number of time steps.
            batch_size (int): Batch size.
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            tuple(list[list[torch.Tensor]], list[list[torch.Tensor]], list[list[torch.Tensor]], list[list[torch.Tensor]]): A tuple containing the initialized outputs, excitatory cell hidden states, inhibitory neuron cell hidden states, and feedback inputs for each layer and time step.
        """
        outs = [[None] * num_steps for _ in range(self.num_layers)]
        h_excs = [[None] * num_steps for _ in range(self.num_layers)]
        h_inhs = [[None] * num_steps for _ in range(self.num_layers)]
        fbs = [
            [0 if self.receives_fb[i] else None] * num_steps
            for i in range(self.num_layers)
        ]

        if (
            out_0 is not None
            and self.num_layers > 1
            and len(out_0) != self.num_layers
        ):
            raise ValueError(
                "The length of out_0 must be equal to the number of layers."
            )

        if (
            h_exc_0 is not None
            and self.num_layers > 1
            and len(h_exc_0) != self.num_layers
        ):
            raise ValueError(
                "The length of h_exc_0 must be equal to the number of layers."
            )

        if (
            h_inh_0 is not None
            and self.num_layers > 1
            and len(h_inh_0) != self.num_layers
        ):
            raise ValueError(
                "The length of h_inh_0 must be equal to the number of layers."
            )

        if (
            fb_0 is not None
            and self.num_layers > 1
            and len(fb_0) != self.num_layers
        ):
            raise ValueError(
                "The length of fb_0 must be equal to the number of layers."
            )

        if not isinstance(out_0, (list, tuple)):
            out_0 = [out_0] * self.num_layers
        if not isinstance(h_exc_0, (list, tuple)):
            h_exc_0 = [h_exc_0] * self.num_layers
        if not isinstance(h_inh_0, (list, tuple)):
            h_inh_0 = [h_inh_0] * self.num_layers
        if not isinstance(fb_0, (list, tuple)):
            fb_0 = [fb_0] * self.num_layers

        if any(x is None for x in out_0):
            out_tmp = self._init_out(
                batch_size, init_mode=self.out_init_mode, device=device
            )
        if any(x is None for x in h_exc_0) or any(x is None for x in h_inh_0):
            h_exc_tmp, h_inh_tmp = self._init_hidden(
                batch_size, init_mode=self.hidden_init_mode, device=device
            )
        if any(x is None for x in fb_0):
            fb_tmp = self._init_fb(
                batch_size, init_mode=self.fb_init_mode, device=device
            )

        for i in range(self.num_layers):
            if out_0[i] is None:
                outs[i][-1] = out_tmp[i]
            if h_exc_0[i] is None:
                h_excs[i][-1] = h_exc_tmp[i]
            if h_inh_0[i] is None:
                h_inhs[i][-1] = h_inh_tmp[i]
            if fb_0[i] is None:
                fbs[i][-1] = fb_tmp[i]

        return outs, h_excs, h_inhs, fbs

    def _format_x(self, x, num_steps):
        """
        Formats the input tensor to match the expected shape.

        Args:
            x (torch.Tensor): Input tensor.
            num_steps (Optional[int]): Number of time steps.

        Returns:
            tuple(torch.Tensor, int): The formatted input tensor and the number of time steps.
        """
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
    def _modulation_identity(x, *args, **kwargs):
        return x

    def _format_modulation_fns(
        self, modulation_out_fn, modulation_exc_fn, modulation_inh_fn
    ):
        """
        Formats the modulation functions.

        Args:
            modulation_out_fn (Optional[torch.Tensor]): Modulation function for outputs.
            modulation_exc_fn (Optional[torch.Tensor]): Modulation function for excitatory cell hidden states.
            modulation_inh_fn (Optional[torch.Tensor]): Modulation function for inhibitory neuron cell hidden states.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): The formatted modulation functions.
        """
        modulation_fns = [
            modulation_out_fn,
            modulation_exc_fn,
            modulation_inh_fn,
        ]
        for i, fn in enumerate(modulation_fns):
            if fn is None:
                modulation_fns[i] = self._modulation_identity
            elif not callable(fn):
                raise ValueError("modulation_fns must be callable or None.")

        return modulation_fns

    def _format_result(self, outs, h_excs, h_inhs, fbs):
        """
        Formats the outputs, hidden states, and feedback inputs.

        Args:
            outs (list[list[torch.Tensor]]): Outputs for each layer and time step.
            h_excs (list[list[torch.Tensor]]): Excitatory cell hidden states for each layer and time step.
            h_inhs (list[list[torch.Tensor]]): Inhibitory neuron cell hidden states for each layer and time step.
            fbs (list[list[torch.Tensor]]): Feedback inputs for each layer and time step.

        Returns:
            tuple(list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]): The formatted outputs, excitatory cell hidden states, inhibitory neuron cell hidden states, and feedback inputs.
        """
        for i in range(self.num_layers):
            outs[i] = torch.stack(outs[i])
            h_excs[i] = torch.stack(h_excs[i])
            if self.h_inh_channels[i] and not self.immediate_inhibition:
                h_inhs[i] = torch.stack(
                    h_inhs[i]
                )  # TODO: Check if this is correct
            else:
                assert not self.h_inh_channels[i] or self.immediate_inhibition
                assert all(x is None for x in h_inhs[i])
                h_inhs[i] = None
            if self.receives_fb[i]:
                fbs[i] = torch.stack(fbs[i])
            else:
                assert all(x is None for x in fbs[i])
                fbs[i] = None
            if self.batch_first:
                outs[i] = outs[i].transpose(0, 1)
                h_excs[i] = h_excs[i].transpose(0, 1)
                if self.h_inh_channels[i] and not self.immediate_inhibition:
                    h_inhs[i] = h_inhs[i].transpose(0, 1)
                if self.receives_fb[i]:
                    fbs[i] = fbs[i].transpose(0, 1)
        if all(x is None for x in h_inhs):
            h_inhs = None
        if all(x is None for x in fbs):
            fbs = None

        return outs, h_excs, h_inhs, fbs

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        out_0: Optional[list[torch.Tensor]] = None,
        h_exc_0: Optional[list[torch.Tensor]] = None,
        h_inh_0: Optional[list[torch.Tensor]] = None,
        fb_0: Optional[list[torch.Tensor]] = None,
        modulation_out_fn: Optional[
            Callable[[torch.Tensor, int, int], torch.Tensor]
        ] = None,
        modulation_exc_fn: Optional[
            Callable[[torch.Tensor, int, int], torch.Tensor]
        ] = None,
        modulation_inh_fn: Optional[
            Callable[[torch.Tensor, int, int], torch.Tensor]
        ] = None,
    ):
        """
        Performs forward pass of the Conv2dEIRNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_size[0], in_size[1]).
            num_steps (Optional[int]): Number of time steps.
            out_0 (Optional[list[torch.Tensor]]): Initial outputs for each layer.
            h_exc_0 (Optional[list[torch.Tensor]]): Initial excitatory cell hidden states for each layer.
            h_inh_0 (Optional[list[torch.Tensor]]): Initial inhibitory neuron cell hidden states for each layer.
            fb_0 (Optional[list[torch.Tensor]]): Initial feedback inputs for each layer.
            modulation_out_fn (Optional[Callable[[torch.Tensor, int, int], torch.Tensor]]): Modulation function for outputs.
            modulation_exc_fn (Optional[Callable[[torch.Tensor, int, int], torch.Tensor]]): Modulation function for excitatory cell hidden states.
            modulation_inh_fn (Optional[Callable[[torch.Tensor, int, int], torch.Tensor]]): Modulation function for inhibitory neuron cell hidden states.
            return_all_layers_out (bool): Whether to return outputs for all layers.

        Returns:
            tuple(torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]): The output tensor, excitatory cell hidden states, inhibitory neuron cell hidden states, and feedback inputs.
        """

        device = x.device

        x, num_steps = self._format_x(x, num_steps)

        batch_size = x.shape[1]

        outs, h_excs, h_inhs, fbs = self._init_state(
            out_0,
            h_exc_0,
            h_inh_0,
            fb_0,
            num_steps,
            batch_size,
            device=device,
        )

        modulation_out_fn, modulation_exc_fn, modulation_inh_fn = (
            self._format_modulation_fns(
                modulation_out_fn, modulation_exc_fn, modulation_inh_fn
            )
        )

        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                # Apply additive modulation
                outs[i][t] = modulation_out_fn(outs[i][t], i, t)
                h_excs[i][t] = modulation_exc_fn(h_excs[i][t], i, t)
                h_inhs[i][t] = modulation_inh_fn(h_inhs[i][t], i, t)

                # Compute layer update and output
                outs[i][t], h_excs[i][t], h_inhs[i][t] = layer(
                    input=(
                        x[t]
                        if i == 0
                        else (
                            outs[i - 1][t - 1]
                            if self.layer_time_delay
                            else outs[i - 1][t]
                        )
                    ),
                    h_exc=h_excs[i][t - 1],
                    h_inh=h_inhs[i][t - 1],
                    fb=fbs[i][t - 1],
                )

                # Apply feedback
                for j in self.fb_adjacency[i]:
                    fbs[j][t] = fbs[j][t] + self.fb_convs[f"fb_conv_{i}_{j}"](
                        outs[i][t]
                    )

        outs, h_excs, h_inhs, fbs = self._format_result(
            outs, h_excs, h_inhs, fbs
        )

        return outs, h_excs, h_inhs, fbs
