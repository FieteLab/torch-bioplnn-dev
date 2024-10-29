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
    Implements a 2D convolutional recurrent neural network cell with excitatory \
    and inhibitory neurons.

    Args:
        in_size (tuple[int, int]): Size of the input data (height, width).
        in_channels (int): Number of input channels.
        h_pyr_channels (int, optional): Number of channels in the pyramidal cell \
            hidden state. Defaults to 16.
        h_inter_channels (list[int], optional): List of number of channels for \
            each interneuron type. Defaults to [16]. The length can be 0, 1, 2, \
            or 4, indicating the number of interneuron types. A length of 0 \
            indicates no interneurons.
        fb_channels (int, optional): Number of channels for the feedback input. \
            Defaults to 0.
        inter_mode (str, optional): Mode for handling interneuron size relative \
            to input size. Must be 'half' or 'same'. Defaults to 'half'.
        use_three_compartments (bool, optional): Whether to use a three- \
            compartment model for pyramidal cells. Defaults to False.
        immediate_inhibition (bool, optional): Whether interneurons provide \
            immediate inhibition. Defaults to False. If True, only one \
            interneuron type is allowed, and the hidden state for the \
            interneurons is not used/updated.
        exc_rectify (bool, optional): Whether to positively rectify excitatory \
            weights. Defaults to False.
        inh_rectify (bool, optional): Whether to positively rectify inhibitory \
            weights. Defaults to False.
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
        h_pyr_channels: int = 16,
        h_inter_channels: Optional[list[int]] = [16],
        fb_channels: Optional[int] = None,
        inter_mode: str = "half",
        use_three_compartments: bool = False,
        immediate_inhibition: bool = False,
        exc_rectify: bool = False,
        inh_rectify: bool = False,
        exc_kernel_size: tuple[int, int] = (3, 3),
        inh_kernel_size: tuple[int, int] = (3, 3),
        pool_kernel_size: Optional[tuple[int, int]] = (3, 3),
        pool_stride: Optional[tuple[int, int]] = (2, 2),
        pool_global: bool = False,
        pre_inh_activation: Optional[str | list[str]] = "relu",
        post_inh_activation: Optional[str | list[str]] = "tanh",
        post_integration_activation: Optional[str | list[str]] = None,
        tau_mode: Optional[str] = "channel",
        bias: bool = True,
    ):
        super().__init__()

        self.in_size = in_size
        # Calculate interneuron size based on mode
        if inter_mode == "half":
            self.inter_size = (ceil(in_size[0] / 2), ceil(in_size[1] / 2))
        elif inter_mode == "same":
            self.inter_size = in_size
        else:
            raise ValueError("inter_mode must be 'half' or 'same'.")

        # Calculate output size based on pooling
        if pool_kernel_size is None != pool_stride is None:
            raise ValueError(
                "pool_kernel_size and pool_stride must both be None or both be provided."
            )
        if pool_kernel_size is not None and pool_global:
            raise ValueError(
                "If pool_global is provided, pool_kernel_size and pool_stride must be None, and vice versa."
            )
        if pool_stride is None and not pool_global:
            self.out_size = self.in_size
        elif pool_global:
            self.out_size = (1, 1)
        else:
            self.out_size = (
                ceil(in_size[0] / pool_stride[0]),
                ceil(in_size[1] / pool_stride[1]),
            )

        # Store necessary parameters
        self.in_channels = in_channels
        self.h_pyr_channels = h_pyr_channels
        self.out_channels = h_pyr_channels
        self.h_inter_channels = h_inter_channels if h_inter_channels is not None else []
        self.use_h_inter = self.h_inter_channels and not immediate_inhibition
        self.fb_channels = fb_channels if fb_channels is not None else 0
        self.use_fb = fb_channels > 0
        self.immediate_inhibition = immediate_inhibition
        self.use_three_compartments = use_three_compartments
        self.pool_stride = pool_stride
        # Create activation functions
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

        # Validate h_inter_channels and use_three_compartments
        if len(self.h_inter_channels) not in (0, 1, 2, 4):
            raise ValueError(
                "h_inter_channels must be None or a list/tuple of length 0, 1, 2, or 4."
                f"Got length {len(self.h_inter_channels)}."
            )
        if len(self.h_inter_channels) == 4:
            if not self.use_fb:
                warnings.warn(
                    "If h_inter_channels has length 4, fb_channels must be greater \
                        than 0. Disabling interneuron types 2 and 3."
                )
                self.h_inter_channels = self.h_inter_channels[:2]
            if not self.use_three_compartments:
                raise ValueError(
                    "If h_inter_channels has length 4, use_three_compartments must be True."
                )

        # Calculate sum of interneuron channels
        self.h_inter_channels_sum = sum(self.h_inter_channels)

        # Validate h_inter_channels and immediate_inhibition
        if self.immediate_inhibition and len(self.h_inter_channels) not in (0, 1):
            raise ValueError(
                "If immediate_inhibition is True, h_inter_channels must be None or a \
                    list/tuple of length 0 or 1."
            )

        # Initialize learnable membrane time constants
        if tau_mode == "channel":
            self.tau_pyr = nn.Parameter(torch.randn((1, self.h_pyr_channels, 1, 1)))
            if self.use_h_inter:
                self.tau_inter = nn.Parameter(
                    torch.randn((1, self.h_inter_channels_sum, 1, 1))
                )
        elif tau_mode == "spatial":
            self.tau_pyr = nn.Parameter(torch.randn((1, 1, *self.in_size)))
            if self.use_h_inter:
                self.tau_inter = nn.Parameter(torch.randn((1, 1, *self.inter_size)))
        elif tau_mode == "channel_spatial":
            self.tau_pyr = nn.Parameter(
                torch.randn((1, self.h_pyr_channels, *self.in_size))
            )
            if self.use_h_inter:
                self.tau_inter = nn.Parameter(
                    torch.randn((1, self.h_inter_channels_sum, *self.inter_size))
                )
        elif tau_mode is None:
            self.tau_pyr = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=False)
            if self.use_h_inter:
                self.tau_inter = nn.Parameter(
                    torch.ones(1, 1, 1, 1), requires_grad=False
                )
        else:
            raise ValueError(
                "tau_mode must be None, 'channel', 'spatial', or 'channel_spatial'."
            )

        # Create pyramidal convolutional layers
        if exc_rectify:
            Conv2dExc = Conv2dPositive
        else:
            Conv2dExc = nn.Conv2d

        if self.use_three_compartments:
            conv_exc_channels = self.h_pyr_channels
            self.conv_exc_pyr_input = Conv2dExc(
                in_channels=self.in_channels,
                out_channels=self.h_pyr_channels,
                kernel_size=exc_kernel_size,
                padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                bias=bias,
            )
            if self.use_fb:
                self.conv_exc_pyr_fb = Conv2dExc(
                    in_channels=self.fb_channels,
                    out_channels=self.h_pyr_channels,
                    kernel_size=exc_kernel_size,
                    padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                    bias=bias,
                )
        else:
            conv_exc_channels = (
                self.h_pyr_channels + self.in_channels + self.fb_channels
            )

        self.conv_exc_pyr = Conv2dExc(
            in_channels=conv_exc_channels,
            out_channels=self.h_pyr_channels,
            kernel_size=exc_kernel_size,
            padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
            bias=bias,
        )

        # Create interneuron convolutional layers
        if self.h_inter_channels:
            if len(self.h_inter_channels) == 4:
                self.conv_exc_input_inter = Conv2dExc(
                    in_channels=self.in_channels,
                    out_channels=self.h_inter_channels[2],
                    kernel_size=exc_kernel_size,
                    stride=2 if inter_mode == "half" else 1,
                    padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                    bias=bias,
                )
                self.conv_exc_inter_fb = Conv2dExc(
                    in_channels=self.fb_channels,
                    out_channels=self.h_inter_channels[3],
                    kernel_size=exc_kernel_size,
                    stride=2 if inter_mode == "half" else 1,
                    padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                    bias=bias,
                )

            self.conv_exc_inter = Conv2dExc(
                in_channels=conv_exc_channels,
                out_channels=self.h_inter_channels_sum,
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
            self.inh_out_channels = [
                self.h_pyr_channels,
                self.h_inter_channels_sum,
                self.h_pyr_channels,
                self.h_pyr_channels,
            ]
            for i, channels in enumerate(self.h_inter_channels):
                conv = Conv2dInh(
                    in_channels=channels,
                    out_channels=self.inh_out_channels[i],
                    kernel_size=inh_kernel_size,
                    padding=(inh_kernel_size[0] // 2, inh_kernel_size[1] // 2),
                    bias=bias,
                )
                if inter_mode == "half" and i != 1:
                    conv = nn.Sequential(
                        nn.Upsample(size=self.in_size, mode="bilinear"), conv
                    )
                if i in (2, 3):
                    conv2 = Conv2dInh(
                        in_channels=channels,
                        out_channels=(self.h_inter_channels[3 if i == 2 else 2]),
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
        if pool_stride is None and not pool_global:
            self.out_pool = nn.Identity()
        elif pool_global:
            self.out_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.out_pool = nn.AvgPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=(pool_kernel_size[0] // 2, pool_kernel_size[1] // 2),
            )

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
            tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the initialized pyramidal cell hidden state and, if interneurons are used, the initialized interneuron cell hidden state.
        """

        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")

        return (
            func(batch_size, self.h_pyr_channels, *self.in_size, device=device),
            func(batch_size, self.h_inter_channels_sum, *self.inter_size, device=device)
            if self.use_h_inter
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
            init_mode (str, optional): Initialization mode. Must be 'zeros' or 'normal'. Defaults to 'zeros'.
            device (torch.device, optional): Device to allocate the output on. Defaults to None.

        Returns:
            torch.Tensor: The initialized output.
        """

        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")

        return func(batch_size, self.out_channels, *self.out_size, device=device)

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
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")

        return (
            func(batch_size, self.fb_channels, *self.in_size, device=device)
            if self.use_fb
            else None
        )

    def forward(
        self,
        input: torch.Tensor,
        h_pyr: torch.Tensor,
        h_inter: Optional[torch.Tensor] = None,
        fb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the cell.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_size[0], in_size[1]).
            h_pyr (torch.Tensor): Pyramidal cell hidden state of shape (batch_size, h_pyr_channels, in_size[0], in_size[1]).
            h_inter (Optional[torch.Tensor]): Interneuron cell hidden state of shape (batch_size, h_inter_channels_sum, inter_size[0], inter_size[1]).
            fb (Optional[torch.Tensor]): Feedback input of shape (batch_size, fb_channels, in_size[0], in_size[1]).

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing the output, new pyramidal cell hidden state, and new interneuron cell hidden state.
        """

        if self.use_fb and fb is None:
            raise ValueError("If use_fb is True, fb must be provided.")

        # Compute excitations for pyramidal cells
        exc_cat = torch.cat(
            [h_pyr, input, fb] if self.use_fb else [h_pyr, input], dim=1
        )
        if self.use_three_compartments:
            exc_pyr_soma = self.pre_inh_activation(self.conv_exc_pyr(h_pyr))
            exc_pyr_basal = self.pre_inh_activation(self.conv_exc_pyr_input(input))
            if self.use_fb:
                exc_pyr_apical = self.pre_inh_activation(self.conv_exc_pyr_fb(fb))
        else:
            exc_pyr_soma = self.pre_inh_activation(self.conv_exc_pyr(exc_cat))

        inhs = [0] * 4
        if self.h_inter_channels:
            # Compute excitations for interneurons
            if len(self.h_inter_channels) == 4:
                exc_inter = self.pre_inh_activation(self.conv_exc_inter(h_pyr))
                exc_input_inter = self.pre_inh_activation(
                    self.conv_exc_input_inter(input)
                )
                exc_fb_inter = self.pre_inh_activation(self.conv_exc_inter_fb(fb))
            else:
                exc_inter = self.pre_inh_activation(self.conv_exc_inter(exc_cat))

            # Compute inhibitions for all neurons
            h_inters = h_inter
            if self.immediate_inhibition:
                if h_inter is not None:
                    raise ValueError(
                        "If h_inter_channels is provided and immediate_inhibition is True, h_inter must not be provided."
                    )
                h_inters = exc_inter
            elif h_inter is None:
                raise ValueError(
                    "If h_inter_channels is provided and immediate_inhibition is False, h_inter must be provided."
                )

            h_inters = torch.split(
                h_inters,
                self.h_inter_channels,
                dim=1,
            )
            inh_inter_2 = inh_inter_3 = 0
            for i in range(len(self.h_inter_channels)):
                conv = self.convs_inh[i]
                if i in (2, 3):
                    conv, conv2 = conv.conv1, conv.conv2
                    if i == 2:
                        inh_inter_3 = self.pre_inh_activation(conv2(h_inters[i]))
                    else:
                        inh_inter_2 = self.pre_inh_activation(conv2(h_inters[i]))
                inhs[i] = self.pre_inh_activation(conv(h_inters[i]))
        inh_pyr_soma, inh_inter, inh_pyr_basal, inh_pyr_apical = inhs

        # Compute new pyramidal cell hidden state
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

        # Compute Euler update for pyramidal cell hidden state
        tau_pyr = torch.sigmoid(self.tau_pyr)
        h_pyr = (1 - tau_pyr) * h_pyr + tau_pyr * h_pyr_new

        # Compute new interneuron cell hidden state
        if self.use_h_inter:
            h_inter_new = exc_inter - inh_inter
            if len(self.h_inter_channels) == 4:
                # Add excitations and inhibitions to interneuron 2
                start = sum(self.h_inter_channels[:2])
                end = start + self.h_inter_channels[2]
                h_inter_new[:, start:end, ...] = (
                    h_inter_new[:, start:end, ...] + exc_input_inter - inh_inter_2
                )
                # Add excitations and inhibitions to interneuron 3
                start = sum(self.h_inter_channels[:3])
                h_inter_new[:, start:, ...] = (
                    h_inter_new[:, start:, ...] + exc_fb_inter - inh_inter_3
                )
            h_inter_new = self.post_integration_activation(
                self.post_inh_activation(h_inter_new)
            )
            # Compute Euler update for interneuron cell hidden state
            tau_inter = torch.sigmoid(self.tau_inter)
            h_inter = (1 - tau_inter) * h_inter + tau_inter * h_inter_new

        # Pool the output
        out = self.out_pool(h_pyr)

        return out, h_pyr, h_inter


class Conv2dEIRNN(nn.Module):
    """
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
        exc_kernel_size (list[int, int] | list[list[int, int]]): Kernel size for
            excitatory convolutions in each layer.
        inh_kernel_size (list[int, int] | list[list[int, int]]): Kernel size for
            inhibitory convolutions in each layer.
        use_three_compartments (bool): Whether to use a three-compartment model for
            pyramidal cells.
        immediate_inhibition (bool): Whether interneurons provide immediate inhibition.
        num_layers (int): Number of layers in the RNN.
        inter_mode (str): Mode for handling interneuron size relative to input size
            ('half' or 'same').
        layer_time_delay (bool): Whether to introduce a time delay between layers.
        exc_rectify (Optional[str]): Activation function for excitatory weights (e.g., \
            'relu').
        inh_rectify (Optional[str]): Activation function for inhibitory weights (e.g., \
            'relu').
        hidden_init_mode (str): Initialization mode for hidden states ('zeros' or
            'normal').
        fb_init_mode (str): Initialization mode for feedback input ('zeros' or \
            'normal').
        out_init_mode (str): Initialization mode for output ('zeros' or 'normal').
        fb_adjacency (Optional[torch.Tensor]): Adjacency matrix for feedback \
            connections.
        pool_kernel_size (list[int, int] | list[list[int, int]]): Kernel size for \
            pooling in each layer.
        pool_stride (list[int, int] | list[list[int, int]]): Stride for pooling in \
            each layer.
        bias (bool | list[bool]): Whether to add bias for convolutions in each layer.
        pre_inh_activation (Optional[str]): Activation function applied before \
            inhibition.
        post_inh_activation (Optional[str]): Activation function applied after \
            inhibition (to pyramidal cell).
        post_integration_activation (Optional[str]): Activation function applied \
            after integration (to pyramidal and interneurons).
        batch_first (bool): Whether the input tensor has batch dimension as the \
            first dimension.
    """

    def __init__(
        self,
        in_size: tuple[int, int],
        in_channels: int,
        h_pyr_channels: int | tuple[int] = 16,
        h_inter_channels: tuple[int] | tuple[tuple[int]] = (16,),
        fb_channels: Optional[int | tuple[int]] = None,
        inter_mode: str = "half",
        use_three_compartments: bool = False,
        immediate_inhibition: bool = False,
        exc_rectify: bool = False,
        inh_rectify: bool = False,
        exc_kernel_size: tuple[int, int] | tuple[tuple[int, int]] = (3, 3),
        inh_kernel_size: tuple[int, int] | tuple[tuple[int, int]] = (3, 3),
        fb_kernel_size: tuple[int, int] | tuple[tuple[int, int]] = (3, 3),
        pool_kernel_size: tuple[int, int] | tuple[tuple[int, int]] = (3, 3),
        pool_stride: tuple[int, int] | tuple[tuple[int, int]] = (2, 2),
        pool_global: bool | tuple[bool] = False,
        pre_inh_activation: Optional[str] = "tanh",
        post_inh_activation: Optional[str] = None,
        post_integration_activation: Optional[str] = None,
        tau_mode: Optional[str] = "channel",
        fb_activation: Optional[str] = None,
        bias: bool | tuple[bool] = True,
        num_layers: int = 1,
        layer_time_delay: bool = False,
        fb_adjacency: Optional[tuple[tuple[int | bool]] | torch.Tensor] = None,
        hidden_init_mode: str = "zeros",
        fb_init_mode: str = "zeros",
        out_init_mode: str = "zeros",
        batch_first: bool = False,
    ):
        super().__init__()

        # Expand the layer specific parameters as necessary
        self.num_layers = num_layers
        self.h_pyr_channels = expand_list(h_pyr_channels, self.num_layers)
        self.h_inter_channels = expand_list(
            h_inter_channels,
            self.num_layers,
            depth=0 if h_inter_channels is None else 1,
        )
        self.fb_channels = expand_list(fb_channels, self.num_layers)
        self.exc_kernel_sizes = expand_list(exc_kernel_size, self.num_layers, depth=1)
        self.inh_kernel_sizes = expand_list(inh_kernel_size, self.num_layers, depth=1)
        self.fb_kernel_sizes = expand_list(fb_kernel_size, self.num_layers, depth=1)
        self.pool_kernel_sizes = expand_list(pool_kernel_size, self.num_layers, depth=1)
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
        fb_activation_class = get_activation_class(fb_activation)

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
                raise ValueError("fb_adjacency must be a non-zero tensor if provided.")

            # Create feedback convolutions
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
                    self.fb_convs[f"fb_conv_{i}_{j}"] = nn.Sequential(
                        nn.Upsample(size=self.in_sizes[j], mode="bilinear"),
                        Conv2dFb(
                            in_channels=self.h_pyr_channels[i],
                            out_channels=self.fb_channels[j],
                            kernel_size=self.fb_kernel_sizes[j],
                            padding=(
                                self.fb_kernel_sizes[j][0] // 2,
                                self.fb_kernel_sizes[j][1] // 2,
                            ),
                            bias=self.biases[j],
                        ),
                        fb_activation_class(),
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
                    in_channels=(in_channels if i == 0 else self.h_pyr_channels[i - 1]),
                    h_pyr_channels=self.h_pyr_channels[i],
                    h_inter_channels=self.h_inter_channels[i],
                    fb_channels=self.fb_channels[i] if self.receives_fb[i] else 0,
                    inter_mode=inter_mode,
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
                the initialized pyramidal and interneuron hidden states for each layer.
        """
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
        """
        Initializes the internal state of the network.

        Args:
            out_0 (Optional[list[torch.Tensor]]): Initial outputs for each layer.
            h_pyr_0 (Optional[list[torch.Tensor]]): Initial pyramidal cell hidden states for each layer.
            h_inter_0 (Optional[list[torch.Tensor]]): Initial interneuron cell hidden states for each layer.
            fb_0 (Optional[list[torch.Tensor]]): Initial feedback inputs for each layer.
            num_steps (int): Number of time steps.
            batch_size (int): Batch size.
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            tuple(list[list[torch.Tensor]], list[list[torch.Tensor]], list[list[torch.Tensor]], list[list[torch.Tensor]]): A tuple containing the initialized outputs, pyramidal cell hidden states, interneuron cell hidden states, and feedback inputs for each layer and time step.
        """
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
    def _modulation_identity(x, i, t):
        return x

    def _format_modulation_fns(
        self, modulation_out_fn, modulation_pyr_fn, modulation_inter_fn
    ):
        """
        Formats the modulation functions.

        Args:
            modulation_out_fn (Optional[torch.Tensor]): Modulation function for outputs.
            modulation_pyr_fn (Optional[torch.Tensor]): Modulation function for pyramidal cell hidden states.
            modulation_inter_fn (Optional[torch.Tensor]): Modulation function for interneuron cell hidden states.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): The formatted modulation functions.
        """
        modulation_fns = [modulation_out_fn, modulation_pyr_fn, modulation_inter_fn]
        for i, fn in enumerate(modulation_fns):
            if fn is None:
                modulation_fns[i] = self._modulation_identity
            elif not callable(fn):
                raise ValueError("modulation_fns must be callable or None.")

        return modulation_fns

    def _format_outputs(self, outs, h_pyrs, h_inters, fbs):
        """
        Formats the outputs, hidden states, and feedback inputs.

        Args:
            outs (list[list[torch.Tensor]]): Outputs for each layer and time step.
            h_pyrs (list[list[torch.Tensor]]): Pyramidal cell hidden states for each layer and time step.
            h_inters (list[list[torch.Tensor]]): Interneuron cell hidden states for each layer and time step.
            fbs (list[list[torch.Tensor]]): Feedback inputs for each layer and time step.

        Returns:
            tuple(list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]): The formatted outputs, pyramidal cell hidden states, interneuron cell hidden states, and feedback inputs.
        """
        for i in range(self.num_layers):
            outs[i] = torch.stack(outs[i])
            h_pyrs[i] = torch.stack(h_pyrs[i])
            if self.h_inter_channels[i] and not self.immediate_inhibition:
                h_inters[i] = torch.stack(h_inters[i])  # TODO: Check if this is correct
            else:
                assert not self.h_inter_channels[i] or self.immediate_inhibition
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
                if self.h_inter_channels[i] and not self.immediate_inhibition:
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
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_size[0], in_size[1]).
            num_steps (Optional[int]): Number of time steps.
            out_0 (Optional[list[torch.Tensor]]): Initial outputs for each layer.
            h_pyr_0 (Optional[list[torch.Tensor]]): Initial pyramidal cell hidden states for each layer.
            h_inter_0 (Optional[list[torch.Tensor]]): Initial interneuron cell hidden states for each layer.
            fb_0 (Optional[list[torch.Tensor]]): Initial feedback inputs for each layer.
            modulation_out_fn (Optional[torch.Tensor]): Modulation function for outputs.
            modulation_pyr_fn (Optional[torch.Tensor]): Modulation function for pyramidal cell hidden states.
            modulation_inter_fn (Optional[torch.Tensor]): Modulation function for interneuron cell hidden states.
            return_all_layers_out (bool): Whether to return outputs for all layers.

        Returns:
            tuple(torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]): The output tensor, pyramidal cell hidden states, interneuron cell hidden states, and feedback inputs.
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
