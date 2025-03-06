import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from bioplnn.utils import (
    expand_array_2d,
    expand_list,
    get_activation,
    init_tensor,
)

TensorInitFnType = Callable[..., torch.Tensor]
ActivationFnType = Callable[[torch.Tensor], torch.Tensor]
T = TypeVar("T", bound=Any)
Param1dType = Union[T, list[T], NDArray[T]]
Param2dType = Union[T, list[list[T]], NDArray[T]]


def circuit_connectivity_df(
    num_exc: int,
    num_inh: int,
    use_fb: bool,
    circuit_connectivity: Optional[Param2dType] = None,
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
        circuit_connectivity = [[0] * len(column_labels)] * len(row_labels)

    df = pd.DataFrame(
        circuit_connectivity, index=row_labels, columns=column_labels
    )
    df.index.name = "from"
    df.columns.name = "to"

    return df


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


@dataclass(kw_only=True, slots=True)
class Conv2dEIRNNLayerConfig:
    """
    Configuration class for Conv2dEIRNN layers.

    Args:
        spatial_size (tuple[int, int]): Size of the input data (height, width).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_neuron_types (int, optional): Number of neuron types. Defaults to 1.
        neuron_channels (Param1dType[int], optional): Number of channels for each
            neuron type. Defaults to 16.
        neuron_type (Param1dType[str], optional): Type of neuron. Defaults to
            "excitatory".
        neuron_spatial_mode (Param1dType[str], optional): Spatial mode for the
            neuron. Defaults to "same".
        fb_channels (int, optional): Number of feedback channels. Defaults to
            None.
        conv_connectivity (Param2dType[int | bool], optional): Connectivity
            matrix for the convolutions. Defaults to [[1, 0], [0, 1]].
        conv_rectify (Param2dType[bool], optional): Whether to rectify the
            convolutions. Defaults to False.
        conv_kernel_size (Param2dType[tuple[int, int]], optional): Kernel size
            for the convolutions. Defaults to (3, 3).
        conv_activation (Param2dType[str | ActivationFnType], optional):
            Activation function for the convolutions. Defaults to None.
        conv_bias (Param2dType[bool], optional): Whether to add a bias term for
            the convolutions. Defaults to True.
        post_agg_activation (Param1dType[str | ActivationFnType], optional):
            Activation function for the post-aggregation. Defaults to "Tanh".
        tau_mode (Param1dType[str], optional): Mode for handling membrane time
            constants. Defaults to "channel".
        tau_init_mode (Param1dType[str | TensorInitFnType], optional):
            Initialization mode for the membrane time constants. Defaults to
            "zeros".
        default_hidden_init_mode (str | TensorInitFnType, optional):
            Initialization mode for the hidden state. Defaults to "zeros".
        default_fb_init_mode (str | TensorInitFnType, optional): Initialization
            mode for the feedback. Defaults to "zeros".
        default_out_init_mode (str | TensorInitFnType, optional): Initialization
            mode for the output. Defaults to "zeros".
    """

    spatial_size: tuple[int, int]
    in_channels: int
    out_channels: int
    num_neuron_types: int = 1
    neuron_channels: Param1dType[int] = 16
    neuron_type: Param1dType[str] = "excitatory"
    neuron_spatial_mode: Param1dType[str] = "same"
    fb_channels: Optional[int] = None
    conv_connectivity: Param2dType[int | bool] = field(
        default_factory=lambda: [[1, 0], [0, 1]]
    )
    conv_rectify: Param2dType[bool] = False
    conv_kernel_size: Param2dType[tuple[int, int]] = (3, 3)
    conv_activation: Optional[Param2dType[str | ActivationFnType]] = None
    conv_bias: Param2dType[bool] = True
    post_agg_activation: Optional[Param1dType[str | ActivationFnType]] = "Tanh"
    tau_mode: Optional[Param1dType[str]] = "channel"
    tau_init_mode: Param1dType[str | TensorInitFnType] = "zeros"
    default_hidden_init_mode: str | TensorInitFnType = "zeros"
    default_fb_init_mode: str | TensorInitFnType = "zeros"
    default_out_init_mode: str | TensorInitFnType = "zeros"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Conv2dEIRNNLayer(nn.Module):
    """
    Implements a 2D convolutional recurrent neural network layer with excitatory
    and inhibitory neurons.

    This layer implements the core computational unit of the EIRNN architecture,
    supporting multiple neuron types (excitatory and inhibitory) with configurable
    connectivity patterns, activation functions, and time constants.

    Args:
        config (Optional[Conv2dEIRNNLayerConfig]): Configuration object that specifies
            the layer architecture and parameters. See Conv2dEIRNNLayerConfig for details.
            If None, parameters must be provided as keyword arguments.
        **kwargs: Keyword arguments that can be used to override or provide parameters
            not specified in the config object. These will be used to populate the config
            if one is not provided.

    The layer implements:
    - Configurable neuron types (excitatory/inhibitory)
    - Convolutional connectivity between neuron populations
    - Recurrent dynamics with learnable time constants
    - Optional feedback connections
    - Customizable activation functions

    Raises:
        ValueError: If invalid configuration arguments are provided.
    """

    def __init__(
        self, config: Optional[Conv2dEIRNNLayerConfig] = None, **kwargs
    ):
        super().__init__()

        #####################################################################
        # Input, output, and feedback dimensions
        #####################################################################

        if config is None:
            config = Conv2dEIRNNLayerConfig(**kwargs)

        self.spatial_size = config.spatial_size
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.fb_channels = (
            config.fb_channels if config.fb_channels is not None else 0
        )
        self.use_fb = self.fb_channels > 0

        #####################################################################
        # Neuron dimensions
        #####################################################################
        self.num_neuron_types = config.num_neuron_types

        # Format neuron type
        self.neuron_channels = expand_list(
            config.neuron_channels, self.num_neuron_types, depth=1
        )
        self.neuron_type = expand_list(
            config.neuron_type, self.num_neuron_types, depth=1
        )
        self.neuron_spatial_mode = expand_list(
            config.neuron_spatial_mode, self.num_neuron_types, depth=1
        )

        if not set(self.neuron_type) <= {"excitatory", "inhibitory"}:
            raise ValueError(
                "neuron_type for each neuron type must be 'excitatory' or 'inhibitory'."
            )
        if not set(self.neuron_spatial_mode) <= {"same", "half"}:
            raise ValueError(
                "neuron_spatial_mode for each neuron type must be 'same' or 'half'."
            )

        # Save number of "types" for the input to and output from the layer
        self.num_input_types = (
            1 + int(self.use_fb) + self.num_neuron_types
        )  # input + fb + neurons
        self.num_output_types = self.num_neuron_types + 1  # neurons + output

        # Calculate half spatial size
        self.half_spatial_size = (
            ceil(self.spatial_size[0] / 2),
            ceil(self.spatial_size[1] / 2),
        )

        # Calculate spatial size for each neuron type based on spatial mode
        self.neuron_spatial_size = [
            self.spatial_size if mode == "same" else self.half_spatial_size
            for mode in self.neuron_spatial_mode
        ]
        #####################################################################
        # Circuit motif connectivity
        #####################################################################

        # Format circuit connectivity
        self.conv_connectivity = np.array(config.conv_connectivity)
        if self.conv_connectivity.shape != (
            self.num_output_types,
            self.num_input_types,
        ):
            raise ValueError(
                "The shape of conv_connectivity must match the number of output and input types."
            )

        # Format connectivity variables to match circuit connectivity
        self.conv_kernel_size = expand_array_2d(
            config.conv_kernel_size,
            self.conv_connectivity.shape[0],
            self.conv_connectivity.shape[1],
            depth=2,
        )
        self.conv_rectify = expand_array_2d(
            config.conv_rectify,
            self.conv_connectivity.shape[0],
            self.conv_connectivity.shape[1],
            depth=1,
        )
        self.conv_activation = expand_array_2d(
            config.conv_activation,
            self.conv_connectivity.shape[0],
            self.conv_connectivity.shape[1],
            depth=1,
        )
        self.conv_bias = expand_array_2d(
            config.conv_bias,
            self.conv_connectivity.shape[0],
            self.conv_connectivity.shape[1],
            depth=1,
        )

        #####################################################################
        # Circuit motif convolutions
        # Here, we represent the circuit connectivity between neuron classes
        # as an array of convolutions (implemented as a dictionary for
        # efficiency). The convolution self.convs[f"{i}->{j}"] corresponds
        # to the connection from neuron class i to neuron class j.
        #####################################################################

        self.convs = nn.ModuleDict()
        for i, row in enumerate(self.conv_connectivity):
            # Handle input neuron channel and spatial mode based on neuron type
            conv_in_type = self.input_type_from_idx(i)
            if conv_in_type == "input":
                conv_in_channels = self.in_channels
                conv_in_spatial_mode = "same"
            elif conv_in_type == "feedback":
                conv_in_channels = self.fb_channels
                conv_in_spatial_mode = "same"
            elif conv_in_type in ("excitatory", "inhibitory"):
                conv_in_channels = self.neuron_channels[
                    i - 1 - int(self.use_fb)
                ]
                conv_in_spatial_mode = self.neuron_spatial_mode[
                    i - 1 - int(self.use_fb)
                ]
            else:
                raise ValueError(f"Invalid neuron type: {conv_in_type}.")

            if conv_in_spatial_mode not in ("same", "half"):
                raise ValueError(
                    f"Invalid neuron spatial mode: {conv_in_spatial_mode}."
                )

            # Handle output neurons
            output_indices = np.nonzero(row)[0]
            for j in output_indices:
                if self.conv_connectivity[i, j]:
                    # Handle rectification
                    if self.conv_rectify[i, j]:
                        Conv2d = Conv2dRectify
                    else:
                        Conv2d = nn.Conv2d

                    # Handle output neuron channel and spatial mode based on neuron type
                    conv_out_type = self.output_type_from_idx(j)
                    if conv_out_type in ("excitatory", "inhibitory"):
                        conv_out_channels = self.neuron_channels[j]
                        conv_out_spatial_mode = self.neuron_spatial_mode[j]
                    else:
                        conv_out_channels = self.out_channels
                        conv_out_spatial_mode = "same"

                    # Handle stride upsampling if necessary
                    conv = nn.Sequential()
                    conv_stride = 1
                    if (
                        conv_in_spatial_mode == "half"
                        and conv_out_spatial_mode == "same"
                    ):
                        conv_stride = 2
                    elif (
                        conv_in_spatial_mode == "same"
                        and conv_out_spatial_mode == "half"
                    ):
                        conv.append(
                            nn.Upsample(
                                size=self.spatial_size, mode="bilinear"
                            )
                        )

                    # Handle upsampling if necessary
                    conv.append(
                        Conv2d(
                            in_channels=conv_in_channels,
                            out_channels=conv_out_channels,
                            kernel_size=self.conv_kernel_size[i, j],
                            stride=conv_stride,
                            padding=(
                                self.conv_kernel_size[i, j][0] // 2,
                                self.conv_kernel_size[i, j][1] // 2,
                            ),
                            bias=self.conv_bias[i, j],
                        )
                    )

                    self.convs[f"{i}->{j}"] = conv

        #####################################################################
        # Post convolution operations
        #####################################################################

        # Format post convolution activation functions
        self.post_agg_activation = expand_list(
            config.post_agg_activation, self.num_neuron_types
        )

        # Initialize membrane time constants
        self.tau_mode = expand_list(config.tau_mode, self.num_neuron_types)
        self.tau_init_mode = expand_list(
            config.tau_init_mode, self.num_neuron_types
        )

        self.tau = nn.ParameterList()
        for i in range(self.num_neuron_types):
            tau_channels = 1
            tau_spatial_size = (1, 1)
            match self.tau_mode[i]:
                case "spatial":
                    tau_spatial_size = self.neuron_spatial_size[i]
                case "channel":
                    tau_channels = self.neuron_channels[i]
                case "channel_spatial":
                    tau_channels = self.neuron_channels[i]
                    tau_spatial_size = self.neuron_spatial_size[i]
                case None:
                    pass
                case _:
                    raise ValueError(
                        f"Invalid tau_mode: {self.tau_mode[i]}. Must be one of: 'spatial', 'channel', 'channel_spatial', or None."
                    )

            self.tau.append(
                nn.Parameter(
                    init_tensor(
                        self.tau_init_mode[i],
                        1,
                        tau_channels,
                        *tau_spatial_size,
                    )
                )
            )

        #####################################################################
        # Tensor initialization
        #####################################################################

        self.default_hidden_init_mode = config.default_hidden_init_mode
        self.default_fb_init_mode = config.default_fb_init_mode
        self.default_out_init_mode = config.default_out_init_mode

    def input_type_from_idx(self, idx: int) -> str:
        if idx == 0:
            return "input"
        elif self.use_fb and idx == 1:
            return "feedback"
        else:
            return self.neuron_type[idx - 1 - int(self.use_fb)]

    def output_type_from_idx(self, idx: int) -> str:
        if idx < self.num_output_types - 1:
            return self.neuron_type[idx]
        else:
            return "output"

    def init_hidden(
        self,
        batch_size: int,
        init_mode: Optional[str | TensorInitFnType] = None,
        device: Optional[torch.device] = None,
    ) -> list[torch.Tensor]:
        """
        Initializes the neuron hidden states.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros',
                'ones', 'randn', 'rand', a function, or None. If None, the
                default initialization mode will be used. If a function, it
                must take a variable number of positional arguments corresponding
                to the shape of the tensor to initialize, as well as a
                `device` keyword argument that send.
            device (torch.device, optional): Device to allocate the hidden
                states on. Defaults to None.

        Returns:
            list[torch.Tensor]: A list containing the initialized neuron
                hidden states.
        """

        init_mode = (
            init_mode
            if init_mode is not None
            else self.default_hidden_init_mode
        )
        if isinstance(init_mode, str):
            return [
                init_tensor(
                    init_mode,
                    batch_size,
                    self.neuron_channels[i],
                    *self.spatial_size,
                    device=device,
                )
                for i in range(self.num_neuron_types)
            ]
        else:
            try:
                return [
                    init_mode(
                        batch_size,
                        self.neuron_channels[i],
                        *self.spatial_size,
                        device=device,
                    )
                    for i in range(self.num_neuron_types)
                ]
            except TypeError:
                return [
                    init_mode(
                        batch_size,
                        self.neuron_channels[i],
                        *self.spatial_size,
                    ).to(device)
                    for i in range(self.num_neuron_types)
                ]

    def init_out(
        self,
        batch_size: int,
        init_mode: Optional[str | TensorInitFnType] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Initializes the output.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros',
                'ones', 'randn', 'rand', a function, or None. If None, the
                default initialization mode will be used.
            device (torch.device, optional): Device to allocate the hidden
                states on. Defaults to None.

        Returns:
            torch.Tensor: The initialized output.
        """

        init_mode = (
            init_mode if init_mode is not None else self.default_out_init_mode
        )
        if isinstance(init_mode, str):
            return init_tensor(
                init_mode,
                batch_size,
                self.out_channels,
                *self.spatial_size,
                device=device,
            )
        else:
            return init_mode(
                batch_size,
                self.out_channels,
                *self.spatial_size,
                device=device,
            )

    def init_fb(
        self,
        batch_size: int,
        init_mode: Optional[str | TensorInitFnType] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor | None:
        """
        Initializes the feedback input.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros',
                'ones', 'randn', 'rand', a function, or None. If None, the
                default initialization mode will be used.
            device (torch.device, optional): Device to allocate the hidden
                states on. Defaults to None.

        Returns:
            Optional[torch.Tensor]: The initialized feedback input if `use_fb` is
                True, otherwise None.
        """

        if not self.use_fb:
            return None

        init_mode = (
            init_mode if init_mode is not None else self.default_fb_init_mode
        )
        if isinstance(init_mode, str):
            return init_tensor(
                init_mode,
                batch_size,
                self.fb_channels,
                *self.spatial_size,
                device=device,
            )
        else:
            return init_mode(
                batch_size,
                self.fb_channels,
                *self.spatial_size,
                device=device,
            )

    def forward(
        self,
        input: torch.Tensor,
        h_neuron: torch.Tensor | list[torch.Tensor],
        fb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass of the EIRNN layer.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_size[0], in_size[1]).
            h_neuron (torch.Tensor | list[torch.Tensor]): List of neuron hidden states of shape (batch_size, neuron_channels[i], in_size[0], in_size[1]) for each neuron type i.
            fb (Optional[torch.Tensor]): Feedback input of shape (batch_size, fb_channels, in_size[0], in_size[1]).

        Returns:
            tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]]: A tuple containing the output and new neuron hidden state.
        """
        # Expand h_neuron to match the number of neuron channels
        if isinstance(h_neuron, torch.Tensor):
            if self.num_neuron_types != 1:
                raise ValueError(
                    "h_neuron must be a list of tensors if num_neuron_types is not 1."
                )
            h_neuron = [h_neuron]

        # Check if feedback is provided if necessary
        if self.use_fb == fb is None:
            raise ValueError(
                "use_fb must be True if and only if fb is provided."
            )

        # Compute convolutions for each connection in the circuit
        conv_ins = [input] + ([fb] if self.use_fb else []) + h_neuron
        conv_outs = [[] for _ in range(self.num_neuron_types + 1)]
        for key, conv in self.convs.items():
            i, j = key.split("->")
            i, j = int(i), int(j)
            if self.input_type_from_idx(i) == "inhibitory":
                sign = -1
            else:
                sign = 1

            conv_outs[j].append(sign * conv(conv_ins[i]))

        conv_outs = [torch.cat(conv_out, dim=1) for conv_out in conv_outs]
        h_neuron_new = conv_outs[:-1]
        out = conv_outs[-1]

        # Compute Euler update for excitatory cell hidden state
        for i in range(self.num_neuron_types):
            tau = torch.sigmoid(self.tau[i])
            h_neuron_new[i] = tau * h_neuron_new[i] + (1 - tau) * h_neuron[i]

        return out, h_neuron_new


class Conv2dEIRNN(nn.Module):
    """
    Implements a deep convolutional recurrent neural network with excitatory-inhibitory
    neurons (EIRNN).

    This module creates a multi-layer convolutional EIRNN network by stacking multiple
    Conv2dEIRNNLayer instances with optional feedback connections between them.

    Args:
        num_layers (int): Number of EIRNN layers in the network. Defaults to 1.
        layer_configs (Conv2dEIRNNLayerConfig | list[Conv2dEIRNNLayerConfig], optional):
            Configuration object(s) for the EIRNN layers. If provided as a list, must match
            the number of layers. If a single config is provided, it will be used for all layers
            with appropriate adjustments. Defaults to None.
        layer_kwargs (Mapping[str, Any] | list[Mapping[str, Any]], optional):
            Additional keyword arguments for each layer. If provided as a list, must match
            the number of layers. Defaults to None.
        common_layer_kwargs (Mapping[str, Any], optional):
            Keyword arguments to apply to all layers. Defaults to None.
        fb_connectivity (Param2dType[int | bool], optional):
            Connectivity matrix for feedback connections between layers. Defaults to None.
        fb_activations (Param2dType[str], optional):
            Activation functions for feedback connections. Defaults to None.
        fb_rectify (Param2dType[bool]):
            Whether to rectify feedback connections. Defaults to False.
        fb_kernel_sizes (Param2dType[tuple[int, int]], optional):
            Kernel sizes for feedback convolutions. Defaults to None.
        pool_mode (str, optional):
            Pooling mode for layer outputs. Defaults to "avg".
        layer_time_delay (bool):
            Whether to introduce a time delay between layers. Defaults to False.
        batch_first (bool):
            Whether the input tensor has batch dimension as the first dimension. Defaults to True.
    """

    def __init__(
        self,
        *,
        num_layers: int = 1,
        layer_configs: Optional[list[Conv2dEIRNNLayerConfig]] = None,
        layer_kwargs: Optional[list[Mapping[str, Any]]] = None,
        common_layer_kwargs: Optional[Mapping[str, Any]] = None,
        fb_connectivity: Optional[Param2dType[int | bool]] = None,
        fb_activations: Optional[Param2dType[str]] = None,
        fb_rectify: Param2dType[bool] = False,
        fb_kernel_sizes: Optional[Param2dType[tuple[int, int]]] = None,
        layer_time_delay: bool = False,
        pool_mode: Optional[str] = "max",
        batch_first: bool = True,
    ):
        super().__init__()

        ############################################################
        # Layer configs
        ############################################################

        self.num_layers = num_layers

        if layer_configs is not None:
            if layer_kwargs is not None or common_layer_kwargs is not None:
                raise ValueError(
                    "layer_configs cannot be provided if layer_configs_kwargs "
                    "or common_layer_config_kwargs is provided."
                )
            if len(layer_configs) != self.num_layers:
                raise ValueError("layer_configs must be of length num_layers.")
        else:
            if layer_kwargs is None:
                if common_layer_kwargs is None:
                    raise ValueError(
                        "layer_kwargs or common_layer_kwargs must be provided if "
                        "layer_configs is not provided."
                    )
                layer_kwargs = [{}] * self.num_layers  # type: ignore
            elif len(layer_kwargs) != self.num_layers:
                raise ValueError("layer_kwargs must be of length num_layers.")

            if common_layer_kwargs is None:
                common_layer_kwargs = {}

            layer_configs = [
                Conv2dEIRNNLayerConfig(
                    **common_layer_kwargs,
                    **layer_kwargs[i],  # type: ignore
                )
                for i in range(self.num_layers)
            ]

        # Validate layer configurations
        for i in range(self.num_layers - 1):
            if (
                layer_configs[i].out_channels
                != layer_configs[i + 1].in_channels
            ):
                raise ValueError(
                    f"The output channels of layer {i} must match the input "
                    f"channels of layer {i + 1}."
                )

        # Check if the first layer input projections are rectified
        warn = False
        if isinstance(layer_configs[0].conv_rectify, bool):
            if layer_configs[0].conv_rectify:
                warn = True
        else:
            if any(layer_configs[0].conv_rectify):
                warn = True
        if warn:
            warnings.warn(
                "Rectification of network input may hinder learning! "
                "Consider setting conv_rectify to False for the input "
                "projections to the first layer or for the whole first "
                "layer (i.e. set layer_configs[0].conv_rectify[0,:] or "
                "layer_configs[0].conv_rectify to False). \n"
                "If your input to the network is already purely positive, "
                "you can safely ignore this warning."
            )

        ############################################################
        # RNN parameters
        ############################################################

        self.layer_time_delay = layer_time_delay
        self.pool_mode = pool_mode
        self.batch_first = batch_first

        ############################################################
        # Initialize layers
        ############################################################

        # Create layers
        self.layers = nn.ModuleList(
            [Conv2dEIRNNLayer(layer_config) for layer_config in layer_configs]
        )

        ############################################################
        # Initialize feedback connections
        ############################################################

        if fb_connectivity is None:
            if any(layer_config.fb_channels for layer_config in layer_configs):
                raise ValueError(
                    "fb_connectivity must be provided if and only if "
                    "fb_channels is provided for at least one layer."
                )
            self.fb_convs = nn.ModuleDict()
        else:
            fb_connectivity = np.array(fb_connectivity, dtype=bool)
            if fb_connectivity.shape != (
                self.num_layers,
                self.num_layers,
            ):
                raise ValueError(
                    "The shape of fb_connectivity must be (num_layers, num_layers)."
                )
            fb_activations = expand_array_2d(
                fb_activations,
                self.num_layers,
                self.num_layers,
            )
            fb_rectify = expand_array_2d(
                fb_rectify,
                self.num_layers,
                self.num_layers,
            )
            fb_kernel_sizes = expand_array_2d(
                fb_kernel_sizes,
                self.num_layers,
                self.num_layers,
            )

            # Validate fb_adjacency tensor
            if (
                fb_connectivity.ndim != 2
                or fb_connectivity.shape[0] != self.num_layers
                or fb_connectivity.shape[1] != self.num_layers
            ):
                raise ValueError(
                    "The dimensions of fb_connectivity must match the number of layers."
                )
            if fb_connectivity.sum() == 0:
                raise ValueError(
                    "fb_connectivity must be a non-zero tensor if provided."
                )

            # Create feedback convolutions
            self.fb_convs = nn.ModuleDict()
            for i, row in enumerate(fb_connectivity):
                nonzero_indices = np.nonzero(row)[0]
                for j in nonzero_indices:
                    if not self.layers[j].use_fb:
                        raise ValueError(
                            f"the connection from layer {i} to layer {j} is "
                            f"not valid because layer {j} does not receive "
                            f"feedback (hint: fb_channels may not be provided)"
                        )
                    if fb_rectify[i, j]:
                        Conv2dFb = Conv2dRectify
                    else:
                        Conv2dFb = nn.Conv2d
                    self.fb_convs[f"{i}->{j}"] = nn.Sequential(
                        nn.Upsample(
                            size=layer_configs[j].spatial_size,
                            mode="bilinear",
                        ),
                        Conv2dFb(
                            in_channels=layer_configs[i].out_channels,
                            out_channels=layer_configs[j].fb_channels,
                            kernel_size=layer_configs[j].fb_kernel_size,
                            padding=(
                                fb_kernel_sizes[j][0] // 2,
                                fb_kernel_sizes[j][1] // 2,
                            ),
                            bias=layer_configs[j].fb_bias,
                        ),
                        get_activation(fb_activations[i, j]),
                    )

    def _init_hidden(
        self,
        batch_size: int,
        init_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> list[list[torch.Tensor]]:
        """
        Initializes the hidden states for all layers.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode ('zeros' or 'normal').
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list containing the initialized neuron hidden states for each layer.
        """
        return [
            layer.init_hidden(batch_size, init_mode, device)
            for layer in self.layers
        ]

    def _init_fb(
        self,
        batch_size: int,
        init_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> list[Optional[torch.Tensor]]:
        """
        Initializes the feedback inputs for all layers.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode ('zeros' or 'normal').
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list of initialized feedback inputs for each layer.
        """
        return [
            layer.init_fb(batch_size, init_mode, device)
            for layer in self.layers
        ]

    def _init_out(
        self,
        batch_size: int,
        init_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> list[torch.Tensor]:
        """
        Initializes the outputs for all layers.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode ('zeros' or 'normal').
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list of initialized outputs for each layer.
        """
        return [
            layer.init_out(batch_size, init_mode, device)
            for layer in self.layers
        ]

    def _init_state(
        self,
        out_0: Optional[Sequence[torch.Tensor | None]],
        h_neuron_0: Optional[Sequence[Sequence[torch.Tensor | None]]],
        fb_0: Optional[Sequence[torch.Tensor | None]],
        num_steps: int,
        batch_size: int,
        out_init_mode: Optional[str] = None,
        hidden_init_mode: Optional[str] = None,
        fb_init_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[
        list[list[torch.Tensor | None]],
        list[list[list[torch.Tensor | None]]],
        list[list[torch.Tensor | int | None]],
    ]:
        """
        Initializes the state of the network.

        Args:
            out_0 (Optional[list[torch.Tensor]]): Initial outputs for each layer.
            h_neuron_0 (Optional[list[torch.Tensor]]): Initial neuron hidden states for each layer.
            fb_0 (Optional[list[torch.Tensor]]): Initial feedback inputs for each layer.
            num_steps (int): Number of time steps.
            batch_size (int): Batch size.
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            tuple(NDArray[np.object_], NDArray[np.object_], NDArray[np.object_]): A tuple
                containing the initialized outputs, neuron hidden states, and
                feedback inputs for each layer and time step.
        """

        # Validate input shapes of out_0, h_neuron_0, fb_0
        if out_0 is None:
            out_0 = [None] * self.num_layers
        elif len(out_0) != self.num_layers:
            raise ValueError(
                "The length of out_0 must be equal to the number of layers."
            )
        if h_neuron_0 is None:
            h_neuron_0 = [
                [None] * self.layers[i].num_neuron_types
                for i in range(self.num_layers)
            ]
        elif len(h_neuron_0) != self.num_layers:
            raise ValueError(
                "The length of h_neuron_0 must be equal to the number of layers."
            )
        if fb_0 is None:
            fb_0 = [None] * self.num_layers
        elif len(fb_0) != self.num_layers:
            raise ValueError(
                "The length of fb_0 must be equal to the number of layers."
            )

        # Initialize default values
        h_neuron_0_default = self._init_hidden(
            batch_size, hidden_init_mode, device
        )
        fb_0_default = self._init_fb(batch_size, fb_init_mode, device)
        out_0_default = self._init_out(batch_size, out_init_mode, device)

        # Initialize output, hidden state, and feedback lists
        outs: list[list[torch.Tensor | None]] = [
            [None] * num_steps for _ in range(self.num_layers)
        ]
        h_neurons: list[list[list[torch.Tensor | None]]] = [
            [
                [None] * self.layers[i].num_neuron_types
                for _ in range(num_steps)
            ]
            for i in range(self.num_layers)
        ]
        fbs: list[list[torch.Tensor | int | None]] = [
            [0 if self.layers[i].use_fb else None] * num_steps
            for i in range(self.num_layers)
        ]

        # Fill time step -1 with the initial values
        for i in range(self.num_layers):
            outs[i][-1] = (
                out_0[i] if out_0[i] is not None else out_0_default[i]
            )
            fbs[i][-1] = fb_0[i] if fb_0[i] is not None else fb_0_default[i]
            for k in range(self.layers[i].num_neuron_types):
                h_neurons[i][-1][k] = (
                    h_neuron_0[i][k]
                    if h_neuron_0[i][k] is not None
                    else h_neuron_0_default[i][k]
                )

        return outs, h_neurons, fbs

    def _format_x(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, int]:
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

    def _format_result(
        self,
        outs: list[list[torch.Tensor]],
        h_neurons: list[list[list[torch.Tensor]]],
        fbs: list[list[torch.Tensor | None]],
    ) -> tuple[
        list[list[torch.Tensor]],
        list[list[list[torch.Tensor]]],
        list[list[torch.Tensor | None]],
    ]:
        """
        Formats the outputs, hidden states, and feedback inputs.

        Args:
            outs (list[list[torch.Tensor]]): Outputs for each layer and time step.
            h_neurons (list[list[torch.Tensor]]): Neuron hidden states for each layer and time step.
            fbs (list[list[torch.Tensor]]): Feedback inputs for each layer and time step.

        Returns:
            tuple(list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]): The formatted outputs, excitatory cell hidden states, inhibitory neuron cell hidden states, and feedback inputs.
        """
        outs_stack: list[torch.Tensor] = []
        h_neurons_stack: list[list[torch.Tensor]] = []
        fbs_stack: list[torch.Tensor | None] | None = []
        for i in range(self.num_layers):
            outs_stack.append(torch.stack(outs[i]))
            h_neurons_stack.append(
                [
                    torch.stack(
                        [h_neurons[i][t][j] for t in range(len(h_neurons[i]))]
                    )
                    for j in range(self.layers[i].num_neuron_types)
                ]
            )
            if self.layers[i].use_fb:
                fbs_stack.append(torch.stack(fbs[i]))  # type: ignore
            else:
                assert all(fb is None for fb in fbs[i])
                fbs_stack.append(None)
            if self.batch_first:
                outs_stack[i] = outs_stack[i].transpose(0, 1)
                for j in range(self.layers[i].num_neuron_types):
                    h_neurons_stack[i][j] = h_neurons_stack[i][j].transpose(
                        0, 1
                    )
                if self.layers[i].use_fb:
                    fbs_stack[i] = fbs_stack[i].transpose(0, 1)  # type: ignore

        return outs, h_neurons, fbs

    def _match_spatial_size(
        self,
        x: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Matches the spatial size of the input tensor to the spatial size of the layer.
        """
        if x.shape[-2:] > spatial_size:
            if self.pool_mode == "avg":
                return F.avg_pool2d(x, spatial_size)
            elif self.pool_mode == "max":
                return F.max_pool2d(x, spatial_size)
            else:
                raise ValueError(f"Invalid pool_mode: {self.pool_mode}")
        elif x.shape[-2:] < spatial_size:
            return F.interpolate(
                x, spatial_size, mode="bilinear", align_corners=False
            )
        else:
            return x

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        out_0: Optional[Sequence[Optional[torch.Tensor]]] = None,
        h_neuron_0: Optional[
            Sequence[Sequence[Optional[torch.Tensor]]]
        ] = None,
        fb_0: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> tuple[
        list[list[torch.Tensor]],
        list[list[list[torch.Tensor]]],
        list[list[torch.Tensor | None]],
    ]:
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

        outs, h_neurons, fbs = self._init_state(
            out_0,
            h_neuron_0,
            fb_0,
            num_steps,
            batch_size,
            device=device,
        )

        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                # Compute layer update and output
                if i == 0:
                    layer_in = x[t]
                    layer_in = F.interpolate(
                        layer_in,
                        self.layers[i].spatial_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    if self.layer_time_delay:
                        layer_in = outs[i - 1][t - 1]
                    else:
                        layer_in = outs[i - 1][t]
                    assert isinstance(layer_in, torch.Tensor)
                    layer_in = self._match_spatial_size(
                        layer_in, self.layers[i].spatial_size
                    )

                outs[i][t], h_neurons[i][t] = layer(
                    input=layer_in,
                    h_neuron=h_neurons[i][t - 1],
                    fb=fbs[i][t - 1],
                )

                # Apply feedback
                for key, conv in self.fb_convs.items():
                    fb_i, fb_j = key.split("->")
                    fb_i, fb_j = int(fb_i), int(fb_j)
                    fbs[fb_j][t] = fbs[fb_j][t] + conv(outs[fb_i][t])

        outs, h_neurons, fbs = self._format_result(outs, h_neurons, fbs)  # type: ignore

        return outs, h_neurons, fbs
