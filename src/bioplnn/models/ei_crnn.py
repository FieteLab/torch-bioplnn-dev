import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from bioplnn.typing import (
    ActivationFnType,
    Param1dType,
    Param2dType,
    TensorInitFnType,
)
from bioplnn.utils import (
    expand_array_2d,
    expand_list,
    get_activation,
    init_tensor,
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


@dataclass
class SpatiallyEmbeddedAreaConfig:
    """Configuration class for SpatiallyEmbeddedArea Module.

    Args:
        in_size (tuple[int, int]): Size of the input data (height, width).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_cell_types (int, optional): Number of neuron types. Defaults to 1.
        num_cell_subtypes (Param1dType[int], optional):
            Number of channels for each neuron type. Defaults to 16.
        cell_type_class (Param1dType[str], optional): Type of neuron. Defaults to
            "excitatory".
        cell_type_density (Param1dType[str], optional):
            Spatial mode for the neuron. Defaults to "same".
        feedback_channels (int, optional): Number of feedback channels. Defaults to
            None.
        inter_cell_type_connectivity (Param2dType[Union[int, bool]], optional):
            Connectivity matrix for the convolutions. Defaults to [[1, 0],
            [0, 1]].
        inter_cell_type_rectify (Param2dType[bool], optional): Whether to rectify the
            convolutions. Defaults to False.
        inter_cell_type_spatial_extents (Param2dType[tuple[int, int]], optional):
            Kernel size for the convolutions. Defaults to (3, 3).
        inter_cell_type_nonlinearity (Param2dType[Union[str, ActivationFnType]], optional):
            Activation function for the convolutions. Defaults to None.
        inter_cell_type_bias (Param2dType[bool], optional):
            Whether to add a bias term for the convolutions. Defaults to True.
        cell_nonlinearity (Param1dType[Union[str, ActivationFnType]], optional):
            Activation function for the post-aggregation. Defaults to "Tanh".
        tau_mode (Param1dType[str], optional): Mode for handling membrane time
            constants. Defaults to "channel".
        tau_init_fn (Param1dType[Union[str, TensorInitFnType]], optional):
            Initialization mode for the membrane time constants. Defaults to
            "ones".
        default_neuron_state_init_fn (Union[str, TensorInitFnType], optional):
            Initialization mode for the hidden state. Defaults to "zeros".
        default_feedback_state_init_fn (Union[str, TensorInitFnType], optional):
            Initialization mode for the feedback state. Defaults to "zeros".
        default_output_state_init_fn (Union[str, TensorInitFnType], optional):
            Initialization mode for the output state. Defaults to "zeros".
    """

    in_size: tuple[int, int]
    in_channels: int
    out_channels: int
    num_cell_types: int = 1
    num_cell_subtypes: Param1dType[int] = 16
    cell_type_class: Param1dType[str] = "hybrid"
    cell_type_density: Param1dType[str] = "same"
    feedback_channels: Optional[int] = None
    inter_cell_type_connectivity: Param2dType[Union[int, bool]] = field(
        default_factory=lambda: [[1, 0], [0, 1]]
    )
    inter_cell_type_rectify: Param2dType[bool] = False
    inter_cell_type_spatial_extents: Param2dType[Optional[tuple[int, int]]] = (
        3,
        3,
    )
    inter_cell_type_nonlinearity: Param2dType[
        Optional[Union[str, ActivationFnType]]
    ] = None
    inter_cell_type_bias: Param2dType[bool] = True
    cell_nonlinearity: Param1dType[Optional[Union[str, ActivationFnType]]] = (
        "Tanh"
    )
    tau_mode: Param1dType[Optional[str]] = "channel"
    tau_init_fn: Param1dType[Union[str, TensorInitFnType]] = "ones"
    default_neuron_state_init_fn: Union[str, TensorInitFnType] = "zeros"
    default_feedback_state_init_fn: Union[str, TensorInitFnType] = "zeros"
    default_output_state_init_fn: Union[str, TensorInitFnType] = "zeros"

    def asdict(self) -> dict[str, Any]:
        """Converts the configuration object to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the configuration.
        """
        return asdict(self)


class SpatiallyEmbeddedArea(nn.Module):
    """Implements a biologically-plausible, spatially embedded neural area.

    This module implements the core computational unit of the SpatiallyEmbeddedRNN.
    It supports multiple neuron types (excitatory and inhibitory) with configurable
    connectivity patterns, activation functions, and time constants.

    Args:
        config (Optional[SpatiallyEmbeddedAreaConfig]): Configuration object that
            specifies the area architecture and parameters. See
            SpatiallyEmbeddedAreaConfig for details. If None, parameters must be
            provided as keyword arguments.
        **kwargs: Keyword arguments that can be used to override or provide
            parameters not specified in the config object. These will be used to
            populate the config if one is not provided.

    The module implements:
    - Configurable neuron types (excitatory/inhibitory)
    - Convolutional connectivity between neuron populations
    - Recurrent dynamics with learnable time constants
    - Optional feedback connections
    - Customizable activation functions

    Raises:
        ValueError: If invalid configuration arguments are provided.
    """

    def __init__(
        self, config: Optional[SpatiallyEmbeddedAreaConfig] = None, **kwargs
    ):
        super().__init__()

        #####################################################################
        # Input, output, and feedback dimensions
        #####################################################################

        if config is None:
            config = SpatiallyEmbeddedAreaConfig(**kwargs)

        self.in_size = config.in_size
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.feedback_channels = (
            config.feedback_channels
            if config.feedback_channels is not None
            else 0
        )
        self.use_feedback = self.feedback_channels > 0

        #####################################################################
        # Neuron dimensions
        #####################################################################
        self.num_cell_types = config.num_cell_types

        # Format neuron type
        self.num_cell_subtypes = expand_list(
            config.num_cell_subtypes, self.num_cell_types, depth=1
        )
        self.cell_type_class = expand_list(
            config.cell_type_class, self.num_cell_types, depth=1
        )
        self.cell_type_density = expand_list(
            config.cell_type_density, self.num_cell_types, depth=1
        )

        if not set(self.cell_type_class) <= {
            "excitatory",
            "inhibitory",
            "hybrid",
        }:
            raise ValueError(
                "cell_type_class for each neuron type must be 'excitatory', "
                "'inhibitory', or 'hybrid'."
            )
        if not set(self.cell_type_density) <= {"same", "half"}:
            raise ValueError(
                "cell_type_density for each neuron type must be 'same' or 'half'."
            )

        # Save number of "types" for the input to and output from the area
        self.num_input_types = (
            1 + int(self.use_feedback) + self.num_cell_types
        )  # input + feedback + neurons
        self.num_output_types = self.num_cell_types + 1  # neurons + output

        # Calculate half spatial size
        self.half_size = (
            ceil(self.in_size[0] / 2),
            ceil(self.in_size[1] / 2),
        )

        # Calculate spatial size for each neuron type based on spatial mode
        self.neuron_size = [
            self.in_size if mode == "same" else self.half_size
            for mode in self.cell_type_density
        ]
        #####################################################################
        # Circuit motif connectivity
        #####################################################################

        # Format circuit connectivity
        self.inter_cell_type_connectivity = np.array(
            config.inter_cell_type_connectivity
        )
        if self.inter_cell_type_connectivity.shape != (
            self.num_output_types,
            self.num_input_types,
        ):
            raise ValueError(
                "The shape of inter_cell_type_connectivity must match the number of "
                "output and input types."
            )

        # Format connectivity variables to match circuit connectivity
        self.inter_cell_type_spatial_extents = expand_array_2d(
            config.inter_cell_type_spatial_extents,
            self.inter_cell_type_connectivity.shape[0],
            self.inter_cell_type_connectivity.shape[1],
            depth=2,
        )
        self.inter_cell_type_rectify = expand_array_2d(
            config.inter_cell_type_rectify,
            self.inter_cell_type_connectivity.shape[0],
            self.inter_cell_type_connectivity.shape[1],
            depth=1,
        )
        self.inter_cell_type_nonlinearity = expand_array_2d(
            config.inter_cell_type_nonlinearity,
            self.inter_cell_type_connectivity.shape[0],
            self.inter_cell_type_connectivity.shape[1],
            depth=1,
        )
        self.inter_cell_type_bias = expand_array_2d(
            config.inter_cell_type_bias,
            self.inter_cell_type_connectivity.shape[0],
            self.inter_cell_type_connectivity.shape[1],
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
        for i, row in enumerate(self.inter_cell_type_connectivity):
            # Handle input neuron channel and spatial mode based on neuron type
            conv_in_type = self.input_type_from_idx(i)
            if conv_in_type == "input":
                conv_in_channels = self.in_channels
                conv_in_density = "same"
            elif conv_in_type == "feedback":
                conv_in_channels = self.feedback_channels
                conv_in_density = "same"
            elif conv_in_type in ("excitatory", "inhibitory"):
                conv_in_channels = self.num_cell_subtypes[
                    i - 1 - int(self.use_feedback)
                ]
                conv_in_density = self.cell_type_density[
                    i - 1 - int(self.use_feedback)
                ]
            else:
                raise ValueError(f"Invalid neuron type: {conv_in_type}.")

            if conv_in_density not in ("same", "half"):
                raise ValueError(
                    f"Invalid neuron spatial mode: {conv_in_density}."
                )

            # Handle output neurons
            to_indices = np.nonzero(row)[0]
            for j in to_indices:
                if self.inter_cell_type_connectivity[i, j]:
                    # Handle rectification
                    if self.inter_cell_type_rectify[i, j]:
                        Conv2d = Conv2dRectify
                    else:
                        Conv2d = nn.Conv2d

                    # Handle output neuron channel and spatial mode based on neuron type
                    conv_out_type = self.output_type_from_idx(j)
                    if conv_out_type in ("excitatory", "inhibitory"):
                        conv_out_channels: int = self.num_cell_subtypes[j]  # type: ignore
                        conv_out_density: str = self.cell_type_density[j]  # type: ignore
                    else:
                        conv_out_channels = self.out_channels
                        conv_out_density = "same"

                    # Handle stride upsampling if necessary
                    conv = nn.Sequential()
                    conv_stride = 1
                    if (
                        conv_in_density == "half"
                        and conv_out_density == "same"
                    ):
                        conv_stride = 2
                    elif (
                        conv_in_density == "same"
                        and conv_out_density == "half"
                    ):
                        conv.append(
                            nn.Upsample(size=self.in_size, mode="bilinear")
                        )

                    # Handle upsampling if necessary
                    conv.append(
                        Conv2d(
                            in_channels=conv_in_channels,
                            out_channels=conv_out_channels,
                            kernel_size=self.inter_cell_type_spatial_extents[
                                i, j
                            ],
                            stride=conv_stride,
                            padding=(
                                self.inter_cell_type_spatial_extents[i, j][0]
                                // 2,
                                self.inter_cell_type_spatial_extents[i, j][1]
                                // 2,
                            ),
                            bias=self.inter_cell_type_bias[i, j],
                        )
                    )

                    self.convs[f"{i}->{j}"] = conv

        #####################################################################
        # Post convolution operations
        #####################################################################

        # Format post convolution activation functions
        self.cell_nonlinearity = expand_list(
            config.cell_nonlinearity, self.num_cell_types
        )

        # Initialize membrane time constants
        self.tau_mode = expand_list(config.tau_mode, self.num_cell_types)
        self.tau_init_fn = expand_list(config.tau_init_fn, self.num_cell_types)

        self.tau = nn.ParameterList()
        for i in range(self.num_cell_types):
            tau_channels = 1
            tau_size = (1, 1)
            if self.tau_mode[i] == "spatial":
                tau_size = self.neuron_size[i]
            elif self.tau_mode[i] == "channel":
                tau_channels = self.num_cell_subtypes[i]
            elif self.tau_mode[i] == "channel_spatial":
                tau_channels = self.num_cell_subtypes[i]
                tau_size = self.neuron_size[i]
            elif self.tau_mode[i] is None:
                pass
            else:
                raise ValueError(
                    f"Invalid tau_mode: {self.tau_mode[i]}. Must be 'spatial', "
                    "'channel', 'channel_spatial', or None."
                )

            self.tau.append(
                nn.Parameter(
                    init_tensor(
                        self.tau_init_fn[i],
                        1,
                        tau_channels,
                        *tau_size,
                    )
                )
            )

        #####################################################################
        # Tensor initialization
        #####################################################################

        self.default_neuron_state_init_fn = config.default_neuron_state_init_fn
        self.default_feedback_state_init_fn = (
            config.default_feedback_state_init_fn
        )
        self.default_output_state_init_fn = config.default_output_state_init_fn

    def input_type_from_idx(self, idx: int) -> Optional[str]:
        """Converts an input index to the corresponding neuron type.

        Args:
            idx (int): Input index in the circuit connectivity matrix.

        Returns:
            str: The neuron type associated with the index. Can be "input",
                "feedback", "excitatory", or "inhibitory".
        """
        if idx == 0:
            return "input"
        elif self.use_feedback and idx == 1:
            return "feedback"
        else:
            return self.cell_type_class[idx - 1 - int(self.use_feedback)]  # type: ignore

    def output_type_from_idx(self, idx: int) -> Optional[str]:
        """Converts an output index to the corresponding neuron type.

        Args:
            idx (int): Output index in the circuit connectivity matrix.

        Returns:
            str: The neuron type associated with the index. Can be "excitatory",
                "inhibitory", or "output".
        """
        if idx < self.num_output_types - 1:
            return self.cell_type_class[idx]  # type: ignore
        else:
            return "output"

    def init_neuron_state(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[torch.Tensor]:
        """Initializes the neuron hidden states.

        Args:
            batch_size (int): Batch size.
            init_fn (Union[str, TensorInitFnType], optional): Initialization mode.
                Must be 'zeros', 'ones', 'randn', 'rand', a function, or None.
                If None, the default initialization mode will be used. If a
                function, it must take a variable number of positional arguments
                corresponding to the shape of the tensor to initialize, as well
                as a `device` keyword argument that sends the device to allocate
                the tensor on.
            device (Union[torch.device, str], optional): Device to allocate the hidden
                states on. Defaults to None.

        Returns:
            list[torch.Tensor]: A list containing the initialized neuron
                hidden states.
        """

        init_fn_corrected = (
            init_fn
            if init_fn is not None
            else self.default_neuron_state_init_fn
        )

        return [
            init_tensor(
                init_fn_corrected,
                batch_size,
                self.num_cell_subtypes[i],
                *self.in_size,
                device=device,
            )
            for i in range(self.num_cell_types)
        ]

    def init_output_state(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """Initializes the output.

        Args:
            batch_size (int): Batch size.
            init_fn (Union[str, TensorInitFnType], optional): Initialization function.
                Must be 'zeros', 'ones', 'randn', 'rand', a function, or None. If
                None, the default initialization mode will be used.
            device (Union[torch.device, str], optional): Device to allocate the hidden
                states on. Defaults to None.

        Returns:
            torch.Tensor: The initialized output.
        """

        init_fn_corrected = (
            init_fn
            if init_fn is not None
            else self.default_output_state_init_fn
        )

        return init_tensor(
            init_fn_corrected,
            batch_size,
            self.out_channels,
            *self.in_size,
            device=device,
        )

    def init_feedback_state(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Union[torch.Tensor, None]:
        """Initializes the feedback input.

        Args:
            batch_size (int): Batch size.
            init_fn (Union[str, TensorInitFnType], optional): Initialization function.
                Must be 'zeros', 'ones', 'randn', 'rand', a function, or None. If
                None, the default initialization mode will be used.
            device (Union[torch.device, str], optional): Device to allocate the hidden
                states on. Defaults to None.

        Returns:
            Union[torch.Tensor, None]: The initialized feedback input if `use_feedback` is
                True, otherwise None.
        """

        if not self.use_feedback:
            return None

        init_fn_corrected = (
            init_fn
            if init_fn is not None
            else self.default_feedback_state_init_fn
        )

        return init_tensor(
            init_fn_corrected,
            batch_size,
            self.feedback_channels,
            *self.in_size,
            device=device,
        )

    def neuron_description_df(self) -> pd.DataFrame:
        """Creates a DataFrame representing the neuron types.

        Returns:
            pd.DataFrame: DataFrame with columns for neuron type, spatial mode, and
                number of channels.
        """

        df_columns = defaultdict(list)
        for i in range(self.num_cell_types):
            df_columns["type"].append(self.cell_type_class[i])
            df_columns["spatial_mode"].append(self.cell_type_density[i])
            df_columns["channels"].append(self.num_cell_subtypes[i])

        df = pd.DataFrame(df_columns)

        return df

    def conv_connectivity_df(self) -> pd.DataFrame:
        """Creates a DataFrame representing connectivity between neural populations.

        Returns:
            pd.DataFrame: DataFrame with rows representing source populations
                ("from") and columns representing target populations ("to").
        """
        row_labels = (
            ["input"]
            + (["feedback"] if self.use_feedback else [])
            + [f"neuron_{i}" for i in range(self.num_cell_types)]
        )
        column_labels = [f"neuron_{i}" for i in range(self.num_cell_types)] + [
            "output"
        ]

        assert len(row_labels) == self.num_input_types
        assert len(column_labels) == self.num_output_types

        array = np.empty((len(row_labels), len(column_labels)), dtype=object)
        for i in range(self.num_input_types):
            for j in range(self.num_output_types):
                if self.inter_cell_type_connectivity[i, j]:
                    content = []
                    content.append(
                        "r" if self.inter_cell_type_rectify[i, j] else "_"
                    )
                    content.append(
                        "b" if self.inter_cell_type_bias[i, j] else "_"
                    )
                    content.append(
                        self.inter_cell_type_nonlinearity[i, j][:2]
                        if self.inter_cell_type_nonlinearity[i, j]
                        else "_"
                    )
                    array[i, j] = ",".join(content)
                else:
                    array[i, j] = ""

        df = pd.DataFrame(
            array.tolist(), index=row_labels, columns=column_labels
        )
        df.index.name = "from"
        df.columns.name = "to"

        return df

    def __repr__(self) -> str:
        """Returns a string representation of the SpatiallyEmbeddedArea.

        Returns:
            str: String representation of the SpatiallyEmbeddedArea.
        """
        repr_str = "SpatiallyEmbeddedArea:\n"
        repr_str += "=" * 80 + "\n"
        repr_str += "Connectivity:\n"
        repr_str += self.conv_connectivity_df().to_string()
        repr_str += "-" * 80 + "\n"
        repr_str += "Neuron Description:\n"
        repr_str += self.neuron_description_df().to_string()
        repr_str += "=" * 80 + "\n"
        return repr_str

    def forward(
        self,
        input: torch.Tensor,
        neuron_state: Union[torch.Tensor, list[torch.Tensor]],
        feedback_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]]:
        """Forward pass of the SpatiallyEmbeddedArea.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels,
                in_size[0], in_size[1]).
            h_neuron (Union[torch.Tensor, list[torch.Tensor]]): List of neuron hidden
                states of shape (batch_size, neuron_channels[i], in_size[0],
                in_size[1]) for each neuron type i.
            feedback_state (Optional[torch.Tensor]): Feedback input of shape (batch_size,
                feedback_channels, in_size[0], in_size[1]).

        Returns:
            tuple[torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]]: A tuple
                containing the output and new neuron hidden state.
        """
        # Expand h_neuron to match the number of neuron channels
        if isinstance(neuron_state, torch.Tensor):
            if self.num_cell_types != 1:
                raise ValueError(
                    "neuron_state must be a list of tensors if num_cell_types is not 1."
                )
            neuron_state = [neuron_state]

        # Check if feedback is provided if necessary
        if self.use_feedback == feedback_state is None:
            raise ValueError(
                "use_feedback must be True if and only if feedback_state is provided."
            )

        # Compute convolutions for each connection in the circuit
        conv_ins = (
            [input]
            + ([feedback_state] if self.use_feedback else [])
            + neuron_state
        )
        conv_outs = [[] for _ in range(self.num_cell_types + 1)]
        for key, conv in self.convs.items():
            i, j = key.split("->")
            i, j = int(i), int(j)
            if self.input_type_from_idx(i) == "inhibitory":
                sign = -1
            else:
                sign = 1

            conv_outs[j].append(sign * conv(conv_ins[i]))

        conv_outs = [torch.cat(conv_out, dim=1) for conv_out in conv_outs]
        neuron_state_new = conv_outs[:-1]
        out = conv_outs[-1]

        # Compute Euler update for excitatory cell hidden state
        for i in range(self.num_cell_types):
            tau = torch.sigmoid(self.tau[i])
            neuron_state_new[i] = (
                tau * neuron_state_new[i] + (1 - tau) * neuron_state[i]
            )

        return out, neuron_state_new


class SpatiallyEmbeddedRNN(nn.Module):
    """T
    This module stacks multiple SpatiallyEmbeddedArea instances with optional
    feedback connections between them.
    It handles spatial dimension matching between areas and provides a
    flexible interface for configuring the network.

    Args:
        num_areas (int): Number of EIRNN areas in the network. Defaults to 1.
        area_configs (list[SpatiallyEmbeddedAreaConfig], optional): Configuration object(s)
            for the EIRNN areas. If provided as a list, must match the number of
            areas. If a single config is provided, it will be used for all areas
            with appropriate adjustments. Defaults to None.
        area_kwargs (list[Mapping[str, Any]], optional): Additional keyword arguments
            for each area. If provided as a list, must match the number of areas.
            Defaults to None.
        common_area_kwargs (Mapping[str, Any], optional): Keyword arguments to apply
            to all areas. Defaults to None.
        inter_area_feedback_connectivity (Param2dType[Union[int, bool]], optional): Connectivity matrix
            for feedback connections between areas. Defaults to None.
        inter_area_feedback_nonlinearity (Param2dType[str], optional): Activation functions for
            feedback connections. Defaults to None.
        inter_area_feedback_rectify (Param2dType[bool]): Whether to rectify feedback connections.
            Defaults to False.
        inter_area_feedback_spatial_extents (Param2dType[tuple[int, int]], optional): Kernel sizes for
            feedback convolutions. Defaults to None.
        pool_mode (str, optional): Pooling mode for area outputs. Defaults to "avg".
        area_time_delay (bool): Whether to introduce a time delay between areas.
            Defaults to False.
        batch_first (bool): Whether the input tensor has batch dimension as the first
            dimension. Defaults to True.
    """

    def __init__(
        self,
        *,
        num_areas: int = 1,
        area_configs: Optional[list[SpatiallyEmbeddedAreaConfig]] = None,
        area_kwargs: Optional[list[Mapping[str, Any]]] = None,
        common_area_kwargs: Optional[Mapping[str, Any]] = None,
        inter_area_feedback_connectivity: Optional[
            Param2dType[Union[int, bool]]
        ] = None,
        inter_area_feedback_nonlinearity: Optional[Param2dType[str]] = None,
        inter_area_feedback_rectify: Param2dType[bool] = False,
        inter_area_feedback_spatial_extents: Optional[
            Param2dType[tuple[int, int]]
        ] = None,
        area_time_delay: bool = False,
        pool_mode: Optional[str] = "max",
        batch_first: bool = True,
    ):
        super().__init__()

        ############################################################
        # area configs
        ############################################################

        self.num_areas = num_areas

        if area_configs is not None:
            if area_kwargs is not None or common_area_kwargs is not None:
                raise ValueError(
                    "area_configs cannot be provided if area_configs_kwargs "
                    "or common_area_config_kwargs is provided."
                )
            if len(area_configs) != self.num_areas:
                raise ValueError("area_configs must be of length num_areas.")
        else:
            if area_kwargs is None:
                if common_area_kwargs is None:
                    raise ValueError(
                        "area_kwargs or common_area_kwargs must be provided if "
                        "area_configs is not provided."
                    )
                area_kwargs = [{}] * self.num_areas  # type: ignore
            elif len(area_kwargs) != self.num_areas:
                raise ValueError("area_kwargs must be of length num_areas.")

            if common_area_kwargs is None:
                common_area_kwargs = {}

            area_configs = [
                SpatiallyEmbeddedAreaConfig(
                    **common_area_kwargs,
                    **area_kwargs[i],  # type: ignore
                )
                for i in range(self.num_areas)
            ]

        # Validate area configurations
        for i in range(self.num_areas - 1):
            if area_configs[i].out_channels != area_configs[i + 1].in_channels:
                raise ValueError(
                    f"The output channels of area {i} must match the input "
                    f"channels of area {i + 1}."
                )

        # Check if the first area input projections are rectified
        warn = False
        if isinstance(area_configs[0].inter_cell_type_rectify, bool):
            if area_configs[0].inter_cell_type_rectify:
                warn = True
        else:
            if any(area_configs[0].inter_cell_type_rectify):
                warn = True
        if warn:
            warnings.warn(
                "Rectification of network input may hinder learning! "
                "Consider setting inter_cell_type_rectify to False for the input "
                "projections to the first area or for the whole first "
                "area (i.e. set area_configs[0].inter_cell_type_rectify[0,:] or "
                "area_configs[0].inter_cell_type_rectify to False). \n"
                "If your input to the network is already purely positive, "
                "you can safely ignore this warning."
            )

        ############################################################
        # RNN parameters
        ############################################################

        self.area_time_delay = area_time_delay
        self.pool_mode = pool_mode
        self.batch_first = batch_first

        ############################################################
        # Initialize areas
        ############################################################

        # Create areas
        self.areas = nn.ModuleList(
            [
                SpatiallyEmbeddedArea(area_config)
                for area_config in area_configs
            ]
        )

        ############################################################
        # Initialize feedback connections
        ############################################################

        if inter_area_feedback_connectivity is None:
            if any(
                area_config.feedback_channels for area_config in area_configs
            ):
                raise ValueError(
                    "inter_area_feedback_connectivity must be provided if and only if "
                    "feedback_channels is provided for at least one area."
                )
            self.feedback_convs = nn.ModuleDict()
        else:
            inter_area_feedback_connectivity = np.array(
                inter_area_feedback_connectivity, dtype=bool
            )
            if inter_area_feedback_connectivity.shape != (
                self.num_areas,
                self.num_areas,
            ):
                raise ValueError(
                    "The shape of inter_area_feedback_connectivity must be (num_areas, num_areas)."
                )
            inter_area_feedback_nonlinearity = expand_array_2d(
                inter_area_feedback_nonlinearity,
                self.num_areas,
                self.num_areas,
            )
            inter_area_feedback_rectify = expand_array_2d(
                inter_area_feedback_rectify,
                self.num_areas,
                self.num_areas,
            )
            inter_area_feedback_spatial_extents = expand_array_2d(
                inter_area_feedback_spatial_extents,
                self.num_areas,
                self.num_areas,
            )

            # Validate inter_area_feedback_connectivity tensor
            if (
                inter_area_feedback_connectivity.ndim != 2
                or inter_area_feedback_connectivity.shape[0] != self.num_areas
                or inter_area_feedback_connectivity.shape[1] != self.num_areas
            ):
                raise ValueError(
                    "The dimensions of inter_area_feedback_connectivity must match the number of areas."
                )
            if inter_area_feedback_connectivity.sum() == 0:
                raise ValueError(
                    "inter_area_feedback_connectivity must be a non-zero tensor if provided."
                )

            # Create feedback convolutions
            self.feedback_convs = nn.ModuleDict()
            for i, row in enumerate(inter_area_feedback_connectivity):
                nonzero_indices = np.nonzero(row)[0]
                for j in nonzero_indices:
                    if not self.areas[j].use_feedback:
                        raise ValueError(
                            f"the connection from area {i} to area {j} is "
                            f"not valid because area {j} does not receive "
                            f"feedback (hint: feedback_channels may not be provided)"
                        )
                    if inter_area_feedback_rectify[i, j]:
                        Conv2dFeedback = Conv2dRectify
                    else:
                        Conv2dFeedback = nn.Conv2d
                    self.feedback_convs[f"{i}->{j}"] = nn.Sequential(
                        nn.Upsample(
                            size=area_configs[j].in_size,
                            mode="bilinear",
                        ),
                        Conv2dFeedback(
                            in_channels=area_configs[i].out_channels,
                            out_channels=area_configs[j].feedback_channels,
                            kernel_size=area_configs[
                                j
                            ].inter_area_feedback_spatial_extents[i, j],
                            padding=(
                                inter_area_feedback_spatial_extents[i, j][0]
                                // 2,
                                inter_area_feedback_spatial_extents[i, j][1]
                                // 2,
                            ),
                            bias=area_configs[
                                j
                            ].default_feedback_state_init_fn,
                        ),
                        get_activation(inter_area_feedback_nonlinearity[i, j]),
                    )

    def init_hidden(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[list[torch.Tensor]]:
        """Initializes the hidden states for all areas.

        Args:
            batch_size (int): Batch size.
            init_fn (Union[str, TensorInitFnType], optional): Initialization
                function.
            device (Union[torch.device, str], optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list containing the initialized neuron hidden
                states for each area.
        """

        return [
            area.init_hidden(batch_size, init_fn, device)
            for area in self.areas
        ]

    def init_fb(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[Optional[torch.Tensor]]:
        """Initializes the feedback inputs for all areas.

        Args:
            batch_size (int): Batch size.
            init_fn (Union[str, TensorInitFnType], optional): Initialization
                function.
            device (Union[torch.device, str], optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list of initialized feedback inputs for each
                area.
        """
        return [
            area.init_fb(batch_size, init_fn, device) for area in self.areas
        ]

    def init_out(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[torch.Tensor]:
        """Initializes the outputs for all areas.

        Args:
            batch_size (int): Batch size.
            init_fn (Union[str, TensorInitFnType], optional): Initialization
                function.
            device (Union[torch.device, str], optional): Device to allocate tensors.

        Returns:
            list[torch.Tensor]: A list of initialized outputs for each
                area.
        """

        return [
            area.init_out(batch_size, init_fn, device) for area in self.areas
        ]

    def init_state(
        self,
        out0: Optional[Sequence[Union[torch.Tensor, None]]],
        h_neuron0: Optional[Sequence[Sequence[Union[torch.Tensor, None]]]],
        fb0: Optional[Sequence[Union[torch.Tensor, None]]],
        num_steps: int,
        batch_size: int,
        out_init_fn: Optional[Union[str, TensorInitFnType]] = None,
        hidden_init_fn: Optional[Union[str, TensorInitFnType]] = None,
        fb_init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> tuple[
        list[list[Union[torch.Tensor, None]]],
        list[list[list[Union[torch.Tensor, None]]]],
        list[list[Union[torch.Tensor, int, None]]],
    ]:
        """Initializes the state of the network.

        Args:
            out0 (Optional[Sequence[Union[torch.Tensor, None]]]): Initial outputs for
                each area. If None, default initialization is used.
            h_neuron0 (Optional[Sequence[Sequence[Union[torch.Tensor, None]]]]):
                Initial neuron hidden states for each area. If None, default
                initialization is used.
            fb0 (Optional[Sequence[Union[torch.Tensor, None]]]): Initial feedback
                inputs for each area. If None, default initialization is used.
            num_steps (int): Number of time steps.
            batch_size (int): Batch size.
            out_init_fn (Optional[Union[str, TensorInitFnType]]): Initialization
                function for outputs if out0 is None. Defaults to None.
            hidden_init_fn (Optional[Union[str, TensorInitFnType]]): Initialization
                function for hidden states if h_neuron0 is None. Defaults to None.
            fb_init_fn (Optional[Union[str, TensorInitFnType]]): Initialization
                function for feedback inputs if fb0 is None. Defaults to None.
            device (Optional[Union[torch.device, str]]): Device to allocate tensors on.
                Defaults to None.

        Returns:
            tuple[list[list[Union[torch.Tensor, None]]],
                  list[list[list[Union[torch.Tensor, None]]],
                  list[list[Union[torch.Tensor, int, None]]]:
                A tuple containing:
                - Initialized outputs for each area and time step
                - Initialized neuron hidden states for each area, time step, and
                  neuron type
                - Initialized feedback inputs for each area and time step

        Raises:
            ValueError: If the length of out0, h_neuron0, or fb0 doesn't match
                the number of areas.
        """

        # Validate input shapes of out0, h_neuron0, fb0
        if out0 is None:
            out0 = [None] * self.num_areas
        elif len(out0) != self.num_areas:
            raise ValueError(
                "The length of out0 must be equal to the number of areas."
            )
        if h_neuron0 is None:
            h_neuron0 = [
                [None] * self.areas[i].num_cell_types
                for i in range(self.num_areas)
            ]
        elif len(h_neuron0) != self.num_areas:
            raise ValueError(
                "The length of h_neuron0 must be equal to the number of areas."
            )
        if fb0 is None:
            fb0 = [None] * self.num_areas
        elif len(fb0) != self.num_areas:
            raise ValueError(
                "The length of fb0 must be equal to the number of areas."
            )

        # Initialize default values
        h_neuron0_default = self.init_hidden(
            batch_size, hidden_init_fn, device
        )
        fb0_default = self.init_fb(batch_size, fb_init_fn, device)
        out0_default = self.init_out(batch_size, out_init_fn, device)

        # Initialize output, hidden state, and feedback lists
        outs: list[list[Union[torch.Tensor, None]]] = [
            [None] * num_steps for _ in range(self.num_areas)
        ]
        h_neurons: list[list[list[Union[torch.Tensor, None]]]] = [
            [[None] * self.areas[i].num_cell_types for _ in range(num_steps)]
            for i in range(self.num_areas)
        ]
        fbs: list[list[Union[torch.Tensor, int, None]]] = [
            [0 if self.areas[i].use_fb else None] * num_steps
            for i in range(self.num_areas)
        ]

        # Fill time step -1 with the initial values
        for i in range(self.num_areas):
            outs[i][-1] = out0[i] if out0[i] is not None else out0_default[i]
            fbs[i][-1] = fb0[i] if fb0[i] is not None else fb0_default[i]
            for k in range(self.areas[i].num_cell_types):
                h_neurons[i][-1][k] = (
                    h_neuron0[i][k]
                    if h_neuron0[i][k] is not None
                    else h_neuron0_default[i][k]
                )

        return outs, h_neurons, fbs

    def _format_x(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, int]:
        """Formats the input tensor to match the expected shape.

        This method handles both single-step (4D) and multi-step (5D) input tensors,
        converting them to a consistent format with seq_len as the first dimension.
        For 4D inputs, it replicates the input across all time steps.

        Args:
            x (torch.Tensor): Input tensor, can be:
                - 4D tensor of shape (batch_size, channels, height, width) for a
                  single time step. Will be expanded to all time steps.
                - 5D tensor of shape (seq_len, batch_size, channels, height, width) or
                  (batch_size, seq_len, channels, height, width) if batch_first=True.
            num_steps (Optional[int]): Number of time steps. Required if x is 4D.
                If x is 5D, it will be inferred from the sequence dimension unless
                explicitly provided.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing:
                - The formatted input tensor with shape (seq_len, batch_size,
                  channels, height, width)
                - The number of time steps

        Raises:
            ValueError: If x has invalid dimensions (not 4D or 5D), if num_steps is
                not provided for 4D inputs, or if num_steps is inconsistent with
                the sequence length of 5D inputs.
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
        fbs: list[list[Optional[torch.Tensor]]],
    ) -> tuple[
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[Optional[torch.Tensor]],
    ]:
        """Formats the outputs, hidden states, and feedback inputs for return.

        This method stacks tensors across time steps and applies batch_first
        transposition if needed. Each tensor has shape (batch_size, channels,
        height, width).

        Args:
            outs (list[list[torch.Tensor]]): Outputs for each area and time
                step. Shape: [num_areas][num_steps],
            h_neurons (list[list[list[torch.Tensor]]]): Neuron hidden states for
                each area, time step, and neuron type. Shape:
                [num_areas][num_steps][num_cell_types].
            fbs (list[list[Optional[torch.Tensor]]]): Feedback inputs for each area
                and time step. Shape: [num_areas][num_steps].

        Returns:
            tuple[list[torch.Tensor], list[list[torch.Tensor]], list[Optional[torch.Tensor]]]:
                A tuple containing:
                - List of stacked outputs per area. Each tensor has shape:
                  (seq_len, batch_size, channels, height, width) or
                  (batch_size, seq_len, channels, height, width) if batch_first=True.
                - List of lists of stacked hidden states per area and neuron type.
                  Same shape pattern as outputs.
                - List of stacked feedback inputs per area (or None if not used).
                  Same shape pattern as outputs.
        """
        outs_stack: list[torch.Tensor] = []
        h_neurons_stack: list[list[torch.Tensor]] = []
        fbs_stack: Optional[list[Optional[torch.Tensor]]] = []

        for i in range(self.num_areas):
            outs_stack.append(torch.stack(outs[i]))
            h_neurons_stack.append(
                [
                    torch.stack(
                        [h_neurons[i][t][j] for t in range(len(h_neurons[i]))]
                    )
                    for j in range(self.areas[i].num_cell_types)
                ]
            )
            if self.areas[i].use_fb:
                fbs_stack.append(torch.stack(fbs[i]))  # type: ignore
            else:
                assert all(feedback_state is None for feedback_state in fbs[i])
                fbs_stack.append(None)
            if self.batch_first:
                outs_stack[i] = outs_stack[i].transpose(0, 1)
                for j in range(self.areas[i].num_cell_types):
                    h_neurons_stack[i][j] = h_neurons_stack[i][j].transpose(
                        0, 1
                    )
                if self.areas[i].use_fb:
                    fbs_stack[i] = fbs_stack[i].transpose(0, 1)  # type: ignore

        return outs_stack, h_neurons_stack, fbs_stack

    def _match_spatial_size(
        self,
        x: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        """Adjusts the spatial dimensions of the input tensor.

        This method ensures that tensors have compatible spatial dimensions when
        passing between areas. It uses either pooling (when downsampling) or
        interpolation (when upsampling).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            spatial_size (tuple[int, int]): Target spatial size (height, width).

        Returns:
            torch.Tensor: Resized tensor matching the target spatial size.

        Raises:
            ValueError: If self.pool_mode is not 'avg' or 'max'.
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
        output_state0: Optional[Sequence[Optional[torch.Tensor]]] = None,
        neuron_state0: Optional[
            Sequence[Sequence[Optional[torch.Tensor]]]
        ] = None,
        feedback_state0: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> tuple[
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[Union[torch.Tensor, None]],
    ]:
        """Performs forward pass of the SpatiallyEmbeddedRNN.

        Args:
            x (torch.Tensor): Input tensor. Can be either:
                - 4D tensor of shape (batch_size, in_channels, height, width)
                  representing a single time step. In this case, num_steps must be
                  provided.
                - 5D tensor of shape (seq_len, batch_size, in_channels, height, width)
                  or (batch_size, seq_len, in_channels, height, width) if
                  batch_first=True.
            num_steps (Optional[int]): Number of time steps. Required if x is 4D.
                If x is 5D, this must match the sequence length dimension in x.
            output_state0 (Optional[Sequence[Optional[torch.Tensor]]]): Initial outputs for
                each area. Length should match the number of areas. Each element
                can be None to use default initialization.
            neuron_state0 (Optional[Sequence[Sequence[Optional[torch.Tensor]]]]):
                Initial neuron hidden states for each area and neuron type. Length
                should match the number of areas, and each inner sequence length
                should match the number of neuron types in that area.
            feedback_state0 (Optional[Sequence[Optional[torch.Tensor]]]): Initial feedback inputs
                for each area. Length should match the number of areas.

        Returns:
            tuple[list[torch.Tensor], list[list[torch.Tensor]], list[Union[torch.Tensor, None]]]:
                A tuple containing:
                - list[torch.Tensor]: Outputs for each area. Each tensor has shape:
                  (seq_len, batch_size, out_channels, height, width) or
                  (batch_size, seq_len, out_channels, height, width) if batch_first=True.
                - list[list[torch.Tensor]]: Hidden states for each area and neuron type.
                  Same shape pattern as outputs but with neuron_channels.
                - list[Union[torch.Tensor, None]]: Feedback inputs for each area (None if the
                  area doesn't use feedback).

        Raises:
            ValueError: If input shape is invalid or num_steps is inconsistent with
                the input shape.
        """

        device = x.device

        x, num_steps = self._format_x(x, num_steps)

        batch_size = x.shape[1]

        output_states, neuron_states, feedback_states = self.init_state(
            output_state0,
            neuron_state0,
            feedback_state0,
            num_steps,
            batch_size,
            device=device,
        )

        for t in range(num_steps):
            for i, area in enumerate(self.areas):
                # Compute area update and output
                if i == 0:
                    area_in = x[t]
                    area_in = F.interpolate(
                        area_in,
                        self.areas[i].neuron_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    if self.area_time_delay:
                        area_in = output_states[i - 1][t - 1]
                    else:
                        area_in = output_states[i - 1][t]
                    assert isinstance(area_in, torch.Tensor)
                    area_in = self._match_spatial_size(
                        area_in, self.areas[i].neuron_size
                    )

                output_states[i][t], neuron_states[i][t] = area(
                    input=area_in,
                    neuron_state=neuron_states[i][t - 1],
                    feedback_state=feedback_states[i][t - 1],
                )

            # Apply feedback
            for key, conv in self.feedback_convs.items():
                area_i, area_j = key.split("->")
                area_i, area_j = int(area_i), int(area_j)
                feedback_states[area_j][t] = feedback_states[area_j][t] + conv(
                    output_states[area_i][t]
                )

        output_states, neuron_states, feedback_states = self._format_result(
            output_states,  # type: ignore
            neuron_states,  # type: ignore
            feedback_states,  # type: ignore
        )

        return output_states, neuron_states, feedback_states
