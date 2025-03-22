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
    Array2dType,
    InterAreaParam,
    ScalarOrArray2dType,
    ScalarOrListLike,
    TensorInitFnType,
)
from bioplnn.utils import (
    check_possible_values,
    expand_array_2d,
    expand_list,
    get_activation,
    init_tensor,
)

# TODO: Some docstrings may be outdated, might need to update


class Conv2dRectify(nn.Conv2d):
    """Applies a 2d convolution with nonnegative weights and biases."""

    def forward(self, *args, **kwargs):
        """Forward pass of the layer.

        Args:
            *args: Positional arguments passed to `nn.Conv2d`.
            **kwargs: Keyword arguments passed to `nn.Conv2d`.

        Returns:
            The convolution output.
        """
        self.weight.data.clamp_(min=0.0)
        if self.bias is not None:
            self.bias.data.clamp_(min=0.0)
        return super().forward(*args, **kwargs)


@dataclass
class SpatiallyEmbeddedAreaConfig:
    """Configuration for `SpatiallyEmbeddedArea`.

    This class defines the configuration for a spatially embedded area. It
    specifies the size of the input data, the number of input and output
    channels, the connectivity matrix for the circuit motif, and the parameters
    for the neuron types.

    The default configuration corresponds to a spatially embedded area with
    one excitatory neuron type which is stimulated by the input and by itself
    (lateral connections).

    Any of the parameters annotated with `ScalarOrListLike` can be either a
    single value that applies to all neuron types, or a list of values that
    apply to each neuron type.

    Any of the parameters annotated with `ScalarOrArray2dType` can be either a
    single value that applies to all connections in the circuit motif, or a
    2D array that applies to each connection in the circuit motif.

    Attributes:
        in_size: Spatial size of the input data (height, width).
            This size determines the spatial sizes of the feedback signal, the
            neuronal states, and the output.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        feedback_channels: Number of feedback channels. If provided, this area
            must receive feedback from another area.
        in_class: Class of input signal. Can be
            "excitatory", "inhibitory", or "hybrid".
        feedback_class: Class of feedback signal. Can be
            "excitatory", "inhibitory", or "hybrid".
        num_neuron_types: Number of neuron types.
        num_neuron_subtypes: Number of subtypes for each neuron type.
        neuron_type_class: Class of neuron type. Can be
            "excitatory", "inhibitory", or "hybrid".
        neuron_type_density: Spatial density of each neuron type. Can be "same"
            or "half".
        neuron_type_nonlinearity: Nonlinearity to apply to each neuron type's
            activity after adding the impact of all connected inputs/neuron
            types in the circuit motif.
        inter_neuron_type_connectivity: Connectivity matrix for the circuit motif.
            The shape should be
            (1 + int(use_feedback) + num_neuron_types, num_neuron_types + 1). Here,
            rows represent source types and columns represent destination types.
            A True entry in the matrix indicates a connection from the source to
            the destination.
            The first row corresponds to the input, the second row corresponds to
            the feedback (if feedback_channels > 0), and the remaining rows
            correspond to the neuron types.
            The first num_neuron_types columns correspond to the neuron types
            and the last column corresponds to the output. To get a template of
            the connectivity matrix for your configuration with appropriate row
            and column labels, use the `inter_neuron_type_connectivity_template_df`
            method of this class.
        inter_neuron_type_spatial_extents: Spatial extent for each circuit
            motif connection. Same shape as `inter_neuron_type_connectivity`.
        inter_neuron_type_num_subtype_groups: Number of subtype groups for each
            circuit motif connection. Same shape as `inter_neuron_type_connectivity`.
        inter_neuron_type_nonlinearity: Nonlinearity for each circuit motif
            connection. Same shape as `inter_neuron_type_connectivity`.
        inter_neuron_type_bias: Whether to add a bias term for each circuit
            motif connection. Same shape as `inter_neuron_type_connectivity`.
        tau_mode: Mode determining which parts of
            neuron activity share a time constant. Can be "type" (one tau for each neuron type),
            "subtype" (one tau per neuron subtype), "spatial" (one tau per spatial location), or
            "subtype_spatial" (one tau per neuron subtype and spatial location)
        tau_init_fn: Initialization mode for the membrane time constants.
        out_nonlinearity: Nonlinearity to apply to the output.
        default_neuron_state_init_fn: Initialization mode for the hidden state.
        default_feedback_state_init_fn: Initialization mode for the feedback state.
        default_output_state_init_fn: Initialization mode for the output state.

    Examples:
        >>> connectivity_df = SpatiallyEmbeddedAreaConfig.inter_neuron_type_connectivity_template_df(
        ...     use_feedback=False,
        ...     num_neuron_types=2,
        ... )
        >>> print(connectivity_df)
                        destination
        source          neuron_0  neuron_1  output
        input           False     False     False
        neuron_0        False     False     False
        neuron_1        False     False     False
        >>> connectivity_df.loc["input", "neuron_0"] = True
        >>> connectivity_df.loc["neuron_0", "neuron_0"] = True
        >>> connectivity_df.loc["neuron_0", "neuron_1"] = True
        >>> connectivity_df.loc["neuron_1", "neuron_0"] = True
        >>> connectivity_df.loc["neuron_0", "output"] = True
        >>> config = SpatiallyEmbeddedAreaConfig(
        ...     in_size=(32, 32),
        ...     in_channels=3,
        ...     out_channels=16,
        ...     inter_neuron_type_connectivity=connectivity_df.to_numpy(),
        ... )
    """

    # Input, output, and feedback parameters
    in_size: tuple[int, int]
    in_channels: int
    out_channels: int
    feedback_channels: Optional[int] = None
    in_class: str = "hybrid"
    feedback_class: str = "hybrid"

    # Neuron type parameters
    num_neuron_types: int = 1
    num_neuron_subtypes: ScalarOrListLike[int] = 16
    neuron_type_class: ScalarOrListLike[str] = "hybrid"
    neuron_type_density: ScalarOrListLike[str] = "same"
    neuron_type_nonlinearity: ScalarOrListLike[
        Optional[Union[str, nn.Module]]
    ] = "Sigmoid"
    tau_mode: ScalarOrListLike[str] = "subtype"
    tau_init_fn: ScalarOrListLike[Union[str, TensorInitFnType]] = "ones"

    # Circuit motif connectivity parameters
    inter_neuron_type_connectivity: Array2dType[Union[int, bool]] = field(
        default_factory=lambda: [[1, 0], [1, 1]]
    )
    inter_neuron_type_spatial_extents: ScalarOrArray2dType[tuple[int, int]] = (
        3,
        3,
    )
    inter_neuron_type_num_subtype_groups: ScalarOrArray2dType[int] = 1
    inter_neuron_type_nonlinearity: ScalarOrArray2dType[
        Optional[Union[str, nn.Module]]
    ] = None
    inter_neuron_type_bias: ScalarOrArray2dType[bool] = True
    out_nonlinearity: Optional[Union[str, nn.Module]] = None
    default_neuron_state_init_fn: Union[str, TensorInitFnType] = "zeros"
    default_feedback_state_init_fn: Union[str, TensorInitFnType] = "zeros"
    default_output_state_init_fn: Union[str, TensorInitFnType] = "zeros"

    def asdict(self) -> dict[str, Any]:
        """Converts the configuration object to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the configuration.
        """
        return asdict(self)

    @staticmethod
    def inter_neuron_type_connectivity_template_df(
        use_feedback: bool, num_neuron_types: int
    ) -> pd.DataFrame:
        """Samples the inter-neuron type connectivity matrix.

        Returns:
            pd.DataFrame: DataFrame representation of the connectivity matrix.
        """
        row_labels = (
            ["input"]
            + (["feedback"] if use_feedback else [])
            + [f"neuron_{i}" for i in range(num_neuron_types)]
        )
        column_labels = [f"neuron_{i}" for i in range(num_neuron_types)] + [
            "output"
        ]

        return pd.DataFrame(
            np.zeros((len(row_labels), len(column_labels)), dtype=np.bool),
            index=row_labels,
            columns=column_labels,
        )


class SpatiallyEmbeddedArea(nn.Module):
    """A biologically-plausible, spatially embedded neural area.

    This module imposes a series of biologically-inspired constraints on
    artificial neural networks. At its core, it is a collection of 2D (hence
    spatially embedded) convolutional layers organized into a 'circuit
    motif'. Here, 'circuit motif' refers to the connectivity pattern between
    the input, feedback, neuron types, and output within a distinct neural
    area. For example, if we have two neuron types, an excitatory and an
    inhibitory, then the circuit motif determines which neuron types receive
    input, which neuron types receive feedback, which neuron types are connected
    to which other neuron types, and which neuron types project to the output of
    the area.

    ## Key features:

    - Configurable neuron types (excitatory/inhibitory/hybrid)
    - Configurable spatial extents of lateral connections (same/half)
    - Convolutional connectivity between neuron populations
    - Recurrent dynamics with learnable time constants
    - Optional feedback connections
    - Customizable activation functions

    Attributes:
        in_size (tuple[int, int]): Spatial size of the input data
            (height, width).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        feedback_channels (int): Number of feedback channels (0 if none).
        use_feedback (bool): Whether this area receives feedback from another
            area.
        in_class (str): Class of input signal ("excitatory", "inhibitory", or
            "hybrid").
        feedback_class (str): Class of feedback signal ("excitatory", "inhibitory",
            or "hybrid").
        out_nonlinearity (nn.Module): Nonlinearity applied to the output.

        num_neuron_types (int): Number of neuron types in the area.
        num_neuron_subtypes (list[int]): Number of subtypes for each neuron type.
        neuron_type_class (list[str]): Class of each neuron type ("excitatory",
            "inhibitory", or "hybrid").
        neuron_type_density (list[str]): Spatial density of each neuron type
            ("same" or "half").
        neuron_type_nonlinearity (nn.ModuleList): Nonlinearity for each neuron
            type's activity.
        neuron_type_size (list[tuple[int, int]]): Spatial size of each neuron type.

        num_rows_connectivity (int): Number of rows in the connectivity matrix.
        num_cols_connectivity (int): Number of columns in the connectivity matrix.
        inter_neuron_type_connectivity (np.ndarray): Connectivity matrix for the
            circuit motif.
        inter_neuron_type_spatial_extents (np.ndarray): Spatial extent for each
            circuit motif connection.
        inter_neuron_type_num_subtype_groups (np.ndarray): Number of subtype groups
            for each circuit motif connection.
        inter_neuron_type_nonlinearity (np.ndarray): Nonlinearity for each circuit
            motif connection.
        inter_neuron_type_bias (np.ndarray): Whether to add a bias term for each
            circuit motif connection.

        tau_mode (list[str]): Mode determining which parts of neuron activity share
            a time constant.
        tau_init_fn (list[Union[str, TensorInitFnType]]): Initialization for the
            membrane time constants.
        tau (nn.ParameterList): Learnable time constants for each neuron type.

        default_neuron_state_init_fn (Union[str, TensorInitFnType]): Default
            initialization for neuron states.
        default_feedback_state_init_fn (Union[str, TensorInitFnType]): Default
            initialization for feedback states.
        default_output_state_init_fn (Union[str, TensorInitFnType]): Default
            initialization for output states.

        convs (nn.ModuleDict): Convolutional layers representing connections between
            neuron types.
        out_convs (nn.ModuleDict): Convolutional layers connecting to the output.

    Examples:
        >>> config = SpatiallyEmbeddedAreaConfig(
        ...     in_size=(32, 32),
        ...     in_channels=3,
        ...     out_channels=16,
        ...     num_neuron_types=2,
        ...     num_neuron_subtypes=16,
        ...     neuron_type_class=["excitatory", "inhibitory"],
        ...     neuron_type_density=["same", "half"],
        ...     neuron_type_nonlinearity=,
        ...     inter_neuron_type_connectivity=[[1, 0, 0], [1, 1, 1], [1, 0, 0]],
        ...     inter_neuron_type_spatial_extents=(3, 3),
        ... )
        >>> area = SpatiallyEmbeddedArea(config)
        >>> print(area.summary())
    """

    def __init__(
        self, config: Optional[SpatiallyEmbeddedAreaConfig] = None, **kwargs
    ):
        """Initialize the SpatiallyEmbeddedArea.

        Args:
            config: Configuration object that specifies the area architecture and parameters.
                See SpatiallyEmbeddedAreaConfig for details. If None, parameters must be
                provided as keyword arguments.
            **kwargs: Keyword arguments to instantiate the configuration if
                `config` is not provided. Cannot provide both `config` and
                keyword arguments.

        Raises:
            ValueError: If an invalid configuration is provided.
        """

        super().__init__()

        if config is None:
            config = SpatiallyEmbeddedAreaConfig(**kwargs)
        elif kwargs:
            raise ValueError(
                "Cannot provide both config and keyword arguments. Please provide "
                "only one of the two."
            )

        #####################################################################
        # Input, output, and feedback parameters
        #####################################################################

        self.in_size = config.in_size
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.feedback_channels = (
            config.feedback_channels
            if config.feedback_channels is not None
            else 0
        )
        self.use_feedback = self.feedback_channels > 0

        self.in_class = config.in_class
        check_possible_values(
            "in_class",
            (self.in_class,),
            ("excitatory", "inhibitory", "hybrid"),
        )
        self.feedback_class = config.feedback_class
        check_possible_values(
            "feedback_class",
            (self.feedback_class,),
            ("excitatory", "inhibitory", "hybrid"),
        )

        self.out_nonlinearity = get_activation(config.out_nonlinearity)

        #####################################################################
        # Neuron type parameters
        #####################################################################

        self.num_neuron_types = config.num_neuron_types

        # Format neuron type
        self.num_neuron_subtypes = expand_list(
            config.num_neuron_subtypes, self.num_neuron_types
        )
        self.neuron_type_class = expand_list(
            config.neuron_type_class, self.num_neuron_types
        )
        check_possible_values(
            "neuron_type_class",
            self.neuron_type_class,
            ("excitatory", "inhibitory", "hybrid"),
        )
        self.neuron_type_density = expand_list(
            config.neuron_type_density, self.num_neuron_types
        )
        # TODO: Add support for quarter
        check_possible_values(
            "neuron_type_density",
            self.neuron_type_density,
            ("same", "half"),
        )
        neuron_type_nonlinearity = expand_list(
            config.neuron_type_nonlinearity, self.num_neuron_types
        )
        self.neuron_type_nonlinearity = nn.ModuleList(
            [
                get_activation(nonlinearity)
                for nonlinearity in neuron_type_nonlinearity
            ]
        )

        # Save number of "types" for the input to and output from the area
        self.num_rows_connectivity = (
            1 + int(self.use_feedback) + self.num_neuron_types
        )  # input + feedback + neurons
        self.num_cols_connectivity = (
            self.num_neuron_types + 1
        )  # neurons + output

        # Calculate half spatial size
        self.half_size = (
            ceil(self.in_size[0] / 2),
            ceil(self.in_size[1] / 2),
        )

        self.neuron_type_size = [
            self.in_size
            if self.neuron_type_density[i] == "same"
            else self.half_size
            for i in range(self.num_neuron_types)
        ]

        #####################################################################
        # Circuit motif connectivity
        #####################################################################

        # Format circuit connectivity
        self.inter_neuron_type_connectivity = np.array(
            config.inter_neuron_type_connectivity
        )
        if self.inter_neuron_type_connectivity.shape != (
            self.num_rows_connectivity,
            self.num_cols_connectivity,
        ):
            raise ValueError(
                "The shape of inter_neuron_type_connectivity must match the number of "
                "rows and columns in the connectivity matrix."
            )

        # Format connectivity variables to match circuit connectivity
        self.inter_neuron_type_spatial_extents = expand_array_2d(
            config.inter_neuron_type_spatial_extents,
            self.inter_neuron_type_connectivity.shape[0],
            self.inter_neuron_type_connectivity.shape[1],
            depth=1,
        )
        self.inter_neuron_type_num_subtype_groups = expand_array_2d(
            config.inter_neuron_type_num_subtype_groups,
            self.inter_neuron_type_connectivity.shape[0],
            self.inter_neuron_type_connectivity.shape[1],
        )
        self.inter_neuron_type_nonlinearity = expand_array_2d(
            config.inter_neuron_type_nonlinearity,
            self.inter_neuron_type_connectivity.shape[0],
            self.inter_neuron_type_connectivity.shape[1],
        )
        self.inter_neuron_type_bias = expand_array_2d(
            config.inter_neuron_type_bias,
            self.inter_neuron_type_connectivity.shape[0],
            self.inter_neuron_type_connectivity.shape[1],
        )

        #####################################################################
        # Circuit motif convolutions
        # Here, we represent the circuit connectivity between neuron classes
        # as an array of convolutions (implemented as a dictionary for
        # efficiency). The convolution self.convs[f"{i}->{j}"] corresponds
        # to the connection from neuron class i to neuron class j.
        #####################################################################

        self.convs = nn.ModuleDict()
        self.out_convs = nn.ModuleDict()
        for i, row in enumerate(self.inter_neuron_type_connectivity):
            # Handle input neuron channel and spatial mode based on neuron type
            conv_in_type = self._source_from_row_idx(i)
            if conv_in_type == "input":
                conv_in_channels = self.in_channels
                conv_in_density = "same"
            elif conv_in_type == "feedback":
                conv_in_channels = self.feedback_channels
                conv_in_density = "same"
            else:
                assert conv_in_type == "cell"
                conv_in_channels = self.num_neuron_subtypes[
                    i - 1 - int(self.use_feedback)
                ]
                conv_in_density = self.neuron_type_density[
                    i - 1 - int(self.use_feedback)
                ]

            conv_in_class = self._class_from_row_idx(i)
            if conv_in_class in ("excitatory", "inhibitory"):
                Conv2d = Conv2dRectify
            else:
                assert conv_in_class == "hybrid"
                Conv2d = nn.Conv2d

            # Handle output neurons
            # TODO: Optimize using smart grouping and convolution sharing
            to_indices = np.nonzero(row)[0]
            for j in to_indices:
                if self.inter_neuron_type_connectivity[i, j]:
                    # Handle output neuron channel and spatial mode based on neuron type
                    conv_out_type = self._destination_from_col_idx(j)
                    if conv_out_type == "cell":
                        conv_out_channels: int = self.num_neuron_subtypes[j]  # type: ignore
                        conv_out_density: str = self.neuron_type_density[j]  # type: ignore
                    else:
                        assert conv_out_type == "output"
                        if conv_in_type in ("input", "feedback"):
                            warnings.warn(
                                "Input or feedback is connected to output. "
                                "This is typically undesired as the signal "
                                "will bypass the neuron types and go directly "
                                "to the output. Consider removing this "
                                "connection in the inter_neuron_type_connectivity "
                                "matrix."
                            )
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
                            kernel_size=self.inter_neuron_type_spatial_extents[
                                i, j
                            ],
                            stride=conv_stride,
                            padding=(
                                self.inter_neuron_type_spatial_extents[i, j][0]
                                // 2,
                                self.inter_neuron_type_spatial_extents[i, j][1]
                                // 2,
                            ),
                            groups=self.inter_neuron_type_num_subtype_groups[
                                i, j
                            ],
                            bias=self.inter_neuron_type_bias[i, j],
                        )
                    )
                    conv.append(
                        get_activation(
                            self.inter_neuron_type_nonlinearity[i, j]
                        )
                    )
                    if conv_out_type == "output":
                        self.out_convs[f"{i}->out"] = conv
                    else:
                        self.convs[f"{i}->{j}"] = conv

        #####################################################################
        # Post convolution operations
        #####################################################################

        # Initialize membrane time constants
        self.tau_mode = expand_list(config.tau_mode, self.num_neuron_types)
        check_possible_values(
            "tau_mode",
            self.tau_mode,
            ("subtype", "spatial", "subtype_spatial", "type"),
        )
        self.tau_init_fn = expand_list(
            config.tau_init_fn, self.num_neuron_types
        )

        self.tau = nn.ParameterList()
        for i in range(self.num_neuron_types):
            if self.tau_mode[i] == "spatial":
                tau_channels = 1
                tau_size = self.neuron_type_size[i]
            elif self.tau_mode[i] == "subtype":
                tau_channels = self.num_neuron_subtypes[i]
                tau_size = (1, 1)
            elif self.tau_mode[i] == "subtype_spatial":
                tau_channels = self.num_neuron_subtypes[i]
                tau_size = self.neuron_type_size[i]
            else:
                assert self.tau_mode[i] == "type"
                tau_channels = 1
                tau_size = (1, 1)

            tau = init_tensor(
                self.tau_init_fn[i],
                1,
                tau_channels,
                *tau_size,
            )
            noise = torch.rand_like(tau) * 1e-6

            self.tau.append(
                nn.Parameter(
                    tau + noise,
                    requires_grad=True,
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

    def _source_from_row_idx(self, idx: int) -> Optional[str]:
        """Converts a row index to the corresponding source.

        Args:
            idx: Row index in the circuit connectivity matrix.

        Returns:
            The source associated with the index. Can be "input",
            "feedback", or "cell".
        """
        if idx == 0:
            return "input"
        elif self.use_feedback and idx == 1:
            return "feedback"
        else:
            return "cell"

    def _class_from_row_idx(self, idx: int) -> Optional[str]:
        """Converts a row index to the corresponding class.

        Args:
            idx: Row index in the circuit connectivity matrix.

        Returns:
            The class associated with the index. Can be "excitatory",
            "inhibitory", or "hybrid".
        """
        source = self._source_from_row_idx(idx)
        if source == "input":
            return self.in_class
        elif source == "feedback":
            return self.feedback_class
        else:
            return self.neuron_type_class[idx - 1 - int(self.use_feedback)]  # type: ignore

    def _destination_from_col_idx(self, idx: int) -> Optional[str]:
        """Converts a column index to the corresponding destination.

        Args:
            idx: Column index in the circuit connectivity matrix.

        Returns:
            The destination associated with the index. Can be "cell" or
            "output".
        """
        if idx < self.num_cols_connectivity - 1:
            return "cell"
        else:
            return "output"

    def _clamp_tau(self) -> None:
        for tau in self.tau:
            tau.data = torch.clamp(tau, min=1.0)

    def init_neuron_state(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[torch.Tensor]:
        """initializers the neuron hidden states.

        Args:
            batch_size: Batch size.
            init_fn: Initialization mode. Must be 'zeros', 'ones', 'randn', 'rand',
                a function, or None. If None, the default initialization mode will be used.
                If a function, it must take a variable number of positional arguments
                corresponding to the shape of the tensor to initialize, as well
                as a `device` keyword argument that sends the device to allocate
                the tensor on.
            device: Device to allocate the hidden states on.

        Returns:
            A list containing the initialized neuron hidden states.
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
                self.num_neuron_subtypes[i],
                *self.in_size,
                device=device,
            )
            for i in range(self.num_neuron_types)
        ]

    def init_output_state(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """Initializes the output.

        Args:
            batch_size: Batch size.
            init_fn: Initialization function. Must be 'zeros', 'ones', 'randn', 'rand',
                a function, or None. If None, the default initialization mode will be used.
            device: Device to allocate the hidden states on.

        Returns:
            The initialized output.
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
            batch_size: Batch size.
            init_fn: Initialization function. Must be 'zeros', 'ones', 'randn', 'rand',
                a function, or None. If None, the default initialization mode will be used.
            device: Device to allocate the hidden states on.

        Returns:
            The initialized feedback input if `use_feedback` is True, otherwise None.
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
            DataFrame with columns for neuron type, spatial mode, and number of channels.
        """

        df_columns = defaultdict(list)
        for i in range(self.num_neuron_types):
            df_columns["type"].append(self.neuron_type_class[i])
            df_columns["spatial_mode"].append(self.neuron_type_density[i])
            df_columns["channels"].append(self.num_neuron_subtypes[i])

        df = pd.DataFrame(df_columns)

        return df

    def conv_connectivity_df(self) -> pd.DataFrame:
        """Creates a DataFrame representing connectivity between neural populations.

        Returns:
            DataFrame with rows representing source populations ("from") and
            columns representing target populations ("to").
        """
        row_labels = (
            ["input"]
            + (["feedback"] if self.use_feedback else [])
            + [f"neuron_{i}" for i in range(self.num_neuron_types)]
        )
        column_labels = [
            f"neuron_{i}" for i in range(self.num_neuron_types)
        ] + ["output"]

        assert len(row_labels) == self.num_rows_connectivity
        assert len(column_labels) == self.num_cols_connectivity

        array = np.empty((len(row_labels), len(column_labels)), dtype=object)
        for i in range(self.num_rows_connectivity):
            for j in range(self.num_cols_connectivity):
                if self.inter_neuron_type_connectivity[i, j]:
                    content = []
                    content.append(
                        "b" if self.inter_neuron_type_bias[i, j] else "_"
                    )
                    content.append(
                        self.inter_neuron_type_nonlinearity[i, j][:2]
                        if self.inter_neuron_type_nonlinearity[i, j]
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

    def summary(self) -> str:
        """Returns a string representation of the SpatiallyEmbeddedArea.

        Returns:
            String representation of the SpatiallyEmbeddedArea.
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
            input: Input tensor of shape (batch_size, in_channels, in_size[0], in_size[1]).
            h_neuron: List of neuron hidden states of shape (batch_size, neuron_channels[i],
                in_size[0], in_size[1]) for each neuron type i.
            feedback_state: Feedback input of shape (batch_size, feedback_channels,
                in_size[0], in_size[1]).

        Returns:
            A tuple containing the output and new neuron hidden state.
        """

        # Expand h_neuron to match the number of neuron channels
        if isinstance(neuron_state, torch.Tensor):
            if self.num_neuron_types != 1:
                raise ValueError(
                    "neuron_state must be a list of tensors if num_neuron_types is not 1."
                )
            neuron_state = [neuron_state]

        # Check if feedback is provided if necessary
        if self.use_feedback == (feedback_state is None):
            raise ValueError(
                "use_feedback must be True if and only if feedback_state is provided."
            )

        # Compute convolutions for each connection in the circuit
        circuit_ins = (
            [input]
            + ([feedback_state] if self.use_feedback else [])
            + neuron_state
        )
        circuit_outs = [[] for _ in range(self.num_neuron_types)]
        for key, conv in self.convs.items():
            i, j = key.split("->")
            i, j = int(i), int(j)
            if self._class_from_row_idx(i) == "inhibitory":
                sign = -1
            else:
                sign = 1

            circuit_outs[j].append(sign * conv(circuit_ins[i]))

        # Update neuron states
        self._clamp_tau()
        neuron_state_new = []
        for i in range(self.num_neuron_types):
            # Aggregate all circuit outputs to this neuron type
            state_new = torch.stack(circuit_outs[i], dim=0).sum(dim=0)
            state_new = self.neuron_type_nonlinearity[i](state_new)

            # Euler update
            state_new = (
                state_new / self.tau[i]
                + (1 - 1 / self.tau[i]) * neuron_state[i]
            )
            neuron_state_new.append(state_new)

        # Compute output
        out = []
        for key, conv in self.out_convs.items():
            i, j = key.split("->")
            i = int(i)

            if self._class_from_row_idx(i) == "inhibitory":
                sign = -1
            else:
                sign = 1

            source = self._source_from_row_idx(i)
            if source == "cell":
                i = i - 1 - int(self.use_feedback)
                out.append(sign * conv(neuron_state_new[i]))
            else:
                warnings.warn(
                    f"Connection from {source} to output is not "
                    "recommended. Consider changing inter_neuron_type_connectivity "
                    "to remove this connection."
                )
                out.append(sign * conv(circuit_ins[i]))

        out = torch.stack(out, dim=0).sum(dim=0)

        return out, neuron_state_new


class SpatiallyEmbeddedRNN(nn.Module):
    """Spatially embedded RNN.

    This module stacks multiple SpatiallyEmbeddedArea instances with optional
    feedback connections between them.
    It handles spatial dimension matching between areas and provides a
    flexible interface for configuring the network.

    Attributes:
        num_areas: Number of SpatiallyEmbeddedArea instances in the network.
        areas: ModuleList containing the SpatiallyEmbeddedArea instances.
        feedback_convs: ModuleDict of feedback convolution layers between areas.
        area_time_delay: Whether to introduce a time delay between areas.
        pool_mode: Pooling mode for area outputs ('max' or 'avg').
        batch_first: Whether input has batch dimension as the first dimension.
        inter_area_feedback_connectivity: Connectivity matrix for feedback
            connections between areas.
        inter_area_feedback_nonlinearity: Nonlinearities for feedback
            connections between areas.
        inter_area_feedback_spatial_extents: Kernel sizes for feedback
            convolutions between areas.

    Examples:
        >>> # Create a SpatiallyEmbeddedRNN with 2 areas
        >>> area_configs = [
        ...     SpatiallyEmbeddedAreaConfig(
        ...         in_channels=1,
        ...         out_channels=16,
        ...         in_size=(64, 64),
        ...         feedback_channels=16,
        ...     ),
        ...     SpatiallyEmbeddedAreaConfig(
        ...         in_channels=16,
        ...         out_channels=32,
        ...         in_size=(32, 32),
        ...         feedback_channels=32,
        ...     ),
        ... ]
        >>> connectivity = [
        ...     [0, 0],
        ...     [1, 0]
        ... ]
        >>> rnn = SpatiallyEmbeddedRNN(
        ...     num_areas=2,
        ...     area_configs=area_configs,
        ...     inter_area_feedback_connectivity=connectivity,
        ... )
        >>> # Create a SpatiallyEmbeddedRNN with 2 areas
        >>> area_configs = [
        >>>     SpatiallyEmbeddedAreaConfig(in_channels=1, out_channels=16, in_size=(64, 64)),
        >>>     SpatiallyEmbeddedAreaConfig(in_channels=16, out_channels=32, in_size=(32, 32)),
        >>> ]
        >>> rnn = SpatiallyEmbeddedRNN(num_areas=2, area_configs=area_configs)
    """

    def __init__(
        self,
        *,
        num_areas: int = 1,
        area_configs: Optional[Sequence[SpatiallyEmbeddedAreaConfig]] = None,
        area_kwargs: Optional[Sequence[Mapping[str, Any]]] = None,
        common_area_kwargs: Optional[Mapping[str, Any]] = None,
        inter_area_feedback_connectivity: Optional[
            InterAreaParam[Union[int, bool]]
        ] = None,
        inter_area_feedback_nonlinearity: Optional[
            InterAreaParam[Union[str, nn.Module, None]]
        ] = None,
        inter_area_feedback_spatial_extents: InterAreaParam[
            tuple[int, int]
        ] = (3, 3),
        area_time_delay: bool = False,
        pool_mode: Optional[str] = "max",
        batch_first: bool = True,
    ):
        """Initialize the SpatiallyEmbeddedRNN.

        Args:
            num_areas: Number of `SpatiallyEmbeddedArea` instances in the network.
            area_configs: Configuration object(s) for the `SpatiallyEmbeddedArea` instances.
                If provided as a list, must match the number of areas. If a single config
                is provided, it will be used for all areas with appropriate adjustments.
            area_kwargs: Additional keyword arguments for each area. If provided as a list,
                must match the number of areas.
            common_area_kwargs: Keyword arguments to apply to all areas.
            inter_area_feedback_connectivity: Connectivity matrix for feedback connections
                between areas of shape (num_areas, num_areas). Must be lower triangular and
                zero/False on the diagonal.
            inter_area_feedback_nonlinearity: Nonlinearities for feedback connections of
                shape (num_areas, num_areas).
            inter_area_feedback_spatial_extents: Kernel sizes for feedback convolutions
                of shape (num_areas, num_areas).
            pool_mode: Pooling mode for area outputs.
            area_time_delay: Whether to introduce a time delay between areas.
            batch_first: Whether the input tensor has batch dimension as the first dimension.

        Raises:
            ValueError: If any of the provided parameters are invalid.
        """
        super().__init__()

        ############################################################
        # Area configs
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

        self.feedback_convs = nn.ModuleDict()
        if inter_area_feedback_connectivity is None:
            if any(
                area_config.feedback_channels for area_config in area_configs
            ):
                raise ValueError(
                    "inter_area_feedback_connectivity must be provided if and only if "
                    "feedback_channels is provided for at least one area."
                )
        else:
            self.inter_area_feedback_connectivity = np.array(
                inter_area_feedback_connectivity, dtype=bool
            )
            if self.inter_area_feedback_connectivity.shape != (
                self.num_areas,
                self.num_areas,
            ):
                raise ValueError(
                    "The shape of inter_area_feedback_connectivity must be (num_areas, num_areas)."
                )
            self.inter_area_feedback_nonlinearity = expand_array_2d(
                inter_area_feedback_nonlinearity,
                self.num_areas,
                self.num_areas,
            )
            self.inter_area_feedback_spatial_extents = expand_array_2d(
                inter_area_feedback_spatial_extents,
                self.num_areas,
                self.num_areas,
                depth=1,
            )

            # Validate inter_area_feedback_connectivity tensor
            if (
                self.inter_area_feedback_connectivity.ndim != 2
                or self.inter_area_feedback_connectivity.shape[0]
                != self.num_areas
                or self.inter_area_feedback_connectivity.shape[1]
                != self.num_areas
            ):
                raise ValueError(
                    "The dimensions of inter_area_feedback_connectivity must match the number of areas."
                )
            if self.inter_area_feedback_connectivity.sum() == 0:
                raise ValueError(
                    "inter_area_feedback_connectivity must be a non-zero tensor if provided."
                )

            # Create feedback convolutions
            for i, row in enumerate(self.inter_area_feedback_connectivity):
                nonzero_indices = np.nonzero(row)[0]
                for j in nonzero_indices:
                    if i <= j:
                        raise ValueError(
                            f"the feedback connection from area {i} to area {j} "
                            f"is not valid because feedback connections must "
                            f"pass information from later areas to earlier "
                            f"areas (i.e. inter_area_feedback_connectivity must "
                            f"be lower triangular and zero on the diagonal)."
                        )
                    if not self.areas[j].use_feedback:
                        raise ValueError(
                            f"the connection from area {i} to area {j} is "
                            f"not valid because area {j} does not receive "
                            f"feedback (hint: feedback_channels may not be provided)"
                        )
                    self.feedback_convs[f"{i}->{j}"] = nn.Sequential(
                        nn.Upsample(
                            size=area_configs[j].in_size,
                            mode="bilinear",
                        ),
                        nn.Conv2d(
                            in_channels=area_configs[i].out_channels,
                            out_channels=area_configs[j].feedback_channels,
                            kernel_size=self.inter_area_feedback_spatial_extents[
                                i, j
                            ],
                            padding=(
                                self.inter_area_feedback_spatial_extents[i, j][
                                    0
                                ]
                                // 2,
                                self.inter_area_feedback_spatial_extents[i, j][
                                    1
                                ]
                                // 2,
                            ),
                            bias=area_configs[
                                j
                            ].default_feedback_state_init_fn,
                        ),
                        get_activation(
                            self.inter_area_feedback_nonlinearity[i, j]
                        ),
                    )

    def init_neuron_states(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[list[torch.Tensor]]:
        """Initializes the hidden states for all areas.

        Args:
            batch_size: Batch size.
            init_fn: Initialization function.
            device: Device to allocate tensors.

        Returns:
            A list containing the initialized neuron hidden states for each area.
        """

        return [
            area.init_neuron_state(batch_size, init_fn, device)  # type: ignore
            for area in self.areas
        ]

    def init_feedback_states(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[Optional[torch.Tensor]]:
        """Initializes the feedback inputs for all areas.

        Args:
            batch_size: Batch size.
            init_fn: Initialization function.
            device: Device to allocate tensors.

        Returns:
            A list of initialized feedback inputs for each area.
        """
        return [
            area.init_feedback_state(batch_size, init_fn, device)  # type: ignore
            for area in self.areas
        ]

    def init_output_states(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[torch.Tensor]:
        """Initializes the outputs for all areas.

        Args:
            batch_size: Batch size.
            init_fn: Initialization function.
            device: Device to allocate tensors.

        Returns:
            A list of initialized outputs for each area.
        """

        return [
            area.init_output_state(batch_size, init_fn, device)  # type: ignore
            for area in self.areas
        ]

    def init_states(
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
            out0: Initial outputs for each area. If None, default initialization is used.
            h_neuron0: Initial neuron hidden states for each area. If None, default
                initialization is used.
            fb0: Initial feedback inputs for each area. If None, default initialization is used.
            num_steps: Number of time steps.
            batch_size: Batch size.
            out_init_fn: Initialization function for outputs if out0 is None. Defaults to None.
            hidden_init_fn: Initialization function for hidden states if h_neuron0 is None.
                Defaults to None.
            fb_init_fn: Initialization function for feedback inputs if fb0 is None.
                Defaults to None.
            device: Device to allocate tensors on. Defaults to None.

        Returns:
            A tuple containing:
            - Initialized outputs for each area and time step
            - Initialized neuron hidden states for each area, time step, and neuron type
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
                [None] * self.areas[i].num_neuron_types  # type: ignore
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
        h_neuron0_default = self.init_neuron_states(
            batch_size, hidden_init_fn, device
        )
        fb0_default = self.init_feedback_states(batch_size, fb_init_fn, device)
        out0_default = self.init_output_states(batch_size, out_init_fn, device)

        # Initialize output, hidden state, and feedback lists
        outs: list[list[Union[torch.Tensor, None]]] = [
            [None] * num_steps for _ in range(self.num_areas)
        ]
        h_neurons: list[list[list[Union[torch.Tensor, None]]]] = [
            [[None] * self.areas[i].num_neuron_types for _ in range(num_steps)]  # type: ignore
            for i in range(self.num_areas)
        ]
        fbs: list[list[Union[torch.Tensor, int, None]]] = [
            [0 if self.areas[i].use_feedback else None] * num_steps
            for i in range(self.num_areas)
        ]

        # Fill time step -1 with the initial values
        for i in range(self.num_areas):
            outs[i][-1] = out0[i] if out0[i] is not None else out0_default[i]
            fbs[i][-1] = fb0[i] if fb0[i] is not None else fb0_default[i]
            for k in range(self.areas[i].num_neuron_types):  # type: ignore
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
            x: Input tensor, can be:
                - 4D tensor of shape (batch_size, channels, height, width) for a
                  single time step. Will be expanded to all time steps.
                - 5D tensor of shape (seq_len, batch_size, channels, height, width) or
                  (batch_size, seq_len, channels, height, width) if batch_first=True.
            num_steps: Number of time steps. Required if x is 4D.
                If x is 5D, it will be inferred from the sequence dimension unless
                explicitly provided.

        Returns:
            A tuple containing:
            - The formatted input tensor with shape (seq_len, batch_size, channels, height, width)
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
            outs: Outputs for each area and time step. Shape: [num_areas][num_steps],
            h_neurons: Neuron hidden states for each area, time step, and neuron type.
                Shape: [num_areas][num_steps][num_neuron_types].
            fbs: Feedback inputs for each area and time step. Shape: [num_areas][num_steps].

        Returns:
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
                    for j in range(self.areas[i].num_neuron_types)  # type: ignore
                ]
            )
            if self.areas[i].use_feedback:
                fbs_stack.append(torch.stack(fbs[i]))  # type: ignore
            else:
                assert all(feedback_state is None for feedback_state in fbs[i])
                fbs_stack.append(None)
            if self.batch_first:
                outs_stack[i] = outs_stack[i].transpose(0, 1)
                for j in range(self.areas[i].num_neuron_types):  # type: ignore
                    h_neurons_stack[i][j] = h_neurons_stack[i][j].transpose(
                        0, 1
                    )
                if self.areas[i].use_feedback:
                    fbs_stack[i] = fbs_stack[i].transpose(0, 1)  # type: ignore

        return outs_stack, h_neurons_stack, fbs_stack

    def _match_spatial_size(
        self,
        x: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        """Adjusts the spatial dimensions of the input tensor.

        This method ensures that tensors have compatible spatial dimensions when
        passing between areas. It uses either pooling (when downsampling) or
        interpolation (when upsampling).

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            size: Target spatial size (height, width).

        Returns:
            Resized tensor matching the target spatial size.

        Raises:
            ValueError: If self.pool_mode is not 'avg' or 'max'.
        """
        if x.shape[-2] > size[0] and x.shape[-1] > size[1]:
            if self.pool_mode == "avg":
                return F.adaptive_avg_pool2d(x, size)
            elif self.pool_mode == "max":
                return F.adaptive_max_pool2d(x, size)
            else:
                raise ValueError(f"Invalid pool_mode: {self.pool_mode}")
        elif x.shape[-2] < size[0] and x.shape[-1] < size[1]:
            return F.interpolate(x, size, mode="bilinear", align_corners=False)
        else:
            assert x.shape[-2] == size[0] and x.shape[-1] == size[1]
            return x

    def query_neuron_states(
        self,
        neuron_states: list[list[torch.Tensor]],
        area: int,
        neuron_type: int,
        time_step: Optional[Union[int, slice]] = None,
        batch: Optional[Union[int, slice]] = None,
        neuron_subtype: Optional[Union[int, slice]] = None,
        spatial_location_i: Optional[Union[int, slice]] = None,
        spatial_location_j: Optional[Union[int, slice]] = None,
    ) -> torch.Tensor:
        """Query the model states for a given area, time step, neuron type, neuron subtype, and spatial location.

        Args:
            neuron_states: List of lists of tensors containing the model states.
            area: The area index.
            neuron_type: The neuron type index.
            time_step: The time step index. If not provided, all time steps are returned.
            batch: The batch index. If not provided, all batches are returned.
            neuron_subtype: The neuron subtype index. If not provided, all neuron subtypes are returned.
            spatial_location_i: The spatial location (height). If not provided, all spatial locations are returned.
            spatial_location_j: The spatial location (width). If not provided, all spatial locations are returned.

        Returns:
            The queried state of shape (batch_size_slice, channels, height, width)
        """
        if time_step is None:
            time_idx = slice(None)
        else:
            time_idx = time_step
        if batch is None:
            batch_idx = slice(None)
        else:
            batch_idx = batch
        if neuron_subtype is None:
            neuron_subtype_idx = slice(None)
        else:
            neuron_subtype_idx = neuron_subtype
        if spatial_location_i is None:
            spatial_location_idx_i = slice(None)
        else:
            spatial_location_idx_i = spatial_location_i
        if spatial_location_j is None:
            spatial_location_idx_j = slice(None)
        else:
            spatial_location_idx_j = spatial_location_j

        if self.batch_first:
            return neuron_states[area][neuron_type][
                batch_idx,
                time_idx,
                neuron_subtype_idx,
                spatial_location_idx_i,
                spatial_location_idx_j,
            ]
        else:
            return neuron_states[area][neuron_type][
                time_idx,
                batch_idx,
                neuron_subtype_idx,
                spatial_location_idx_i,
                spatial_location_idx_j,
            ]

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
            x: Input tensor. Can be either:
                - 4D tensor of shape (batch_size, in_channels, height, width)
                  representing a single time step. In this case, num_steps must be
                  provided.
                - 5D tensor of shape (seq_len, batch_size, in_channels, height, width)
                  or (batch_size, seq_len, in_channels, height, width) if
                  batch_first=True.
            num_steps: Number of time steps. Required if x is 4D.
                If x is 5D, this must match the sequence length dimension in x.
            output_state0: Initial outputs for each area. Length should match the
                number of areas. Each element can be None to use default initialization.
            neuron_state0: Initial neuron hidden states for each area and neuron type.
                Length should match the number of areas, and each inner sequence length
                should match the number of neuron types in that area.
            feedback_state0: Initial feedback inputs for each area. Length should match
                the number of areas.

        Returns:
            A tuple containing:
            - Outputs for each area. Each tensor has shape:
              (seq_len, batch_size, out_channels, height, width) or
              (batch_size, seq_len, out_channels, height, width) if batch_first=True.
            - Hidden states for each area and neuron type.
              Same shape pattern as outputs but with neuron_channels.
            - Feedback inputs for each area (None if the area doesn't use feedback).

        Raises:
            ValueError: If input shape is invalid or num_steps is inconsistent with
                the input shape.
        """

        device = x.device

        x, num_steps = self._format_x(x, num_steps)

        batch_size = x.shape[1]

        output_states, neuron_states, feedback_states = self.init_states(
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
                else:
                    if self.area_time_delay:
                        area_in = output_states[i - 1][t - 1]
                    else:
                        area_in = output_states[i - 1][t]
                    assert isinstance(area_in, torch.Tensor)
                    area_in = self._match_spatial_size(
                        area_in,
                        self.areas[i].in_size,  # type: ignore
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
