from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplnn.models.sparse import SparseODERNN, SparseRNN
from bioplnn.typing import PathLikeType, ScalarOrListLike, TensorInitFnType
from bioplnn.utils import (
    get_activation,
    init_tensor,
    load_array,
)

# TODO: Some docstrings may be outdated, might need to update

attributes_docstring = """
    num_neurons (int): Number of neurons in the network (equivalent to
        hidden_size).
    num_neuron_types (int): Number of distinct neuron types.
    default_neuron_state_init_fn (callable): Default initialization
        function for neuron states.
    neuron_type_mask (torch.Tensor): Tensor mapping each neuron to its
        type index.
    neuron_type_indices (List[torch.Tensor]): List of indices for each
        neuron type.
    neuron_type_index_map (dict): Mapping from original type labels to
        consecutive indices.
    neuron_sign_mask (torch.Tensor): Mask indicating excitatory (+1) or
        inhibitory (-1) neurons.
    neuron_class_mode (str): Determines if neuron classes are 'per_neuron'
        or 'per_neuron_type'.
    tau (torch.Tensor): Time constant parameters for neurons or types.
    neuron_tau_mode (str): Whether tau is 'per_neuron' or 'per_neuron_type'.
    neuron_nonlinearity (Union[nn.Module, nn.ModuleList]): Activation
        functions per neuron type.
"""

key_features_docstring = """
        - Sparse connectivity: The connectivity matrix is sparse, meaning that only
          a small fraction of the neurons are connected to each other.
        - Biologically-inspired dynamics: The dynamics of the network are inspired
          by the dynamics of neurons in the brain.
        - Efficient simulation: The network is simulated efficiently using the
          `SparseRNN` class.
"""


class _ConnectomeRNNMixIn:
    f"""MixIn for ConnectomeRNN and ConnectomeODERNN.

    This class is used for common initialization and forward methods for
    ConnectomeRNN and ConnectomeODERNN. It is not intended to be used directly.
    It MUST be the first class in the MRO of ConnectomeRNN or ConnectomeODERNN
    (besides itself).

    Attributes:
        {attributes_docstring}
    """

    def __init__(
        self,
        input_size: int,
        num_neurons: int,
        connectome: Union[torch.Tensor, PathLikeType],
        output_size: Optional[int] = None,
        input_projection: Optional[Union[torch.Tensor, PathLikeType]] = None,
        output_projection: Optional[Union[torch.Tensor, PathLikeType]] = None,
        use_dense_input_projection: bool = False,
        use_dense_output_projection: bool = False,
        train_connectome: bool = True,
        train_input_projection: bool = True,
        train_output_projection: bool = True,
        num_neuron_types: int = 0,
        neuron_type: Optional[
            Union[torch.Tensor, Iterable[int], PathLikeType]
        ] = None,
        neuron_class: Union[
            Mapping[str, Union[str, int]],
            ScalarOrListLike[Union[str, int]],
            torch.Tensor,
            Iterable[Union[str, int]],
            PathLikeType,
        ] = "excitatory",
        neuron_class_mode: str = "per_neuron",
        neuron_tau_init: Optional[
            Union[
                Mapping[str, float],
                ScalarOrListLike[float],
                torch.Tensor,
                Iterable[float],
                PathLikeType,
            ]
        ] = None,
        neuron_tau_init_fn: Union[str, TensorInitFnType] = "ones",
        neuron_tau_mode: str = "per_neuron",
        train_tau: bool = True,
        neuron_nonlinearity: Union[
            Mapping[str, Optional[Union[str, nn.Module]]],
            ScalarOrListLike[Union[str, nn.Module, None]],
        ] = "Sigmoid",
        default_neuron_state_init_fn: Union[str, TensorInitFnType] = "zeros",
        connectome_bias: bool = True,
        output_projection_bias: bool = True,
        batch_first: bool = True,
        compile_solver_kwargs: Optional[Mapping[str, Any]] = None,
        compile_update_fn_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize ConnectomeRNN and ConnectomeODERNN.

        Args:
            input_size: Size of the input layer.
            num_neurons: Number of neurons in the network.
            connectome: Connectivity matrix or path to it.
            output_size: Size of the output layer. Defaults to None.
            input_projection: Input projection matrix or path. Defaults to None.
            output_projection: Output projection matrix or path. Defaults to None.
            use_dense_input_projection: Use dense input projection if True.
            use_dense_output_projection: Use dense output projection if True.
            train_connectome: Train the connectome if True.
            train_input_projection: Train input projection if True.
            train_output_projection: Train output projection if True.
            num_neuron_types: Number of neuron types.
            neuron_type: Tensor/path defining neuron type indices. Required when
                num_neuron_types > 0.
            neuron_class: Neuron class specifications (excitatory/inhibitory).
            neuron_class_mode: Determines if neuron classes are 'per_neuron' or
                'per_neuron_type'.
            neuron_tau_init: Initial time constants for neurons. Can provide
                per-type or per-neuron values.
            neuron_tau_init_fn: Initialization function for time constants
                (clamped ≥ 1.0). Prefer initializers near 1.0 like 'ones'.
            neuron_tau_mode: Whether tau is 'per_neuron' or 'per_neuron_type'.
            train_tau: Whether time constants are trainable parameters.
            neuron_nonlinearity: Nonlinearity for neurons. Can be single or list
                of activations matching num_neuron_types.
            default_neuron_state_init_fn: Hidden state initialization function.
            connectome_bias: Add bias to connectome if True.
            output_projection_bias: Add bias to output projection if True.
            batch_first: Whether input tensors have batch dimension first.
            compile_solver_kwargs: Kwargs for ODE solver compilation.
            compile_update_fn_kwargs: Kwargs for update_fn compilation.
        """

        # Initialize parent class
        assert isinstance(self, (ConnectomeRNN, ConnectomeODERNN))
        kwargs = {
            "input_size": input_size,
            "hidden_size": num_neurons,
            "output_size": output_size,
            "connectivity_hh": connectome,
            "connectivity_ih": input_projection,
            "connectivity_ho": output_projection,
            "bias_hh": connectome_bias,
            "bias_ho": output_projection_bias,
            "use_dense_ih": use_dense_input_projection,
            "use_dense_ho": use_dense_output_projection,
            "train_hh": train_connectome,
            "train_ih": train_input_projection,
            "train_ho": train_output_projection,
            "default_hidden_init_fn": default_neuron_state_init_fn,
            "batch_first": batch_first,
        }
        if compile_solver_kwargs is not None:
            kwargs["compile_solver_kwargs"] = compile_solver_kwargs
        if compile_update_fn_kwargs is not None:
            kwargs["compile_update_fn_kwargs"] = compile_update_fn_kwargs

        super().__init__(**kwargs)

        # Save relevant attributes
        self.num_neurons = self.hidden_size
        self.num_neuron_types = num_neuron_types
        self.default_neuron_state_init_fn = self.default_hidden_init_fn

        # Neuron type parameters
        if num_neuron_types > 0 and neuron_type is None:
            raise ValueError(
                "neuron_type must be provided iff num_neuron_types > 0."
            )
        if num_neuron_types > 0:
            self._init_neuron_type(neuron_type)
        self._init_neuron_class(neuron_class, neuron_class_mode)
        self._init_neuron_tau_init(
            neuron_tau_init, neuron_tau_init_fn, neuron_tau_mode, train_tau
        )
        self._init_neuron_nonlinearity(neuron_nonlinearity)

    def _init_neuron_type(
        self,
        neuron_type: Optional[
            Union[torch.Tensor, Iterable[Any], PathLikeType]
        ],
    ):
        """Initialize the neuron type parameters."""

        if neuron_type is None:
            raise ValueError(
                "neuron_type must be provided if num_neuron_types > 0."
            )

        # Loads array of neuron type labels per neuron
        neuron_type = self.load_neuron_array(neuron_type, self.num_neurons)

        # Check that neuron_type contains exactly num_neuron_types unique values
        unique_types = np.sort(np.unique(neuron_type))
        if unique_types.shape[0] != self.num_neuron_types:
            raise ValueError(
                "neuron_type must contain exactly num_neuron_types unique values."
            )

        # Create a mapping from neuron type labels to consecutive integers
        # starting from 0
        self.neuron_type_index_map = {
            type: idx for idx, type in enumerate(unique_types)
        }

        # Check if unique_types is already a list of consecutive integers
        # starting from 0
        if np.all(unique_types == np.arange(self.num_neuron_types)):
            assert all(
                [
                    type == idx
                    for type, idx in self.neuron_type_index_map.items()
                ]
            )
            neuron_type_mask = neuron_type
        else:
            neuron_type_mask = np.zeros(self.num_neurons, dtype=np.int_)
            for type, idx in self.neuron_type_index_map.items():
                neuron_type_mask[neuron_type == type] = idx

        # Create a mask of neuron types per neuron
        self.neuron_type_mask = nn.Parameter(
            torch.tensor(neuron_type_mask, dtype=torch.int),
            requires_grad=False,
        )

        # Create a list of indices of neurons of each type
        self.neuron_type_indices = nn.ParameterList()
        for i in range(self.num_neuron_types):
            indices_tensor = torch.where(self.neuron_type_mask == i)[0]
            self.neuron_type_indices.append(
                nn.Parameter(indices_tensor, requires_grad=False)
            )

    def _init_neuron_class(
        self,
        neuron_class: Union[
            Mapping[str, Union[str, int]],
            ScalarOrListLike[Union[str, int]],
            torch.Tensor,
            Iterable[Union[str, int]],
            PathLikeType,
        ],
        neuron_class_mode: str = "per_neuron",
    ) -> None:
        """Initialize the neuron class parameter and associated neuron sign mask."""

        self.neuron_class_mode = neuron_class_mode

        if self.neuron_class_mode == "per_neuron":
            size = self.num_neurons
        elif self.neuron_class_mode == "per_neuron_type":
            if self.num_neuron_types == 0:
                raise ValueError(
                    "neuron_class_mode must be 'per_neuron' if num_neuron_types is 0."
                )
            size = self.num_neuron_types
        else:
            raise ValueError(
                "neuron_class_mode must be either 'per_neuron' or 'per_neuron_type'."
            )
        # Convert mapping to list of neuron type indices
        if isinstance(neuron_class, Mapping):
            assert (
                len(neuron_class) == self.num_neuron_types
                and neuron_class_mode == "per_neuron_type"
            )
            neuron_class = [
                self.neuron_type_index_map[k] for k in neuron_class
            ]

        if isinstance(neuron_class, str) and neuron_class in (
            "excitatory",
            "inhibitory",
        ):
            # Expand scalar neuron_class to a tensor of length num_neurons
            if self.neuron_class_mode == "per_neuron_type":
                raise ValueError(
                    "neuron_class must be non-scalar if neuron_class_mode is 'per_neuron_type'."
                )
            neuron_class_sign = torch.ones(self.num_neurons, dtype=torch.float)
            if neuron_class == "inhibitory":
                neuron_class_sign *= -1
        else:
            # Load neuron class from file or iterable and convert to int if necessary
            self.neuron_class = self.load_neuron_array(
                neuron_class, size, expand=True
            )
            unique_classes = np.sort(np.unique(self.neuron_class))
            if np.all(np.isin(unique_classes, ("excitatory", "inhibitory"))):
                neuron_class_sign = torch.zeros(size, dtype=torch.float)
                neuron_class_sign[self.neuron_class == "excitatory"] = 1
                neuron_class_sign[self.neuron_class == "inhibitory"] = -1
            else:
                assert np.all(np.isin(unique_classes, (-1, 1)))
                neuron_class_sign = torch.tensor(
                    self.neuron_class, dtype=torch.float
                )
        if neuron_class_sign.ndim != 1:
            raise ValueError(
                "neuron_class_sign must be a 1D tensor of length num_neurons."
            )

        # Create a mask of neuron signs per neuron (to be multiplied with
        # hidden state during update_fn)
        neuron_sign_mask = torch.zeros(self.num_neurons, dtype=torch.float)
        if self.neuron_class_mode == "per_neuron_type":
            if neuron_class_sign.shape[0] != self.num_neuron_types:
                raise ValueError(
                    "neuron_class_sign must be of length num_neuron_types."
                )
            for i in range(self.num_neuron_types):
                neuron_sign_mask[self.neuron_type_indices[i]] = (
                    neuron_class_sign[i]
                )
        else:
            if neuron_class_sign.shape[0] != self.num_neurons:
                raise ValueError(
                    "neuron_class_sign must be of length num_neurons."
                )
            neuron_sign_mask = neuron_class_sign

        self.neuron_sign_mask = nn.Parameter(
            neuron_sign_mask.unsqueeze(-1), requires_grad=False
        )

    def _init_neuron_tau_init(
        self,
        neuron_tau_init: Optional[
            Union[
                Mapping[str, float],
                ScalarOrListLike[float],
                torch.Tensor,
                Iterable[float],
                PathLikeType,
            ]
        ],
        neuron_tau_init_fn: Union[str, TensorInitFnType],
        neuron_tau_mode: str,
        train_tau: bool,
    ) -> None:
        """Initialize the neuron type time constant parameters."""

        self.neuron_tau_mode = neuron_tau_mode

        if self.neuron_tau_mode == "per_neuron":
            size = self.num_neurons
        elif self.neuron_tau_mode == "per_neuron_type":
            if self.num_neuron_types == 0:
                raise ValueError(
                    "neuron_tau_mode must be 'per_neuron' if num_neuron_types is 0."
                )
            size = self.num_neuron_types
        else:
            raise ValueError(
                "neuron_tau_mode must be either 'per_neuron' or 'per_neuron_type'."
            )

        if neuron_tau_init is None:
            tau = init_tensor(neuron_tau_init_fn, size, dtype=torch.float)
        else:
            tau = torch.tensor(
                self.load_neuron_array(neuron_tau_init, size, expand=True),
                dtype=torch.float,
            )

        # Validate tau tensor
        assert tau.ndim == 1
        if tau.shape[0] != size:
            raise ValueError(
                f"neuron_tau_init must be of length {size} for {self.neuron_tau_mode} mode, got {tau.shape[0]}."
            )
        if (tau < 1.0).any():
            raise ValueError("tau must be at least 1.0.")

        # Add small noise to tau to avoid zero gradients
        close_to_one = torch.isclose(tau, torch.tensor(1.0))
        num_close_to_one = int(close_to_one.sum().item())
        if num_close_to_one > 0:
            noise = torch.rand(num_close_to_one) * 1e-6
            tau[close_to_one] = tau[close_to_one] + noise

        self.tau = nn.Parameter(tau.unsqueeze(-1), requires_grad=train_tau)

    def _clamp_tau(self) -> None:
        self.tau.data.clamp_(min=1.0)

    def _init_neuron_nonlinearity(
        self,
        neuron_nonlinearity: Union[
            Mapping[str, Optional[Union[str, nn.Module]]],
            ScalarOrListLike[Union[str, nn.Module, None]],
        ] = "Sigmoid",
    ) -> None:
        """Initialize the neuron type nonlinearity parameters."""

        if neuron_nonlinearity is None or isinstance(
            neuron_nonlinearity, (str, nn.Module)
        ):
            self.neuron_nonlinearity = get_activation(neuron_nonlinearity)
            self.neuron_nonlinearity_mode = "one"
        elif len(neuron_nonlinearity) == self.num_neuron_types:
            self.neuron_nonlinearity = nn.ModuleList(
                [
                    get_activation(nonlinearity)
                    for nonlinearity in neuron_nonlinearity
                ]
            )
            self.neuron_nonlinearity_mode = "per_neuron_type"
        else:
            raise ValueError(
                "neuron_nonlinearity must be a string or nn.Module, or a list of "
                "strings or nn.Modules of length num_neuron_types. Per neuron "
                "nonlinearity is currently unsupported."
            )

    @staticmethod
    def load_neuron_array(
        x: Union[Any, torch.Tensor, Iterable[Any], PathLikeType],
        size: int,
        expand: bool = False,
    ) -> np.ndarray:
        """Load a tensor from a file or iterable of integers.

        Args:
            x: Tensor or path to file containing indices.
            size: Expected size of the loaded tensor.

        Returns:
            torch.Tensor: Loaded tensor.
        """

        array = load_array(x).squeeze()

        if array.ndim > 1:
            raise ValueError("x must be a 1D tensor")
        if expand and (array.ndim == 0 or array.shape[0] == 1):
            array = array.repeat(size)
        elif array.shape[0] != size:
            raise ValueError(
                f"x must be of length {size}, got {array.shape[0]}."
            )

        return array

    def init_neuron_state(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """Initialize the hidden state.

        Args:
            batch_size (int): Batch size.
            init_fn (Optional[Union[str, TensorInitFnType]], optional): Initialization
                function. Defaults to None (uses default_init_fn).
            device (Optional[Union[torch.device, str]], optional): Device to allocate the
                hidden state on. Defaults to None.

        Returns:
            torch.Tensor: The initialized hidden state of shape (batch_size, hidden_size).
        """
        if init_fn is None:
            init_fn = self.default_neuron_state_init_fn

        return init_tensor(
            init_fn, batch_size, self.num_neurons, device=device
        )

    def query_neuron_states(
        self,
        neuron_states: torch.Tensor,
        time_step: Optional[Union[int, slice]] = None,
        batch: Optional[Union[int, slice]] = None,
        neuron_type: Optional[Union[str, int]] = None,
        neurons: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Query the model states for a given area, time step, neuron type, neuron subtype, and spatial location.

        Args:
            neuron_states: Tensor containing the neuron states.
            time_step: The time step index. If not provided, all time steps are returned.
            batch: The batch index. If not provided, all batches are returned.
            neuron_type: The neuron type label. If not provided, all neuron types are returned.
            neurons: The neuron indices. If not provided, all neurons are returned.

        Returns:
            The queried state of shape (batch_size_slice, channels, height, width)
        """
        time_idx = time_step if time_step is not None else slice(None)
        batch_idx = batch if batch is not None else slice(None)

        if neurons is not None:
            if neuron_type is not None:
                raise ValueError(
                    "neuron_type must be None if neurons is provided."
                )
            neuron_idx = neurons
        elif neuron_type is not None:
            neuron_type_idx = self.neuron_type_index_map[neuron_type]
            neuron_idx = self.neuron_type_indices[neuron_type_idx]
        else:
            neuron_idx = slice(None)

        if self.batch_first:  # type: ignore
            return neuron_states[batch_idx, time_idx, neuron_idx]
        else:
            return neuron_states[time_idx, batch_idx, neuron_idx]

    def forward(self, *args, **kwargs):
        """Forward pass of the ConnectomeRNN or ConnectomeODERNN.

        Args:
            *args: Arguments to pass to parent class forward method.
            **kwargs: Keyword arguments to pass to parent class forward method.

        Returns:
            - The output sequence.
        """
        self._clamp_tau()

        return super().forward(*args, **kwargs)  # type: ignore

    def update_fn(self, *args, **kwargs):
        raise NotImplementedError(
            "update_fn must be implemented by a subclass of ConnectomeRNNMixIn"
        )


class ConnectomeRNN(_ConnectomeRNNMixIn, SparseRNN):
    f"""Connectome Recurrent Neural Network.

    An RNN that leverages the `SparseRNN` class to efficiently simulate a
    sparsely-connected network of neurons with biologically-inspired dynamics.

    Key features:
        {key_features_docstring}

    Attributes:
        {attributes_docstring}
    """

    def update_fn(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """update hidden state for one timestep.

        Args:
            x_t (torch.Tensor): Input at current timestep.
            h (torch.Tensor): Hidden state at previous timestep.

        Returns:
            torch.Tensor: Updated hidden state.
        """

        # Apply sign mask to input
        h_signed = h * self.neuron_sign_mask

        # Compute new hidden state
        h_new = self.ih(x_t) + self.hh(h_signed)

        if self.neuron_nonlinearity_mode == "one":
            h_new = self.neuron_nonlinearity(h_new)
        else:
            for i in range(self.num_neuron_types):
                h_new[self.neuron_type_indices[i]] = self.neuron_nonlinearity[  # type: ignore
                    i
                ](h_new[self.neuron_type_indices[i]])

        # Rectify hidden state to ensure it's non-negative
        # Note: The user is still expected to provide a nonlinearity that
        #   ensures non-negativity, e.g. sigmoid.
        h_new = F.relu(h_new)

        # Compute rate of change of hidden state
        if self.neuron_tau_mode == "per_neuron":
            tau_inv = 1 / self.tau
            return tau_inv * h_new + (1 - tau_inv) * h
        else:
            tau_inv = 1 / self.tau[self.neuron_type_mask, :]
            return tau_inv * h_new + (1 - tau_inv) * h

    def forward(
        self,
        x,
        num_steps: Optional[int] = None,
        neuron_state0: Optional[torch.Tensor] = None,
        neuron_state_init_fn: Optional[Union[str, TensorInitFnType]] = None,
    ):
        """Forward pass of the ConnectomeODERNN layer.

        Wraps the `SparseODERNN.forward` method to change nomenclature.

        See `SparseODERNN.forward` for more details.
        """
        self._clamp_tau()

        return super().forward(
            x=x,
            num_steps=num_steps,
            h0=neuron_state0,
            hidden_init_fn=neuron_state_init_fn,
        )


class ConnectomeODERNN(_ConnectomeRNNMixIn, SparseODERNN):
    f"""Continuous-time ConnectomeRNN using ODE solver integration.

    A continuous-time version of the ConnectomeRNN that simulates neural
    dynamics using an ODE solver and computes the gradient with respect to the
    parameters for efficient training.

    Key features:
        - ODE solver integration: Built on top of `SparseODERNN`, allowing for
          the efficient computation of the gradients of the hidden states with
          respect to the parameters. In the particular, the `torchode.AutoDiffAdjoint`
          solver allows for O(1) computational complexity for the backward pass with
          respect to the number of simulated timesteps.
        {key_features_docstring}

    Attributes:
        {attributes_docstring}

    Example:
        >>> connectome = create_sparse_topographic_connectome((10, 10), 0.1, 10, True)
        >>> rnn = ConnectomeRNN(10, 10, connectome)
        >>> odenn = ConnectomeODERNN(10, 10, connectome)
    """

    def update_fn(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ) -> torch.Tensor:
        """ODE function for neural dynamics with neuron type differentiation.

        Args:
            t: Current time point.
            h: Current hidden state.
            args: Additional arguments containing:
                x: Input sequence tensor
                start_time: Integration start time
                end_time: Integration end time

        Returns:
            Rate of change of hidden state (dh/dt)
        """
        h = h.t()

        x = args["x"]
        start_time = args["start_time"]
        end_time = args["end_time"]
        batch_size = x.shape[-1]

        # Get index corresponding to time t
        idx = self._index_from_time(t, x, start_time, end_time)

        # Get input at time t
        batch_indices = torch.arange(batch_size, device=idx.device)
        x_t = x[idx, :, batch_indices].t()

        # Apply sign mask to input
        h_signed = h * self.neuron_sign_mask

        # Compute new hidden state
        h_new = self.ih(x_t) + self.hh(h_signed)

        if self.neuron_nonlinearity_mode == "one":
            h_new = self.neuron_nonlinearity(h_new)
        else:
            for i in range(self.num_neuron_types):
                h_new[self.neuron_type_indices[i]] = self.neuron_nonlinearity[  # type: ignore
                    i
                ](h_new[self.neuron_type_indices[i]])

        # Rectify hidden state to ensure it's non-negative
        # Note: The user is still expected to provide a nonlinearity that
        #   ensures non-negativity, e.g. sigmoid.
        h_new = F.relu(h_new)

        # Compute rate of change of hidden state
        if self.neuron_tau_mode == "per_neuron":
            dhdt = (h_new - h) / self.tau
        else:
            dhdt = (h_new - h) / self.tau[self.neuron_type_mask, :]

        return dhdt.t()

    def forward(
        self,
        x,
        num_evals: int = 2,
        start_time: float = 0.0,
        end_time: float = 1.0,
        neuron_state0: Optional[torch.Tensor] = None,
        neuron_state_init_fn: Optional[Union[str, TensorInitFnType]] = None,
    ):
        """Forward pass of the ConnectomeODERNN layer.

        Wraps the `SparseODERNN.forward` method to change nomenclature.

        See `SparseODERNN.forward` for more details.
        """
        self._clamp_tau()

        return super().forward(
            x=x,
            num_evals=num_evals,
            start_time=start_time,
            end_time=end_time,
            h0=neuron_state0,
            hidden_init_fn=neuron_state_init_fn,
        )
