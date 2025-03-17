from collections.abc import Mapping, Sequence
from math import isclose
from os import PathLike
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplnn.models.sparse import SparseODERNN, SparseRNN
from bioplnn.typing import NeuronTypeParam, TensorInitFnType
from bioplnn.utils import (
    check_possible_values,
    expand_list,
    get_activation,
    init_tensor,
)


class ConnectomeRNNMixIn:
    """MixIn for ConnectomeRNN and ConnectomeODERNN.

    This class is used for common initialization and forward methods for
    ConnectomeRNN and ConnectomeODERNN. It is not intended to be used directly.
    It MUST be the first class in the MRO of ConnectomeRNN or ConnectomeODERNN
    (besides itself).

    Args:
        num_neuron_types (Optional[int], optional): Number of neuron types.
            Defaults to None.
        neuron_type_class (Optional[Sequence[str]], optional): Class of each
            neuron type. Defaults to None.
        neuron_type_indices (Optional[Sequence[Sequence[int]]], optional):
            Indices of neurons for each neuron type. Defaults to None.
        neuron_type_nonlinearity (Optional[Sequence[Union[str, nn.Module]]],
            optional): Nonlinearity for each neuron type. Defaults to None.
        neuron_type_tau_init (Optional[Sequence[float]], optional): Initial
            time constant for each neuron type. Defaults to None.
        tau_init_fn (Union[str, TensorInitFnType], optional): Function or name
            of torch function to initialize time constants. Note that tau is
            clamped to be at least 1.0 after initialization, so it is
            recommended to use a function that initializes to values close to
            1.0, e.g. "ones" or "normal". Defaults to "ones".
    """

    def __init__(
        self,
        *args,
        output_neurons: Optional[Union[torch.Tensor, PathLike]] = None,
        num_neuron_types: Optional[int] = None,
        neuron_type_class: NeuronTypeParam[str] = "hybrid",
        neuron_type_indices: Optional[Sequence[Sequence[int]]] = None,
        neuron_type_nonlinearity: NeuronTypeParam[
            Optional[Union[str, nn.Module]]
        ] = "Sigmoid",
        neuron_type_tau_init: NeuronTypeParam[float] = 1.0,
        default_tau_init_fn: Union[str, TensorInitFnType] = "ones",
        **kwargs,
    ):
        # Initialize parent class
        if isinstance(self, ConnectomeRNN):
            super(SparseRNN, self).__init__(*args, **kwargs)
        else:
            assert isinstance(self, ConnectomeODERNN)
            super(SparseODERNN, self).__init__(*args, **kwargs)

        self.output_neurons = self._init_output_neurons(output_neurons)

        # Neuron type parameters
        self.num_neuron_types = num_neuron_types
        if num_neuron_types is not None:
            # Neuron type indices
            if (
                neuron_type_indices is None
                or len(neuron_type_indices) != num_neuron_types
            ):
                raise ValueError(
                    "neuron_type_indices must be provided and have length "
                    f"equal to num_neuron_types ({num_neuron_types})."
                )
            self.neuron_type_indices = []
            for i in range(num_neuron_types):
                self.neuron_type_indices.append(
                    torch.tensor(neuron_type_indices[i], dtype=torch.long)
                )

            # Neuron type class
            self.neuron_type_class = expand_list(
                neuron_type_class, num_neuron_types
            )
            check_possible_values(
                "neuron_type_class",
                self.neuron_type_class,
                ("excitatory", "inhibitory", "hybrid"),
            )

            # Neuron type nonlinearity
            neuron_type_nonlinearity = expand_list(
                neuron_type_nonlinearity, num_neuron_types
            )
            self.neuron_type_nonlinearity = [
                get_activation(nonlinearity)
                for nonlinearity in neuron_type_nonlinearity
            ]

            # Neuron type time constants
            self.neuron_type_tau_init = expand_list(
                neuron_type_tau_init, num_neuron_types
            )
            self.taus = nn.ParameterList()
            for i in range(num_neuron_types):
                tau = torch.tensor(self.neuron_type_tau_init[i])
                if tau < 1.0:
                    raise ValueError(
                        "neuron_type_tau_init must be at least 1.0."
                    )
                if isclose(tau, 1.0):
                    tau += torch.rand_like(tau) * 1e-6
                self.taus.append(nn.Parameter(tau))
        else:
            tau = init_tensor(default_tau_init_fn, 1, self.hidden_size)  # type: ignore
            noise = torch.rand_like(tau) * 1e-6
            self.tau = nn.Parameter(tau + noise)

    def _init_output_neurons(
        self,
        output_neurons: Union[torch.Tensor, PathLike, None] = None,
    ) -> Union[torch.Tensor, None]:
        """Initialize output neuron indices.

        Args:
            output_neurons (Union[torch.Tensor, PathLike], optional): Tensor or
                path to file containing indices of output neurons. Defaults to
                None.

        Returns:
            Union[torch.Tensor, None]: Tensor of output neuron indices or None.

        Raises:
            TypeError: If output_neurons is not a tensor, PathLike, or None.
        """
        output_neurons_tensor: torch.Tensor
        if output_neurons is not None:
            if isinstance(output_neurons, torch.Tensor):
                output_neurons_tensor = output_neurons
            else:
                output_neurons_tensor = torch.load(
                    output_neurons, weights_only=True
                ).squeeze()

            if output_neurons_tensor.dim() > 1:
                raise ValueError("Output indices must be a 1D tensor")

            return output_neurons_tensor
        else:
            return None

    def _clamp_taus(self) -> None:
        for tau in self.taus:
            tau.data = torch.clamp(tau, min=1.0)

    def forward(self, *args, **kwargs):
        """Forward pass of the ConnectomeRNN or ConnectomeODERNN.

        Args:
            *args: Arguments to pass to parent class forward method.
            **kwargs: Keyword arguments to pass to parent class forward method.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - First tensor is the output sequence. If output_neurons is None,
                  this is the full hidden state. Otherwise, it's a subset of the
                  hidden state corresponding to output_neurons.
                - Second tensor is the full hidden state sequence.
        """
        self._clamp_taus()

        if isinstance(self, ConnectomeRNN):
            hs = super(SparseRNN, self).forward(*args, **kwargs)  # type: ignore
        else:
            assert isinstance(self, ConnectomeODERNN)
            hs, ts = super(SparseODERNN, self).forward(*args, **kwargs)  # type: ignore

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        if isinstance(self, ConnectomeRNN):
            return outs, hs
        else:
            return outs, hs, ts


class ConnectomeRNN(ConnectomeRNNMixIn, SparseRNN):
    """Connectome Recurrent Neural Network.

    A recurrent neural network inspired by brain connectomes, using a sparse
    connectivity structure and biologically-inspired dynamics.

    Args:
        output_neurons (Union[torch.Tensor, PathLike], optional): Tensor or path
            to file containing indices of output neurons. Defaults to None.
        tau_init_fn (Union[str, TensorInitFnType], optional): Function or name
            of torch function to initialize time constants. Note that tau is
            clamped to be at least 1.0 after initialization, so it is
            recommended to use a function that initializes to values close to
            1.0, e.g. "ones" or "normal". Defaults to "ones".
        *args: Additional positional arguments to pass to the parent class.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    @torch.compile(dynamic=False, fullgraph=True, mode="max-autotune")
    def update_fn(
        self, x_t: torch.Tensor, h_t_minus_1: torch.Tensor
    ) -> torch.Tensor:
        """Update hidden state for one timestep.

        Args:
            x_t (torch.Tensor): Input at current timestep.
            h_t_minus_1 (torch.Tensor): Hidden state at previous timestep.

        Returns:
            torch.Tensor: Updated hidden state.
        """
        h_t = self.nonlinearity(self.ih(x_t) + self.hh(h_t_minus_1))
        return 1 / self.tau * h_t + (1 - 1 / self.tau) * h_t_minus_1


class ConnectomeODERNN(ConnectomeRNNMixIn, SparseODERNN):
    """Connectome Ordinary Differential Equation Recurrent Neural Network.

    A continuous-time version of the ConnectomeRNN, which integrates neural
    dynamics using an ODE solver.

    Args:
        output_neurons (Union[torch.Tensor, PathLike], optional): Tensor or path
            to file containing indices of output neurons. Defaults to None.
        tau_init_fn (Union[str, TensorInitFnType], optional): Function or name
            of function to initialize time constants. Defaults to "ones".
        *args: Additional positional arguments to pass to the parent class.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    @torch.compile(dynamic=False, fullgraph=True, mode="max-autotune")
    def update_fn(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ) -> torch.Tensor:
        """ODE function for neural dynamics.

        Args:
            t (torch.Tensor): Current time point.
            h (torch.Tensor): Current hidden state.
            args (Mapping[str, Any]): Additional arguments, including 'x' for
                the input at this time point.

        Returns:
            torch.Tensor: Rate of change of the hidden state (dh/dt).
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

        # Compute new hidden state
        h_new = self.nonlinearity(self.ih(x_t) + self.hh(h))

        # Rectify hidden state to ensure it's non-negative
        # Note: The user is still expected to provide a nonlinearity that
        #       ensures non-negativity, e.g. sigmoid.
        h_new = F.relu(h_new)

        # Compute rate of change of hidden state
        dhdt = (h_new - h) / self.tau

        return dhdt.t()
