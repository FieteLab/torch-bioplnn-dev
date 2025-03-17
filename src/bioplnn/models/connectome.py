from collections.abc import Mapping, Sequence
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
    load_tensor,
)


class ConnectomeRNNMixIn:
    """MixIn for ConnectomeRNN and ConnectomeODERNN.

    This class is used for common initialization and forward methods for
    ConnectomeRNN and ConnectomeODERNN. It is not intended to be used directly.
    It MUST be the first class in the MRO of ConnectomeRNN or ConnectomeODERNN
    (besides itself).

    Args:
        num_neuron_types (int, optional): Number of neuron types.
            Defaults to None.
        neuron_type_class (Sequence[str], optional): Class of each
            neuron type. Can be "excitatory" or "inhibitory". Defaults to
            None.
        neuron_type_indices (Sequence[Sequence[int]], optional): Indices of
            neurons for each neuron type. Defaults to None.
        neuron_type_nonlinearity (Sequence[Union[str, nn.Module]], optional):
            Nonlinearity for each neuron type. Defaults to None.
        neuron_type_tau_init (Sequence[float], optional): Initial
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
        num_neuron_types: int = 0,
        neuron_type_class: NeuronTypeParam[str] = "excitatory",
        neuron_type_indices: Optional[
            Sequence[Union[torch.Tensor, PathLike]]
        ] = None,
        neuron_type_nonlinearity: NeuronTypeParam[
            Optional[Union[str, nn.Module]]
        ] = "Sigmoid",
        neuron_type_tau_init: NeuronTypeParam[float] = 1.0,
        train_tau: bool = True,
        default_tau_init_fn: Union[str, TensorInitFnType] = "ones",
        **kwargs,
    ):
        # Initialize parent class
        assert isinstance(self, (ConnectomeRNN, ConnectomeODERNN))
        super().__init__(*args, **kwargs)

        self.output_neurons = self._load_output_neurons(output_neurons)

        # Neuron type parameters
        self.num_neuron_types = num_neuron_types
        if num_neuron_types > 0:
            ##############################################################
            #  Neuron type indices
            ##############################################################

            if (
                neuron_type_indices is None
                or len(neuron_type_indices) != num_neuron_types
            ):
                raise ValueError(
                    "neuron_type_indices must be provided and have length "
                    f"equal to num_neuron_types ({num_neuron_types})."
                )

            indices_tensors = self._load_indices_tensors(neuron_type_indices)
            self.neuron_type_indices = nn.ParameterList()
            type_mask = torch.zeros(self.hidden_size, dtype=torch.int)
            for i in range(num_neuron_types):
                indices_tensor = indices_tensors[i]
                type_mask[indices_tensor] = i
                self.neuron_type_indices.append(
                    nn.Parameter(indices_tensor, requires_grad=False)
                )

            self.neuron_type_mask = nn.Parameter(
                type_mask, requires_grad=False
            )

            ##############################################################
            #  Neuron type class
            ##############################################################

            self.neuron_type_class = expand_list(
                neuron_type_class, num_neuron_types
            )
            check_possible_values(
                "neuron_type_class",
                self.neuron_type_class,
                ("excitatory", "inhibitory"),
            )
            sign_mask = torch.zeros(self.hidden_size, dtype=torch.bool)
            for i in range(num_neuron_types):
                sign_mask[self.neuron_type_indices[i]] = (
                    1 if self.neuron_type_class[i] == "excitatory" else -1
                )
            self.neuron_sign_mask = nn.Parameter(
                sign_mask.unsqueeze(-1), requires_grad=False
            )

            ##############################################################
            #  Neuron type nonlinearity
            ##############################################################

            neuron_type_nonlinearity = expand_list(
                neuron_type_nonlinearity, num_neuron_types
            )
            self.neuron_type_nonlinearity = nn.ModuleList(
                [
                    get_activation(nonlinearity)
                    for nonlinearity in neuron_type_nonlinearity
                ]
            )

            ##############################################################
            #  Neuron type time constants
            ##############################################################

            self.neuron_type_tau_init = expand_list(
                neuron_type_tau_init, num_neuron_types
            )
            taus = []
            for i in range(num_neuron_types):
                tau = torch.tensor(
                    self.neuron_type_tau_init[i], dtype=torch.float
                )
                if tau < 1.0:
                    raise ValueError(
                        "neuron_type_tau_init must be at least 1.0."
                    )
                if tau == 1.0:
                    tau += torch.rand_like(tau) * 1e-6
                taus.append(tau)
            self.tau = nn.Parameter(
                torch.stack(taus).unsqueeze(-1), requires_grad=train_tau
            )
        else:
            tau = init_tensor(
                default_tau_init_fn, self.hidden_size, 1, dtype=torch.float
            )
            noise = torch.rand_like(tau) * 1e-6
            self.tau = nn.Parameter(tau + noise, requires_grad=train_tau)

    def _load_output_neurons(
        self,
        output_neurons: Optional[Union[torch.Tensor, PathLike]],
    ) -> Optional[torch.Tensor]:
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

        if output_neurons is None:
            return None

        output_neurons = load_tensor(output_neurons)
        if output_neurons.dim() > 1:
            raise ValueError("Output indices must be a 1D tensor")
        return output_neurons

    def _load_indices_tensors(
        self,
        indices_tensors: Sequence[Union[torch.Tensor, PathLike]],
    ) -> Sequence[torch.Tensor]:
        """Load index tensors.

        Args:
            index_tensors (Sequence[Union[torch.Tensor, PathLike]]): Sequence of
                tensors or paths to files containing indices.

        Returns:
            Sequence[torch.Tensor]: Sequence of loaded index tensors.
        """
        loaded_tensors = []
        for t in indices_tensors:
            loaded_tensor = load_tensor(t)
            if loaded_tensor.dim() > 1:
                raise ValueError("Index tensors must be a 1D tensor")
            loaded_tensors.append(loaded_tensor)

        # Check that all indices are present
        tensors_cat = torch.cat(loaded_tensors)
        if (
            tensors_cat.shape[0] != self.hidden_size  # type: ignore
            or tensors_cat.unique().shape[0] != self.hidden_size  # type: ignore
        ):
            raise ValueError(
                "the neuron_type_indices must collectively contain all "
                "neuron indices"
            )

        return loaded_tensors

    def _clamp_tau(self) -> None:
        self.tau.data.clamp_(min=1.0)

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
        self._clamp_tau()

        res = super().forward(*args, **kwargs)  # type: ignore
        if isinstance(self, ConnectomeRNN):
            hs = res
        else:
            hs, ts = res

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        if isinstance(self, ConnectomeRNN):
            return outs, hs
        else:
            return outs, hs, ts

    def update_fn(self, *args, **kwargs):
        raise NotImplementedError(
            "update_fn must be implemented by a subclass of ConnectomeRNNMixIn"
        )


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
    def _update_fn_no_neuron_types(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ) -> torch.Tensor:
        """ODE function for neural dynamics without neuron types."""
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

    @torch.compile(dynamic=False, fullgraph=True, mode="max-autotune")
    def _update_fn_with_neuron_types(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ) -> torch.Tensor:
        """ODE function for neural dynamics with neuron types.

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

        # Apply sign mask to input
        h_signed = h * self.neuron_sign_mask

        # Compute new hidden state
        h_new = self.ih(x_t) + self.hh(h_signed)
        for i in range(self.num_neuron_types):
            h_new[self.neuron_type_indices[i]] = self.neuron_type_nonlinearity[
                i
            ](h_new[self.neuron_type_indices[i]])

        # Rectify hidden state to ensure it's non-negative
        # Note: The user is still expected to provide a nonlinearity that
        #   ensures non-negativity, e.g. sigmoid.
        h_new = F.relu(h_new)

        # Compute rate of change of hidden state
        dhdt = (h_new - h) / self.tau[self.neuron_type_mask, :]

        return dhdt.t()

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

        if self.num_neuron_types > 0:
            return self._update_fn_with_neuron_types(t, h, args)
        else:
            return self._update_fn_no_neuron_types(t, h, args)
