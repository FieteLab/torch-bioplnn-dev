from collections.abc import Mapping
from os import PathLike
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplnn.models.sparse import SparseODERNN, SparseRNN
from bioplnn.typing import TensorInitFnType
from bioplnn.utils import init_tensor


class ConnectomeRNN(SparseRNN):
    """Connectome Recurrent Neural Network.

    A recurrent neural network inspired by brain connectomes, using a sparse
    connectivity structure and biologically-inspired dynamics.

    Args:
        output_neurons (Union[torch.Tensor, PathLike], optional): Tensor or path
            to file containing indices of output neurons. Defaults to None.
        tau_init_fn (Union[str, TensorInitFnType], optional): Function or name
            of function to initialize time constants. Defaults to "ones".
        *args: Additional positional arguments to pass to the parent class.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(
        self,
        *args,
        output_neurons: Optional[Union[torch.Tensor, PathLike]] = None,
        tau_init_fn: Union[str, TensorInitFnType] = "ones",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.output_neurons = self._init_output_neurons(output_neurons)

        # Time constant
        self.tau = init_tensor(tau_init_fn, 1, self.hidden_size)
        self.tau = nn.Parameter(self.tau)

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
        h_t = self.layernorm(
            self.nonlinearity(self.ih(x_t) + self.hh(h_t_minus_1))
        )
        assert (self.tau > 1).all()
        return 1 / self.tau * h_t + (1 - 1 / self.tau) * h_t_minus_1

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the ConnectomeRNN.

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
        hs = super().forward(*args, **kwargs)

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        return outs, hs


class ConnectomeODERNN(SparseODERNN):
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

    def __init__(
        self,
        *args,
        output_neurons: Optional[Union[torch.Tensor, PathLike]] = None,
        tau_init_fn: Union[str, TensorInitFnType] = "ones",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.output_neurons = self._init_output_neurons(output_neurons)

        # Time constant
        self.tau = init_tensor(tau_init_fn, self.hidden_size, 1)
        self.tau = nn.Parameter(self.tau)

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
        # Note: The use is still expected to provide a nonlinearity that
        #       ensures non-negativity, e.g. sigmoid.
        h_new = F.relu(h_new)

        # Compute rate of change of hidden state
        dhdt = (h_new - h) / self.tau

        return dhdt.t()

    def forward(
        self, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the ConnectomeODERNN.

        Args:
            *args: Arguments to pass to parent class forward method.
            **kwargs: Keyword arguments to pass to parent class forward method.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - First tensor is the output sequence. If output_neurons is None,
                  this is the full hidden state. Otherwise, it's a subset of the
                  hidden state corresponding to output_neurons.
                - Second tensor is the full hidden state sequence.
                - Third tensor contains the time points.
        """
        # Ensure tau is at least 1
        self.tau.data = torch.clamp(self.tau, min=1)

        hs, ts = super().forward(*args, **kwargs)

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        return outs, hs, ts
