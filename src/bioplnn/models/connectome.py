from collections.abc import Mapping
from os import PathLike
from typing import Any, Optional

import torch
import torch.nn.functional as F

from bioplnn.models.sparse import SparseODERNN, SparseRNN
from bioplnn.typing import TensorInitFnType
from bioplnn.utils import init_tensor


class ConnectomeRNN(SparseRNN):
    """
    Connectome Recurrent Neural Network

    ConnectomeRNN is a recurrent neural network designed to
    This base class provides common functionalities for all ConnectomeRNN variants.

    Args:
        sheet_size (tuple[int, int]): Size of the sheet-like topology (height, width).
        synapse_std (float): Standard deviation for random synapse initialization.
        synapses_per_neuron (int): Number of synapses per neuron.
        self_recurrence (bool): Whether to include self-recurrent connections.
        connectivity_hh (Optional[str | torch.Tensor]): Path to a file containing the hidden-to-hidden connectivity matrix or the matrix itself.
        connectivity_ih (Optional[str | torch.Tensor]): Path to a file containing the input-to-hidden connectivity matrix or the matrix itself.
        num_classes (int): Number of output classes.
        batch_first (bool): Whether the input is in (batch_size, seq_len, input_size) format.
        input_indices (Optional[str | torch.Tensor]): Path to a file containing the input indices or the tensor itself (specifying which neurons receive input).
        output_indices (Optional[str | torch.Tensor]): Path to a file containing the output indices or the tensor itself (specifying which neurons contribute to the output).
        out_nonlinearity (str): Nonlinearity applied to the output layer.
    """

    def __init__(
        self,
        *args,
        output_neurons: Optional[torch.Tensor | PathLike] = None,
        tau_init_fn: str | TensorInitFnType = "ones",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.output_neurons = self._init_output_neurons(output_neurons)

        # Time constant
        self.tau = init_tensor(tau_init_fn, 1, self.num_neurons)
        self._tau_hook(self, None)
        self.register_forward_pre_hook(self._tau_hook)

    @staticmethod
    def _tau_hook(module, args):
        module.tau.data = F.softplus(module.tau) + 1

    def _init_output_neurons(
        self,
        output_neurons: Optional[torch.Tensor | PathLike] = None,
    ) -> torch.Tensor | None:
        """
        Initialize the output neurons.
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
        """
        Update function for the ConnectomeRNN.
        """
        h_t = self.layernorm(
            self.nonlinearity(self.ih(x_t) + self.hh(h_t_minus_1))
        )
        assert (self.tau > 1).all()
        return 1 / self.tau * h_t + (1 - 1 / self.tau) * h_t_minus_1

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConnectomeRNN layer.

        Passes all arguments and keyword arguments to SparseRNN's forward
        method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output and hidden state for each
                time step.
        """

        hs = super().forward(*args, **kwargs)

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        return outs, hs


class ConnectomeODERNN(SparseODERNN):
    def update_fn(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ) -> torch.Tensor:
        """
        ODE term for the ConnectomeODERNN.

        Args:
            t (torch.Tensor): Time points.
            h (torch.Tensor): Hidden states.
            args (Mapping[str, Any]): Additional arguments.

        Returns:
            torch.Tensor: dh/dt
        """

        h = h.t()
        x = args["x"]
        start_time = args["start_time"]
        end_time = args["end_time"]

        idx = self._index_from_time(t, x, start_time, end_time)

        h_new = self.nonlinearity(self.ih(x[idx]) + self.hh(h))
        h_new = self.layernorm(h_new)

        assert (self.tau > 1).all()
        dhdt = (h_new - h) / self.tau

        return dhdt

    def forward(
        self, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConnectomeODERNN layer.

        Passes all arguments and keyword arguments to SparseODERNN's forward
        method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, hidden
                state and time steps.
        """

        hs, ts = super().forward(*args, **kwargs)

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        return outs, hs, ts
