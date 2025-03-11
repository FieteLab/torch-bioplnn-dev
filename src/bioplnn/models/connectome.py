from os import PathLike
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchode as to

from bioplnn.models.sparse import SparseRNN


class ConnectomeRNN(SparseRNN):
    """
    Base class for Topographical Recurrent Neural Networks (TRNNs).

    TRNNs are a type of recurrent neural network designed to model spatial dependencies on a sheet-like topology.
    This base class provides common functionalities for all TRNN variants.

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
        in_size: int,
        num_neurons: int,
        connectivity_hh: PathLike | torch.Tensor,
        connectivity_ih: Optional[PathLike | torch.Tensor] = None,
        output_neurons: Optional[torch.Tensor | PathLike] = None,
        default_hidden_init_mode: str = "zeros",
        nonlinearity: str = "tanh",
        use_layernorm: bool = False,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__(
            in_size=in_size,
            hidden_size=num_neurons,
            connectivity_ih=connectivity_ih,
            connectivity_hh=connectivity_hh,
            use_layernorm=use_layernorm,
            nonlinearity=nonlinearity,
            default_hidden_init_mode=default_hidden_init_mode,
            batch_first=batch_first,
            bias=bias,
        )

        self.output_neurons = self._init_output_neurons(output_neurons)

        # Time constant
        self.tau = nn.Parameter(
            torch.ones(self.num_neurons), requires_grad=True
        )
        self._tau_hook(self, None)
        self.register_forward_pre_hook(self._tau_hook)

    @staticmethod
    def _tau_hook(module, args):
        module.tau.data = F.softplus(module.tau) + 1

    def _init_output_neurons(
        self,
        output_neurons: Optional[torch.Tensor | PathLike] = None,
    ) -> torch.Tensor | None:
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

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        h_0: Optional[torch.Tensor] = None,
        loss_all_timesteps: bool = False,
    ):
        """
         Forward pass of the SparseRNN layer.

        Args:
             x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size) if batch_first, else (sequence_length, batch_size, input_size)
             num_steps (int, optional): Number of time steps. Defaults to None.
             h_0 (torch.Tensor, optional): Initial hidden state of shape (batch_size, hidden_size). Defaults to None.

         Returns:
             torch.Tensor: Output tensor.
        """
        device = x.device

        x, num_steps = self._format_x(x, num_steps)

        batch_size = x.shape[-1]

        hs = self._init_state(
            h_0,
            num_steps,
            batch_size,
            device=device,
        )
        # Process input sequence
        for t in range(num_steps):
            hs[t] = self.nonlinearity(self.ih(x[t]) + self.hh(hs[t - 1]))
            hs[t] = self.layernorm(hs[t])
            assert self.tau > 1
            hs[t] = 1 / self.tau * hs[t] + (1 - 1 / self.tau) * hs[t - 1]

        # Stack outputs and adjust dimensions if necessary
        hs = self._format_result(hs)

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        if loss_all_timesteps:
            return outs, hs
        elif self.batch_first:
            return outs[:, -1], hs
        else:
            return outs[-1], hs


class ConnectomeODERNN(ConnectomeRNN):
    def _forward(self, t: torch.Tensor, h: torch.Tensor, x: torch.Tensor):
        h = h.transpose(0, 1)

        x_t = self._format_x_ode(x)

        h_new = self.nonlinearity(self.ih(x_t) + self.hh(h))
        h_new = self.layernorm(h_new)

        assert self.tau > 1
        dhdt = 1 / self.tau * (h_new - h)

        return dhdt

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor,
        num_steps: int,
        start_time: float = 0.0,
        end_time: float = 1.0,
        return_activations: bool = False,
        loss_all_timesteps: bool = False,
    ):
        device = x.device

        if num_steps == 1:
            t_eval = torch.tensor(end_time, device=device).unsqueeze(0)
        else:
            t_eval = (
                torch.linspace(start_time, end_time, num_steps)
                .unsqueeze(0)
                .to(device)
            )

        term = to.ODETerm(self._forward, with_args=True)  # type: ignore
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(
            atol=1e-6, rtol=1e-3, term=term
        )
        solver = to.AutoDiffAdjoint(step_method, step_size_controller).to(  # type: ignore
            device
        )
        # Solve ODE
        problem = to.InitialValueProblem(y0=h0, t_eval=t_eval)  # type: ignore
        sol = solver.solve(
            problem,
            args=x,
        )
        ys = sol.ys.transpose(0, 1)

        # Stack outputs and adjust dimensions if necessary
        hs = self._format_result(ys)  # type: ignore

        # Select output indices if provided
        if self.output_neurons is not None:
            outs = hs[..., self.output_neurons]
        else:
            outs = hs

        if not loss_all_timesteps:
            if self.batch_first:
                outs = outs[:, -1]
            else:
                outs = outs[-1]

        if return_activations:
            return outs, hs
        else:
            return outs
