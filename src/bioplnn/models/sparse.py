from collections.abc import Mapping
from os import PathLike
from typing import Any, Optional

import torch
import torch.nn as nn
import torch_sparse
import torchode as to

from bioplnn.utils import get_activation


class SparseLinear(nn.Module):
    """
    Sparse linear layer for efficient operations with sparse matrices.

    This layer implements a sparse linear transformation, similar to nn.Linear,
    but operates on sparse matrices for memory efficiency.

    Args:
        in_features (int): Size of the input feature dimension.
        out_features (int): Size of the output feature dimension.
        connectivity (torch.Tensor): Sparse connectivity matrix in COO format.
        sparse_format (str, optional): Format of the sparse matrix ('torch_sparse', 'coo', or 'csr'). Defaults to "torch_sparse".
        mm_function (str, optional): Matrix multiplication function to use ('torch_sparse', 'native', or 'tsgu'). Defaults to "torch_sparse".
        feature_dim (int, optional): Dimension on which features reside (0 for rows, 1 for columns). Defaults to -1 (unchanged).
        bias (bool, optional): If set to False, no bias term is added. Defaults to True.
        requires_grad (bool, optional): Whether the weight and bias parameters require gradient updates. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity: torch.Tensor,
        feature_dim: int = -1,
        bias: bool = True,
        requires_grad: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.feature_dim = feature_dim

        # Validate connectivity format
        if connectivity.layout != torch.sparse_coo:
            raise ValueError("connectivity must be in COO format.")

        # Validate input and output sizes against connectivity
        if in_features != connectivity.shape[1]:
            raise ValueError(
                f"Input size ({in_features}) must be equal to the number of columns in connectivity ({connectivity.shape[1]})."
            )
        if out_features != connectivity.shape[0]:
            raise ValueError(
                f"Output size ({out_features}) must be equal to the number of rows in connectivity ({connectivity.shape[0]})."
            )

        # Create sparse matrix
        indices: torch.Tensor
        values: torch.Tensor
        indices, values = torch_sparse.coalesce(
            connectivity.indices().clone(),
            connectivity.values().clone(),
            self.out_features,
            self.in_features,
        )  # type: ignore

        self.indices = nn.Parameter(indices, requires_grad=False)
        self.values = nn.Parameter(values.float(), requires_grad=requires_grad)

        self.bias = (
            nn.Parameter(
                torch.zeros(self.out_features, 1), requires_grad=requires_grad
            )
            if bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs sparse linear transformation on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (H, *) if feature_dim is 0, otherwise (*, H).

        Returns:
            torch.Tensor: Output tensor after sparse linear transformation.
        """
        shape = list(x.shape)

        if self.feature_dim != 0:
            permutation = torch.arange(x.dim())
            permutation[self.feature_dim] = 0
            permutation[0] = self.feature_dim
            x = x.permute(*permutation)  # type: ignore

        x = x.flatten(start_dim=1)

        x = torch_sparse.spmm(
            self.indices,
            self.values,
            self.out_features,
            self.in_features,
            x,
        )

        if self.bias is not None:
            x = x + self.bias

        if self.feature_dim != 0:
            x = x.permute(*permutation)  # type: ignore

        shape[self.feature_dim] = self.out_features
        x = x.view(*shape)

        return x


class SparseRNN(nn.Module):
    """
    Sparse Recurrent Neural Network (RNN) layer.

    Implements a RNN using sparse linear transformations.

    Args:
        in_size int: Size of the input.
        hidden_size int: Size of the hidden state.
        connectivity_ih torch.Tensor: Connectivity matrix for input-to-hidden connections.
        connectivity_hh torch.Tensor: Connectivity matrix for hidden-to-hidden connections.
        batch_first (bool, optional): Whether the input is in (batch_size, seq_len, in_size) format. Defaults to True.
        use_layernorm (bool, optional): Whether to use layer normalization. Defaults to True.
        nonlinearity (str, optional): Nonlinearity function. Defaults to "tanh".
        bias (bool, optional): Whether to use bias. Defaults to True.
    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        connectivity_hh: PathLike | torch.Tensor,
        connectivity_ih: Optional[PathLike | torch.Tensor] = None,
        default_hidden_init_mode: str = "zeros",
        use_layernorm: bool = False,
        nonlinearity: str = "tanh",
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.default_hidden_init_mode = default_hidden_init_mode
        self.nonlinearity = get_activation(nonlinearity)
        self.batch_first = batch_first

        self.connectivity_hh, self.connectivity_ih = self._init_connectivity(
            connectivity_hh, connectivity_ih
        )

        self.hh = SparseLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            connectivity=self.connectivity_hh,
            feature_dim=0,
            bias=bias,
        )

        if self.connectivity_ih is not None:
            self.ih = SparseLinear(
                in_features=in_size,
                out_features=hidden_size,
                connectivity=self.connectivity_ih,
                feature_dim=0,
                bias=False,
            )
        else:
            self.ih = nn.Linear(
                in_features=in_size,
                out_features=hidden_size,
                bias=False,
            )

        self.layernorm = (
            nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()
        )

    def _init_connectivity(
        self,
        connectivity_hh: PathLike | torch.Tensor,
        connectivity_ih: Optional[PathLike | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        connectivity_hh_tensor: torch.Tensor
        connectivity_ih_tensor: torch.Tensor | None = None

        if isinstance(connectivity_hh, torch.Tensor):
            connectivity_hh_tensor = connectivity_hh
        else:
            connectivity_hh_tensor = torch.load(
                connectivity_hh, weights_only=True
            )

        if connectivity_ih is not None:
            if isinstance(connectivity_ih, torch.Tensor):
                connectivity_ih_tensor = connectivity_ih
            else:
                connectivity_ih_tensor = torch.load(
                    connectivity_ih, weights_only=True
                )

        # Validate connectivity matrix format
        if connectivity_hh_tensor.layout != torch.sparse_coo or (
            connectivity_ih_tensor is not None
            and connectivity_ih_tensor.layout != torch.sparse_coo
        ):
            raise ValueError("Connectivity matrices must be in COO format")

        # Validate connectivity matrix dimensions
        if not (
            self.hidden_size
            == connectivity_hh_tensor.shape[0]
            == connectivity_hh_tensor.shape[1]
        ):
            raise ValueError(
                "connectivity_ih.shape[0], connectivity_hh.shape[0], and connectivity_hh.shape[1] must be equal"
            )

        if connectivity_ih_tensor is not None and (
            self.in_size != connectivity_ih_tensor.shape[1]
            or self.hidden_size != connectivity_ih_tensor.shape[0]
        ):
            raise ValueError(
                "connectivity_ih.shape[1] and in_size must be equal"
            )

        return connectivity_hh_tensor, connectivity_ih_tensor

    def _init_hidden(
        self,
        batch_size: int,
        init_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Initializes the hidden state.

        Args:
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros' or 'normal'. Defaults to 'zeros'.
            device (torch.device, optional): Device to allocate the hidden state on. Defaults to None.

        Returns:
            torch.Tensor: The initialized hidden state.
        """

        return getattr(
            torch,
            init_mode
            if init_mode is not None
            else self.default_hidden_init_mode,
        )(batch_size, self.hidden_size, device=device)

    def _init_state(
        self,
        h_0: Optional[torch.Tensor],
        num_steps: int,
        batch_size: int,
        init_mode: Optional[str] = None,
        device=None,
    ) -> list[torch.Tensor | None]:
        """
        Initializes the internal state of the network.

        Args:
            h_0 (Optional[torch.Tensor]): Initial hidden states for each layer.
            num_steps (int): Number of time steps.
            batch_size (int): Batch size.
            init_mode (str, optional): Initialization mode. Must be 'zeros', 'ones', 'randn', or 'rand'. Defaults to 'zeros'.
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            list[Optional[torch.Tensor]]: The initialized hidden states for each time step.
        """
        hs: list[torch.Tensor | None] = [None] * num_steps

        if h_0 is None:
            hs[-1] = self._init_hidden(
                batch_size,
                init_mode=init_mode,
                device=device,
            ).t()
        else:
            hs[-1] = h_0.t()

        return hs

    def _format_x(self, x: torch.Tensor, num_steps: Optional[int] = None):
        """
        Formats the input tensor to match the expected shape.

        Args:
            x (torch.Tensor): Input tensor.
            num_steps (Optional[int]): Number of time steps.

        Returns:
            tuple(torch.Tensor, int): The formatted input tensor and the number of time steps.
        """
        if x.dim() == 2:
            if num_steps is None or num_steps < 1:
                raise ValueError(
                    "If x is 2D, num_steps must be provided and greater than 0"
                )
            x = x.t()
            x = x.unsqueeze(0).expand((num_steps, -1, -1))
        elif x.dim() == 3:
            if self.batch_first:
                x = x.permute(1, 2, 0)
            else:
                x = x.permute(0, 2, 1)
            if num_steps is not None and num_steps != x.shape[0]:
                raise ValueError(
                    "If x is 3D and num_steps is provided, it must match the sequence length."
                )
            num_steps = x.shape[0]
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D, but got {x.dim()} dimensions."
            )
        return x, num_steps

    def _format_result(self, hs: list[torch.Tensor | None]) -> torch.Tensor:
        """
        Formats the hidden states.

        Args:
            hs (list[torch.Tensor]): Hidden states for each layer and time step.

        Returns:
            torch.Tensor: The formatted hidden states.
        """
        assert all(h is not None for h in hs)
        h = torch.stack(hs)  # type: ignore
        if self.batch_first:
            h = h.permute(2, 0, 1)
        else:
            h = h.permute(0, 2, 1)

        return h

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        h_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the SparseRNN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, in_size) if batch_first, else (sequence_length, batch_size, in_size)
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

        # Stack outputs and adjust dimensions if necessary
        hs = self._format_result(hs)

        return hs


class SparseRNNODE(SparseRNN):
    def __init__(self, *args, compile: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_compile = compile

    def _format_x(self, x: torch.Tensor):
        """
        Formats the input tensor to match the expected shape.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The formatted input tensor.
        """
        if x.dim() == 2:
            x = x.t()
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            if self.batch_first:
                x = x.permute(1, 2, 0)
            else:
                x = x.permute(0, 2, 1)
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D, but got {x.dim()} dimensions."
            )
        return x

    def term_ode(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ):
        h = h.transpose(0, 1)
        x = args["x"]
        start_time = args["start_time"]
        end_time = args["end_time"]

        idx = int((t - start_time) / (end_time - start_time) * x.shape[0])
        if idx == x.shape[0]:
            idx = x.shape[0] - 1

        h = self.nonlinearity(self.ih(x[idx]) + self.hh(h))

        return h

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

        x = self._format_x(x)

        if num_steps == 1:
            t_eval = torch.tensor(end_time, device=device).unsqueeze(0)
        else:
            t_eval = (
                torch.linspace(start_time, end_time, num_steps)
                .unsqueeze(0)
                .to(device)
            )

        term = to.ODETerm(self.term_ode, with_args=True)  # type: ignore
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(
            atol=1e-6, rtol=1e-3, term=term
        )
        solver = to.AutoDiffAdjoint(step_method, step_size_controller).to(  # type: ignore
            device
        )
        if self.do_compile:
            solver = torch.compile(solver)

        # Solve ODE
        problem = to.InitialValueProblem(y0=h0, t_eval=t_eval)  # type: ignore
        sol = solver.solve(
            problem,
            args={"x": x, "start_time": start_time, "end_time": end_time},
        )

        ys = sol.ys.transpose(0, 1)

        # Stack outputs and adjust dimensions if necessary
        hs = self._format_result(ys)  # type: ignore

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
