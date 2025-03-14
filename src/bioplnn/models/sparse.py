import warnings
from collections.abc import Mapping
from os import PathLike
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch_sparse
import torchode as to

from bioplnn.typing import TensorInitFnType
from bioplnn.utils import get_activation, init_tensor


class SparseLinear(nn.Module):
    """Sparse linear layer for efficient operations with sparse matrices.

    This layer implements a sparse linear transformation, similar to nn.Linear,
    but operates on sparse matrices for memory efficiency.

    Args:
        in_features (int): Size of the input feature dimension.
        out_features (int): Size of the output feature dimension.
        connectivity (torch.Tensor): Sparse connectivity matrix in COO format.
        feature_dim (int, optional): Dimension on which features reside (0 for
            rows, 1 for columns). Defaults to -1 (unchanged).
        bias (bool, optional): If set to False, no bias term is added.
            Defaults to True.
        requires_grad (bool, optional): Whether the weight and bias parameters
            require gradient updates. Defaults to True.
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
        """Performs sparse linear transformation on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (H, *) if feature_dim is 0,
                otherwise (*, H).

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
    """Sparse Recurrent Neural Network (RNN) layer.

    Implements a RNN using sparse linear transformations.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state.
        connectivity_hh (Union[PathLike, torch.Tensor]): Connectivity matrix for
            hidden-to-hidden connections.
        connectivity_ih (Union[PathLike, torch.Tensor], optional): Connectivity
            matrix for input-to-hidden connections. If not provided, the
            input-to-hidden connections will be dense. Defaults to None.
        default_hidden_init_fn (str, optional): Initialization mode for the
            hidden state. Defaults to "zeros".
        use_layernorm (bool, optional): Whether to use layer normalization.
            Defaults to False.
        nonlinearity (str, optional): Nonlinearity function. Defaults to "Tanh".
        batch_first (bool, optional): Whether the input is in (batch_size,
            seq_len, input_size) format. Defaults to True.
        bias (bool, optional): Whether to use bias. Defaults to True.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        connectivity_hh: Union[PathLike, torch.Tensor],
        connectivity_ih: Optional[Union[PathLike, torch.Tensor]] = None,
        default_hidden_init_fn: str = "zeros",
        use_layernorm: bool = False,
        nonlinearity: str = "ReLU",
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.default_hidden_init_fn = default_hidden_init_fn
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
                in_features=input_size,
                out_features=hidden_size,
                connectivity=self.connectivity_ih,
                feature_dim=0,
                bias=False,
            )
        else:
            warnings.warn(
                "connectivity_ih is not provided and input_size does not "
                "match hidden_size, using a dense linear layer instead"
            )
            if input_size >= hidden_size:
                warnings.warn(
                    "input_size is greater than or equal to hidden_size, "
                    "the dense linear layer weight will have shape "
                    f"({input_size}, {hidden_size}). This may result in too "
                    "many parameters to fit on your GPU. Consider providing "
                    "connectivity_ih or decreasing input_size."
                )
            self.ih = nn.Linear(
                in_features=input_size,
                out_features=hidden_size,
                bias=False,
            )

        self.layernorm = (
            nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()
        )

    def _init_connectivity(
        self,
        connectivity_hh: Union[PathLike, torch.Tensor],
        connectivity_ih: Optional[Union[PathLike, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Initialize connectivity matrices.

        Args:
            connectivity_hh (Union[PathLike, torch.Tensor]): Connectivity matrix
                for hidden-to-hidden connections or path to load it from.
            connectivity_ih (Union[PathLike, torch.Tensor], optional):
                Connectivity matrix for input-to-hidden connections or path to
                load it from. Defaults to None.

        Returns:
            tuple[torch.Tensor, Union[torch.Tensor, None]]: Tuple containing the
                hidden-to-hidden connectivity tensor and input-to-hidden
                connectivity tensor (or None).

        Raises:
            ValueError: If connectivity matrices are not in COO format or have
                invalid dimensions.
        """
        connectivity_hh_tensor: torch.Tensor
        connectivity_ih_tensor: Union[torch.Tensor, None] = None

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
            self.input_size != connectivity_ih_tensor.shape[1]
            or self.hidden_size != connectivity_ih_tensor.shape[0]
        ):
            raise ValueError(
                "connectivity_ih.shape[1] and input_size must be equal"
            )

        return connectivity_hh_tensor, connectivity_ih_tensor

    def init_hidden(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """Initialize the hidden state.

        Args:
            batch_size (int): Batch size.
            init_fn (Union[str, TensorInitFnType], optional): Initialization
                function. Defaults to None (uses default_hidden_init_fn).
            device (Union[torch.device, str], optional): Device to allocate the
                hidden state on. Defaults to None.

        Returns:
            torch.Tensor: The initialized hidden state of shape
                (batch_size, hidden_size).
        """
        if init_fn is None:
            init_fn = self.default_hidden_init_fn

        return init_tensor(
            init_fn, batch_size, self.hidden_size, device=device
        )

    def init_state(
        self,
        num_steps: int,
        batch_size: int,
        h0: Optional[torch.Tensor] = None,
        hidden_init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> list[Optional[torch.Tensor]]:
        """Initialize the internal state of the network.

        Args:
            num_steps (int): Number of time steps.
            batch_size (int): Batch size.
            h0 (torch.Tensor, optional): Initial hidden states. Defaults to
                None (uses hidden_init_fn or default_hidden_init_fn).
            hidden_init_fn (Union[str, TensorInitFnType], optional):
                Initialization function. Defaults to None (uses
                default_hidden_init_fn).
            device (Union[torch.device, str], optional): Device to allocate
                tensors on. Defaults to None.

        Returns:
            list[Optional[torch.Tensor]]: The initialized hidden states for each
                time step.
        """
        hs: list[Optional[torch.Tensor]] = [None] * num_steps
        if h0 is None:
            h0 = self.init_hidden(
                batch_size,
                init_fn=hidden_init_fn,
                device=device,
            )
        hs[-1] = h0.t()
        return hs

    def _format_x(self, x: torch.Tensor, num_steps: Optional[int] = None):
        """Format the input tensor to match the expected shape.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,
                sequence_length, input_size) if batch_first, else
                (sequence_length, batch_size, input_size).
            num_steps (int, optional): Number of time steps. Defaults to None.

        Returns:
            tuple[torch.Tensor, int]: The formatted input tensor of shape
                (num_steps, batch_size, input_size) and the corrected number of
                time steps.

        Raises:
            ValueError: If x is 2D and num_steps is None or < 1, or if x is 3D
                and num_steps doesn't match sequence length.
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
                    "If x is 3D and num_steps is provided, it must match the "
                    "sequence length."
                )
            num_steps = x.shape[0]
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D, but got {x.dim()} dimensions."
            )
        return x, num_steps

    def _format_hs(
        self,
        hs: Union[torch.Tensor, list[torch.Tensor]],
    ) -> torch.Tensor:
        """Format the hidden states for output.

        Args:
            hs (Union[torch.Tensor, list[torch.Tensor]]): Hidden states for each
                time step.

        Returns:
            torch.Tensor: The formatted hidden states of shape
                (batch_size, num_steps, hidden_size) if batch_first, else
                (num_steps, batch_size, hidden_size).
        """
        if not isinstance(hs, torch.Tensor):
            hs = torch.stack(hs)

        if self.batch_first:
            return hs.permute(2, 0, 1)
        else:
            return hs.permute(0, 2, 1)

    def update_fn(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Update function for the SparseRNN.

        Computes the new hidden state based on input and previous hidden state.

        Args:
            x (torch.Tensor): Input tensor at current timestep.
            h (torch.Tensor): Hidden state from previous timestep.

        Returns:
            torch.Tensor: Updated hidden state.
        """
        return self.nonlinearity(self.ih(x) + self.hh(h))

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        h0: Optional[torch.Tensor] = None,
        hidden_init_fn: Optional[Union[str, TensorInitFnType]] = None,
    ) -> torch.Tensor:
        """Forward pass of the SparseRNN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,
                sequence_length, input_size) if batch_first, else
                (sequence_length, batch_size, input_size).
            num_steps (int, optional): Number of time steps. Defaults to None.
            h0 (torch.Tensor, optional): Initial hidden state of shape
                (batch_size, hidden_size). Defaults to None.
            hidden_init_fn (Union[str, TensorInitFnType], optional):
                Initialization function for the hidden state. Defaults to None.

        Returns:
            torch.Tensor: Hidden states of shape (batch_size, num_steps,
                hidden_size) if batch_first, else (num_steps, batch_size,
                hidden_size).
        """

        x, num_steps = self._format_x(x, num_steps)
        batch_size = x.shape[-1]
        device = x.device

        hs = self.init_state(
            num_steps,
            batch_size,
            h0=h0,
            hidden_init_fn=hidden_init_fn,
            device=device,
        )

        for t in range(num_steps):
            hs[t] = self.update_fn(x[t], hs[t - 1])  # type: ignore

        assert all(h is not None for h in hs)

        return self._format_hs(hs)  # type: ignore


class SparseODERNN(SparseRNN):
    """Sparse Ordinary Differential Equation Recurrent Neural Network.

    A continuous-time version of SparseRNN that uses an ODE solver to compute
    neural dynamics.

    Args:
        compile_solver_kwargs (Mapping[str, Any], optional): Keyword arguments
            to pass to torch.compile for the ODE solver. Defaults to None.
        *args: Additional positional arguments to pass to the parent class.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(
        self,
        *args,
        compile_solver_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Define ODE solver
        term = to.ODETerm(self.update_fn, with_args=True)  # type: ignore
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(
            atol=1e-6, rtol=1e-3, term=term
        )
        self.solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # type: ignore

        # Compile solver
        if compile_solver_kwargs is not None:
            print(f"Compiling solver with kwargs: {compile_solver_kwargs}")
            self.solver = torch.compile(self.solver, **compile_solver_kwargs)

    def _format_x(self, x: torch.Tensor):
        """Format the input tensor to match the expected shape.

        Args:
            x (torch.Tensor): Input tensor. If 2-dimensional, it is assumed
                to be of shape (batch_size, input_size). If 3-dimensional,
                it is assumed to be of shape (batch_size, sequence_length,
                input_size) if batch_first, else (sequence_length, batch_size,
                input_size).

        Returns:
            torch.Tensor: The formatted input tensor of shape
                (sequence_length, input_size, batch_size).

        Raises:
            ValueError: If input tensor is not 2D or 3D.
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

    def _format_ts(self, ts: torch.Tensor) -> torch.Tensor:
        """Format the time points based on batch_first setting.

        Args:
            ts (torch.Tensor): Time points tensor.

        Returns:
            torch.Tensor: Formatted time points tensor.
        """
        if self.batch_first:
            return ts
        else:
            return ts.transpose(0, 1)

    def _index_from_time(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        start_time: float,
        end_time: float,
    ) -> torch.Tensor:
        """Calculate the index of the input tensor corresponding to the given time.

        Args:
            t (torch.Tensor): Current time point.
            x (torch.Tensor): Input tensor.
            start_time (float): Start time for simulation.
            end_time (float): End time for simulation.

        Returns:
            torch.Tensor: Index tensor for selecting the correct input.
        """
        idx = (t - start_time) / (end_time - start_time) * x.shape[0]
        idx[idx == x.shape[0]] = x.shape[0] - 1

        return idx.long()

    def update_fn(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ) -> torch.Tensor:
        """ODE function for the SparseODERNN.

        Args:
            t (torch.Tensor): Current time point.
            h (torch.Tensor): Current hidden state.
            args (Mapping[str, Any]): Additional arguments including input 'x',
                'start_time', and 'end_time'.

        Returns:
            torch.Tensor: Rate of change of the hidden state (dh/dt).
        """
        h = h.t()
        x = args["x"]
        start_time = args["start_time"]
        end_time = args["end_time"]

        idx = self._index_from_time(t, x, start_time, end_time)

        h_new = self.nonlinearity(self.ih(x[idx]) + self.hh(h))
        h_new = self.layernorm(h_new)

        dhdt = h_new - h

        return dhdt.t()

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int,
        start_time: float = 0.0,
        end_time: float = 1.0,
        h0: Optional[torch.Tensor] = None,
        hidden_init_fn: Optional[Union[str, TensorInitFnType]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the SparseODERNN layer.

        Solves the initial value problem for the ODE defined by update_fn.

        Args:
            x (torch.Tensor): Input tensor.
            num_steps (int): Number of time steps.
            start_time (float, optional): Start time for simulation.
                Defaults to 0.0.
            end_time (float, optional): End time for simulation.
                Defaults to 1.0.
            h0 (torch.Tensor, optional): Initial hidden state. Defaults to None.
            hidden_init_fn (Union[str, TensorInitFnType], optional):
                Initialization function for the hidden state. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Hidden states of shape (batch_size, num_steps, hidden_size) if
                  batch_first, else (num_steps, batch_size, hidden_size)
                - Time points of shape (batch_size, num_steps) if batch_first,
                  else (num_steps, batch_size)

        Raises:
            ValueError: If num_steps is less than 2.
        """
        # Format input and initialize variables
        x = self._format_x(x)
        batch_size = x.shape[-1]
        device = x.device

        # Define evaluation time points
        if num_steps < 2:
            raise ValueError("num_steps must be greater than 1")
        t_eval = (
            torch.linspace(start_time, end_time, num_steps, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Initialize hidden state
        if h0 is None:
            h0 = self.init_hidden(
                batch_size,
                init_fn=hidden_init_fn,
                device=device,
            )

        # Solve ODE
        problem = to.InitialValueProblem(y0=h0, t_eval=t_eval)  # type: ignore
        sol = self.solver.solve(
            problem,
            args={
                "x": x,
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        hs = sol.ys.permute(1, 2, 0)
        assert hs.shape == (num_steps, self.hidden_size, batch_size)

        # Format outputs
        hs = self._format_hs(hs)
        ts = self._format_ts(sol.ts)

        return hs, ts
