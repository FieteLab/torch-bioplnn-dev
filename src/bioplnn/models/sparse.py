import warnings
from collections.abc import Mapping
from os import PathLike
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch_sparse
import torchode as to

from bioplnn.typing import TensorInitFnType
from bioplnn.utils import get_activation, init_tensor, load_sparse_tensor

# TODO: Some docstrings may be outdated, might need to update


class SparseLinear(nn.Module):
    """Sparse linear layer for efficient operations with sparse matrices.

    This layer implements a sparse linear transformation, similar to nn.Linear,
    but operates on sparse matrices for memory efficiency.

    Args:
        in_features: Size of the input feature dimension.
        out_features: Size of the output feature dimension.
        connectivity: Sparse connectivity matrix in COO format.
        feature_dim: Dimension on which features reside (0 for rows, 1 for columns).
        bias: If set to False, no bias term is added.
        requires_grad: Whether the weight and bias parameters require gradient updates.
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
            x: Input tensor of shape (H, *) if feature_dim is 0, otherwise (*, H).

        Returns:
            Output tensor after sparse linear transformation.
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

    A sparse variant of the standard RNN that uses truly sparse linear
    transformations to compute the input-to-hidden and hidden-to-hidden
    transformations (and optionally the hidden-to-output transformations).

    These sparse transformations are computed using the `torch_sparse` package
    and allow for efficient memory usage for large networks.

    This allows for the network weights to directly be trained, a departure
    from GANs, which typically use fixed sparse weights.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        connectivity_hh: Union[torch.Tensor, PathLike, str],
        output_size: Optional[int] = None,
        connectivity_ih: Optional[Union[torch.Tensor, PathLike, str]] = None,
        connectivity_ho: Optional[Union[torch.Tensor, PathLike, str]] = None,
        bias_hh: bool = True,
        bias_ih: bool = False,
        bias_ho: bool = True,
        use_dense_ih: bool = False,
        use_dense_ho: bool = False,
        train_hh: bool = True,
        train_ih: bool = True,
        train_ho: bool = True,
        default_hidden_init_fn: str = "zeros",
        nonlinearity: str = "Sigmoid",
        batch_first: bool = True,
    ):
        """Initialize the SparseRNN layer.

        Args:
            input_size: Size of the input features.
            hidden_size: Size of the hidden state.
            connectivity_hh: Connectivity matrix for hidden-to-hidden connections.
            output_size: Size of the output features.
            connectivity_ih: Connectivity matrix for input-to-hidden connections.
            connectivity_ho: Connectivity matrix for hidden-to-output connections.
            bias_hh: Whether to use bias in the hidden-to-hidden connections.
            bias_ih: Whether to use bias in the input-to-hidden connections.
            bias_ho: Whether to use bias in the hidden-to-output connections.
            use_dense_ih: Whether to use a dense linear layer for input-to-hidden connections.
            use_dense_ho: Whether to use a dense linear layer for hidden-to-output connections.
            train_hh: Whether to train the hidden-to-hidden connections.
            train_ih: Whether to train the input-to-hidden connections.
            train_ho: Whether to train the hidden-to-output connections.
            default_hidden_init_fn: Initialization mode for the hidden state.
            nonlinearity: Nonlinearity function.
            batch_first: Whether the input is in (batch_size, seq_len, input_size) format.
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = (
            output_size if output_size is not None else hidden_size
        )
        self.default_hidden_init_fn = default_hidden_init_fn
        self.nonlinearity = get_activation(nonlinearity)
        self.batch_first = batch_first

        connectivity_hh, connectivity_ih, connectivity_ho = (
            self._init_connectivity(
                connectivity_hh, connectivity_ih, connectivity_ho
            )
        )

        self.hh = SparseLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            connectivity=connectivity_hh,
            feature_dim=0,
            bias=bias_hh,
            requires_grad=train_hh,
        )

        if connectivity_ih is not None:
            if use_dense_ih:
                raise ValueError(
                    "use_dense_ih must be False if connectivity_ih is provided"
                )
            self.ih = SparseLinear(
                in_features=input_size,
                out_features=hidden_size,
                connectivity=connectivity_ih,
                feature_dim=0,
                bias=bias_ih,
                requires_grad=train_ih,
            )
        elif use_dense_ih:
            warnings.warn(
                "connectivity_ih is not provided and use_dense_ih is True, "
                "using a dense linear layer for input-to-hidden connections. "
                "This may result in memory issues. If you are running out of "
                "memory, consider providing connectivity_ih or decreasing "
                "input_size."
            )
            if not train_ih:
                raise ValueError(
                    "train_ih must be True if connectivity_ih is not provided"
                )
            self.ih = nn.Linear(
                in_features=input_size,
                out_features=hidden_size,
                bias=bias_ih,
            )
        else:
            if input_size != hidden_size:
                raise ValueError(
                    "input_size must be equal to hidden_size if "
                    "connectivity_ih is not provided and use_dense_ih is False"
                )
            self.ih = nn.Identity()

        if connectivity_ho is not None:
            if use_dense_ho:
                raise ValueError(
                    "use_dense_ho must be False if connectivity_ho is provided"
                )
            if output_size is None:
                raise ValueError(
                    "output_size must be provided if and only if connectivity_ho is provided"
                )
            self.ho = SparseLinear(
                in_features=hidden_size,
                out_features=output_size,
                connectivity=connectivity_ho,
                feature_dim=0,
                bias=bias_ho,
                requires_grad=train_ho,
            )
        elif use_dense_ho:
            warnings.warn(
                "connectivity_ho is not provided and use_dense_ho is True, "
                "using a dense linear layer for hidden-to-output connections. "
                "This may result in memory issues. If you are running out of "
                "memory, consider providing connectivity_ho or decreasing "
                "hidden_size."
            )
            if output_size is None:
                raise ValueError(
                    "output_size must be provided if and only if use_dense_ho is True"
                )
            if not train_ho:
                raise ValueError(
                    "train_ho must be True if connectivity_ho is not provided"
                )
            self.ho = nn.Linear(
                in_features=hidden_size,
                out_features=output_size,
                bias=bias_ho,
            )
        else:
            if output_size is not None and output_size != hidden_size:
                raise ValueError(
                    "output_size should not be provided or should be equal to "
                    "hidden_size if connectivity_ho is not provided and "
                    "use_dense_ho is False"
                )
            self.ho = nn.Identity()

    def _init_connectivity(
        self,
        connectivity_hh: Union[torch.Tensor, PathLike, str],
        connectivity_ih: Optional[Union[torch.Tensor, PathLike, str]] = None,
        connectivity_ho: Optional[Union[torch.Tensor, PathLike, str]] = None,
    ) -> tuple[
        torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]
    ]:
        """Initialize connectivity matrices.

        Args:
            connectivity_hh: Connectivity matrix for hidden-to-hidden connections or path to load it from.
            connectivity_ih: Connectivity matrix for input-to-hidden connections or path to load it from.
            connectivity_ho: Connectivity matrix for hidden-to-output connections or path to load it from.

        Returns:
            Tuple containing the hidden-to-hidden connectivity tensor and input-to-hidden connectivity tensor (or None).

        Raises:
            ValueError: If connectivity matrices are not in COO format or have
                invalid dimensions.
        """

        connectivity_hh_tensor = load_sparse_tensor(connectivity_hh)

        if connectivity_ih is not None:
            connectivity_ih_tensor = load_sparse_tensor(connectivity_ih)
        else:
            connectivity_ih_tensor = None

        if connectivity_ho is not None:
            connectivity_ho_tensor = load_sparse_tensor(connectivity_ho)
        else:
            connectivity_ho_tensor = None

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

        if connectivity_ho_tensor is not None and (
            self.hidden_size != connectivity_ho_tensor.shape[1]
            or self.output_size != connectivity_ho_tensor.shape[0]
        ):
            raise ValueError(
                "connectivity_ho.shape[0] and output_size must be equal"
            )

        return (
            connectivity_hh_tensor,
            connectivity_ih_tensor,
            connectivity_ho_tensor,
        )

    def init_hidden(
        self,
        batch_size: int,
        init_fn: Optional[Union[str, TensorInitFnType]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """Initialize the hidden state.

        Args:
            batch_size: Batch size.
            init_fn: Initialization function.
            device: Device to allocate the hidden state on.

        Returns:
            The initialized hidden state of shape (batch_size, hidden_size).
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
            num_steps: Number of time steps.
            batch_size: Batch size.
            h0: Initial hidden states.
            hidden_init_fn: Initialization function.
            device: Device to allocate tensors on.

        Returns:
            The initialized hidden states for each time step.
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
            x: Input tensor.
            num_steps: Number of time steps.

        Returns:
            The formatted input tensor and corrected number of time steps.

        Raises:
            ValueError: For invalid input dimensions or step counts.
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

    def _format_result(
        self,
        outs: torch.Tensor,
        hs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Format the hidden states and outputs for output.

        Args:
            hs: Hidden states for each time step.
            outs: Outputs for each time step.

        Returns:
            Formatted outputs and hidden states.
        """

        if self.batch_first:
            return outs.permute(2, 0, 1), hs.permute(2, 0, 1)
        else:
            return outs.permute(0, 2, 1), hs.permute(0, 2, 1)

    def _clamp_connectivity(self) -> None:
        """Ensure the connectivity matrix is nonnegative."""

        self.hh.values.data.clamp_(min=0.0)

    def update_fn(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Update function for the SparseRNN.

        Args:
            x: Input tensor at current timestep.
            h: Hidden state from previous timestep.

        Returns:
            Updated hidden state.
        """
        return self.nonlinearity(self.ih(x) + self.hh(h))

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        h0: Optional[torch.Tensor] = None,
        hidden_init_fn: Optional[Union[str, TensorInitFnType]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the SparseRNN layer.

        Args:
            x: Input tensor.
            num_steps: Number of time steps.
            h0: Initial hidden state.
            hidden_init_fn: Initialization function.

        Returns:
            Hidden states and outputs.
        """

        # Ensure connectivity matrix is nonnegative
        self._clamp_connectivity()

        # Format input and initialize variables
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
        hs = torch.stack(hs)  # type: ignore

        assert hs.shape == (num_steps, self.hidden_size, batch_size)
        outs = self.ho(hs.transpose(0, 1).flatten(1))
        outs = outs.view(self.output_size, num_steps, batch_size).transpose(
            0, 1
        )

        return self._format_result(outs, hs)


class SparseODERNN(SparseRNN):
    """Sparse Ordinary Differential Equation Recurrent Neural Network.

    A continuous-time version of SparseRNN that uses an ODE solver to
    simulate the dynamics of the network and simultaneously compute the
    parameter gradients (see `torchode.AutoDiffAdjoint`).

    Args:
        compile_solver_kwargs: Keyword arguments for torch.compile.
    """

    def __init__(
        self,
        *args,
        compile_solver_kwargs: Optional[Mapping[str, Any]] = None,
        compile_update_fn_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Compile update_fn
        if compile_update_fn_kwargs is not None:
            self.update_fn = torch.compile(
                self.update_fn, **compile_update_fn_kwargs
            )

        # Define ODE solver
        term = to.ODETerm(self.update_fn, with_args=True)  # type: ignore
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(
            atol=1e-6, rtol=1e-3, term=term
        )
        self.solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # type: ignore

        # Compile solver
        if compile_solver_kwargs is not None:
            self.solver = torch.compile(self.solver, **compile_solver_kwargs)

    def _format_x(self, x: torch.Tensor):
        """Format the input tensor to match the expected shape.

        Args:
            x: Input tensor. If 2-dimensional, it is assumed to be of shape
                (batch_size, input_size). If 3-dimensional, it is assumed to be
                of shape (batch_size, sequence_length, input_size) if
                batch_first, else (sequence_length, batch_size, input_size).

        Returns:
            Formatted input tensor of shape (sequence_length, batch_size,
            input_size)

        Raises:
            ValueError: For invalid input dimensions.
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
            ts: Time points tensor.

        Returns:
            Formatted time points tensor.
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
            t: Current time point.
            x: Input tensor.
            start_time: Start time for simulation.
            end_time: End time for simulation.

        Returns:
            Index tensor for selecting the correct input.
        """
        idx = (t - start_time) / (end_time - start_time) * x.shape[0]
        idx[idx == x.shape[0]] = x.shape[0] - 1

        return idx.long()

    def update_fn(
        self, t: torch.Tensor, h: torch.Tensor, args: Mapping[str, Any]
    ) -> torch.Tensor:
        """ODE function for the SparseODERNN.

        Args:
            t: Current time point.
            h: Current hidden state.
            args: Additional arguments including input data.

        Returns:
            Rate of change of the hidden state.
        """
        h = h.t()
        x = args["x"]
        start_time = args["start_time"]
        end_time = args["end_time"]

        idx = self._index_from_time(t, x, start_time, end_time)

        h_new = self.nonlinearity(self.ih(x[idx]) + self.hh(h))

        dhdt = h_new - h

        return dhdt.t()

    def forward(
        self,
        x: torch.Tensor,
        num_evals: int = 2,
        start_time: float = 0.0,
        end_time: float = 1.0,
        h0: Optional[torch.Tensor] = None,
        hidden_init_fn: Optional[Union[str, TensorInitFnType]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the SparseODERNN layer.

        Solves the initial value problem for the ODE defined by update_fn.
        The gradients of the parameters are computed using the adjoint method
        (see `torchode.AutoDiffAdjoint`).

        Args:
            x: Input tensor.
            num_evals: Number of evaluations to return. The default of 2 means
                that the ODE will be evaluated at the start and end of the
                simulation and those values will be returned. Note that this
                does not mean the `update_fn` will be called `num_evals` times.
                It only affects the number of values returned, the step size
                controller determines the number of times the solver will call
                the `update_fn`.
            start_time: Start time for simulation.
            end_time: End time for simulation.
            h0: Initial hidden state.
            hidden_init_fn: Initialization function.

        Returns:
            Hidden states, outputs, and time points.
        """

        # Ensure connectivity matrix is nonnegative
        self._clamp_connectivity()

        # Format input and initialize variables
        x = self._format_x(x)
        batch_size = x.shape[-1]
        device = x.device

        # Define evaluation time points
        if num_evals < 2:
            raise ValueError("num_evals must be greater than 1")
        t_eval = (
            torch.linspace(start_time, end_time, num_evals, device=device)
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

        # Project to output space
        outs = self.ho(hs.transpose(0, 1).flatten(1))
        outs = outs.view(self.output_size, num_evals, batch_size).transpose(
            0, 1
        )

        # Format outputs
        ts = self._format_ts(sol.ts)
        outs, hs = self._format_result(outs, hs)

        return outs, hs, ts
