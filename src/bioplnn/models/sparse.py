from typing import Optional

import torch
import torch.nn as nn
import torch_sparse

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

    def forward(self, x):
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
            x = x.permute(*permutation)

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
        input_size int: Size of the input.
        hidden_size int: Size of the hidden state.
        connectivity_ih torch.Tensor: Connectivity matrix for input-to-hidden connections.
        connectivity_hh torch.Tensor: Connectivity matrix for hidden-to-hidden connections.
        batch_first (bool, optional): Whether the input is in (batch_size, seq_len, input_size) format. Defaults to True.
        use_layernorm (bool, optional): Whether to use layer normalization. Defaults to True.
        nonlinearity (str, optional): Nonlinearity function. Defaults to "tanh".
        bias (bool, optional): Whether to use bias. Defaults to True.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        connectivity_ih: torch.Tensor,
        connectivity_hh: torch.Tensor,
        hidden_init_mode: str = "zeros",
        use_layernorm: bool = False,
        nonlinearity: str = "tanh",
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_init_mode = hidden_init_mode
        self.nonlinearity = get_activation(nonlinearity)
        self.batch_first = batch_first

        # Create layers and layer normalization modules
        self.ih = SparseLinear(
            input_size,
            hidden_size,
            connectivity_ih,
            feature_dim=0,
            bias=False,
        )
        self.hh = SparseLinear(
            hidden_size,
            hidden_size,
            connectivity_hh,
            feature_dim=0,
            bias=bias,
        )

        self.layernorm = (
            nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()
        )

    def _init_hidden(
        self,
        batch_size: int,
        init_mode: str = "zeros",
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

        return getattr(torch, init_mode)(
            batch_size, self.hidden_size, device=device
        )

    def init_state(
        self,
        h_0: Optional[torch.Tensor],
        num_steps: int,
        batch_size: int,
        device=None,
    ) -> list[torch.Tensor]:
        """
        Initializes the internal state of the network.

        Args:
            h_0 (Optional[torch.Tensor]): Initial hidden states for each layer.
            num_steps (int): Number of time steps.
            batch_size (int): Batch size.
            device (torch.device, optional): Device to allocate tensors.

        Returns:
            list[Optional[torch.Tensor]]: The initialized hidden states for each time step.
        """
        hs: list[torch.Tensor] = [None] * num_steps  # type: ignore

        if h_0 is None:
            hs[-1] = self._init_hidden(
                batch_size, init_mode=self.hidden_init_mode, device=device
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

    def _format_result(self, hs: list[torch.Tensor]) -> torch.Tensor:
        """
        Formats the hidden states.

        Args:
            hs (list[torch.Tensor]): Hidden states for each layer and time step.

        Returns:
            torch.Tensor: The formatted hidden states.
        """
        h = torch.stack(hs)
        if self.batch_first:
            h = h.permute(2, 0, 1)
        else:
            h = h.permute(0, 2, 1)

        return h

    def _format_x_ode(self, x: torch.Tensor):
        """
        Formats the input tensor to match the expected shape.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The formatted input tensor.
        """
        if x.dim() == 2:
            x = x.t()
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

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        h_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        hs = self.init_state(
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


class SparseRNNBaseODE(SparseRNN):
    def _forward(self, t, h, x):
        h = h.transpose(0, 1)

        x = self._format_x_ode(x)

        h = self.nonlinearity(self.ih(x[t]) + self.hh(h))
        h = self.layernorm(h)

        return h
