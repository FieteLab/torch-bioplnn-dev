from typing import Optional

import torch
import torch.nn as nn

from bioplnn.models.sparse import SparseLinear, SparseRNN
from bioplnn.models.topography import TopographicalRNNBase
from bioplnn.utils import expand_list, get_activation_class


class SparseSplineLinear(SparseLinear):
    """
    Sparse linear layer with spline basis functions.

    Extends the `SparseLinear` class to incorporate spline basis functions.

    Args:
        *args: Positional arguments passed to the base class.
        init_scale (float, optional): Standard deviation for weight initialization. Defaults to 0.1.
        **kwargs: Keyword arguments passed to the base class.
    """

    def __init__(
        self,
        *args,
        init_scale: float = 0.1,
        **kwargs,
    ) -> None:
        self.init_scale = init_scale
        super().__init__(
            *args,
            **kwargs,
        )

    def reset_parameters(self) -> None:
        """
        Resets the parameters of the layer, initializing the weights with a truncated normal distribution.
        """
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    """
    Radial basis function (RBF) layer.

    Computes the RBF activations for a given input.

    Args:
        grid_min (float, optional): Minimum value of the grid. Defaults to -2.0.
        grid_max (float, optional): Maximum value of the grid. Defaults to 2.0.
        num_grids (int, optional): Number of grid points. Defaults to 8.
        denominator (float, optional): Denominator for the RBF formula. Defaults to (grid_max - grid_min) / (num_grids - 1).
    """

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float = None,
    ) -> None:
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (
            num_grids - 1
        )

    def forward(self, x):
        """
        Computes the RBF activations for the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: RBF activations.
        """
        return torch.exp(
            -(((x[..., None] - self.grid) / self.denominator) ** 2)
        )


class SparseKANLayer(nn.Module):
    """
    Sparse Kernel Approximation Network (KAN) layer.

    Implements the KAN layer using sparse linear transformations and RBF activations.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        connectivity (torch.Tensor): Sparse connectivity matrix.
        grid_min (float, optional): Minimum value of the RBF grid. Defaults to -2.0.
        grid_max (float, optional): Maximum value of the RBF grid. Defaults to 2.0.
        num_grids (int, optional): Number of RBF grid points. Defaults to 8.
        use_base_update (bool, optional): Whether to use a base update. Defaults to True.
        base_nonlinearity (str, optional): Nonlinearity for the base update. Defaults to "silu".
        spline_weight_init_scale (float, optional): Standard deviation for spline weight initialization. Defaults to 0.1.
        sparse_format (str, optional): Sparse format for the connectivity matrix. Defaults to "torch_sparse".
        mm_function (str, optional): Matrix multiplication function. Defaults to "torch_sparse".
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity: torch.Tensor,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_nonlinearity="silu",
        spline_weight_init_scale: float = 0.1,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
    ) -> None:
        super().__init__()
        if (
            in_features != connectivity.shape[1]
            or out_features != connectivity.shape[0]
        ):
            raise ValueError(
                f"Input size ({in_features}) must be equal to the number of columns in connectivity ({connectivity.shape[1]}) and output size "
                f"({out_features}) must be equal to the number of rows in connectivity ({connectivity.shape[0]})"
            )
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.use_base_update = use_base_update

        # Create extended connectivity matrix for spline basis
        offset = torch.arange(num_grids) * in_features
        offset = torch.stack((torch.zeros_like(offset), offset)).unsqueeze(1)
        indices_spline = (
            connectivity.indices()
            .clone()
            .unsqueeze(-1)
            .expand(-1, -1, num_grids)
        )
        indices_spline = indices_spline + offset
        indices_spline = indices_spline.flatten(1)
        values_spline = connectivity.values().clone().repeat(num_grids)
        connectivity_spline = torch.sparse_coo_tensor(
            indices_spline,
            values_spline,
            (connectivity.shape[0], in_features * num_grids),
        ).coalesce()

        # Create sparse spline linear layer
        self.spline_linear = SparseSplineLinear(
            in_features * num_grids,
            out_features,
            connectivity_spline,
            sparse_format,
            mm_function,
            feature_dim=-1,
            bias=False,
            init_scale=spline_weight_init_scale,
        )

        # Create base linear layer if use_base_update is True
        if use_base_update:
            self.base_nonlinearity = get_activation_class(base_nonlinearity)()
            connectivity_base = torch.sparse_coo_tensor(
                connectivity.indices().clone(),
                connectivity.values().clone(),
                connectivity.shape,
            ).coalesce()
            self.base_linear = SparseLinear(
                in_features,
                out_features,
                connectivity_base,
                sparse_format,
                mm_function,
                feature_dim=-1,
                bias=False,
            )

    def forward(self, x):
        """
        Forward pass of the SparseKANLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.flatten(1))
        if self.use_base_update:
            base = self.base_linear(self.base_nonlinearity(x))
            ret = ret + base
        return ret


class SparseChebyKANLayer(nn.Module):
    """
    Sparse Chebyshev Kernel Approximation Network (KAN) layer.

    Implements the KAN layer using sparse linear transformations and Chebyshev polynomials.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        connectivity (torch.Tensor): Sparse connectivity matrix.
        degree (int): Degree of the Chebyshev polynomials.
        sparse_format (str, optional): Sparse format for the connectivity matrix. Defaults to "torch_sparse".
        mm_function (str, optional): Matrix multiplication function. Defaults to "torch_sparse".
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity: torch.Tensor,
        degree: int,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
    ) -> None:
        super().__init__()
        self.degree = degree
        self.sparse_format = sparse_format
        self.mm_function = mm_function

        # Pre-compute offsets for efficient indexing
        offset = torch.arange(degree + 1) * in_features
        offset = torch.stack((torch.zeros_like(offset), offset)).unsqueeze(1)

        # Create indices for the expanded connectivity matrix
        indices = (
            connectivity.indices()
            .clone()
            .unsqueeze(-1)
            .expand(-1, -1, degree + 1)
        )
        indices = indices + offset
        indices = indices.flatten(1)

        # Create values for the expanded connectivity matrix
        values = connectivity.values().clone().repeat(degree + 1)

        # Construct the expanded sparse connectivity matrix
        connectivity_expanded = torch.sparse_coo_tensor(
            indices,
            values,
            (connectivity.shape[0], in_features * (degree + 1)),
        ).coalesce()

        # Create a sparse linear layer for the expanded connectivity
        self.linear = SparseLinear(
            in_features * (degree + 1),
            out_features,
            connectivity_expanded,
            sparse_format,
            mm_function,
            feature_dim=-1,
            bias=False,
        )

        # Create a parameter for the Chebyshev polynomial degree (not trainable)
        self.arange = nn.Parameter(
            torch.arange(0, degree + 1), requires_grad=False
        )

    def forward(self, x, eps=1e-7):
        """
        Forward pass of the SparseChebyKANLayer.

        Args:
            x (torch.Tensor): Input tensor.
            eps (float, optional): Epsilon value for numerical stability during normalization. Defaults to 1e-7.

        Returns:
            torch.Tensor: Output tensor after applying Chebyshev polynomials and sparse linear transformation.
        """

        # Normalize input to the range [-1, 1] using tanh with a small epsilon for stability
        x = torch.tanh(x)
        x = torch.clamp(x, -1 + eps, 1 - eps)

        # Expand the input with degree + 1 repetitions for Chebyshev polynomials
        x = x.unsqueeze(-1).expand(-1, -1, self.degree + 1)

        # Apply arccosine to the normalized input
        x = x.acos()

        # Multiply by the pre-computed Chebyshev polynomial degree range
        x *= self.arange

        # Apply cosine to obtain the Chebyshev basis functions
        x = x.cos()

        # Compute the Chebyshev interpolation using the sparse linear layer
        y = self.linear(x.flatten(1))

        return y


class SparseRKANBase(nn.Module):
    """
    Base class for Sparse Recurrent Kernel Approximation Network (RKAN) layers.

    Provides the common functionality for SparseRKAN and SparseRChebyKAN.

    Args:
        hidden_size (int | list[int]): Size of the hidden state.
        connectivity_ih (torch.Tensor | list[torch.Tensor]): Connectivity matrix for input-to-hidden connections.
        connectivity_hh (torch.Tensor | list[torch.Tensor]): Connectivity matrix for hidden-to-hidden connections.
        num_layers (int): Number of layers.
        batch_first (bool): Whether the input is in (batch_size, seq_len, input_size) format.
        use_layernorm (bool, optional): Whether to use layer normalization. Defaults to True.
    """

    def __init__(
        self,
        hidden_size: int | list[int],
        connectivity_ih: torch.Tensor | list[torch.Tensor],
        connectivity_hh: torch.Tensor | list[torch.Tensor],
        num_layers: int,
        batch_first: bool,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first

        self.hidden_size = expand_list(hidden_size, num_layers)
        self.connectivity_ih = expand_list(connectivity_ih, num_layers)
        self.connectivity_hh = expand_list(connectivity_hh, num_layers)

        self.layernorms = nn.ModuleList(
            [
                nn.LayerNorm(self.hidden_size[i])
                if use_layernorm
                else nn.Identity()
                for i in range(num_layers)
            ]
        )

    def forward(self, x, num_steps=None):
        """
        Forward pass of the SparseRKANBase layer.

        Args:
            x (torch.Tensor): Input tensor.
            num_steps (int, optional): Number of time steps. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        device = x.device

        # Check input dimensions and prepare for processing
        if x.dim() == 2:
            if not self.batch_first:  # x.shape == (batch_size, input_size)
                raise ValueError("batch_first must be True for 2D input.")
            if num_steps is None:
                raise ValueError("num_steps must be provided for 2D input.")
            batch_size = x.shape[0]
            x = [x] * num_steps
        elif x.dim() == 3:
            if (
                self.batch_first
            ):  # x.shape == (batch_size, num_steps, input_size)
                x = x.transpose(0, 1)
            if num_steps is not None and x.shape[0] != num_steps:
                raise ValueError(
                    "num_steps must be None or equal to the length of the first dimension of x"
                )
            batch_size = x.shape[1]
            num_steps = x.shape[0]
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D, but got {x.dim()} dimensions."
            )

        # Initialize hidden state
        h = [
            torch.zeros(batch_size, self.hidden_size[0]).to(device)
            for _ in range(len(self.layers))
        ]
        out = []

        # Process input sequence
        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                h[i] = layer.ih(x[t] if i == 0 else h[i - 1]) + layer.hh(h[i])
                h[i] = self.layernorms[i](h[i])
            out.append(h[-1])

        # Stack outputs and adjust dimensions if necessary
        out = torch.stack(out)

        if self.batch_first:
            out = out.transpose(0, 1)

        return out, h


class TopographicalRNN(TopographicalRNNBase):
    """
    Topographical Recurrent Neural Network (TRNN) using SparseRNN.

    Extends the TopographicalRNNBase class.

    Args:
        sheet_size (tuple[int, int], optional): Size of the sheet-like topology. Defaults to (150, 300).
        synapse_std (float, optional): Standard deviation for random synapse initialization. Defaults to 10.
        synapses_per_neuron (int, optional): Number of synapses per neuron. Defaults to 32.
        self_recurrence (bool, optional): Whether to include self-recurrent connections. Defaults to True.
        connectivity_hh (Optional[str | torch.Tensor], optional): Path to a file containing the hidden-to-hidden connectivity matrix or the matrix itself. Defaults to None.
        connectivity_ih (Optional[str | torch.Tensor], optional): Path to a file containing the input-to-hidden connectivity matrix or the matrix itself. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to 10.
        batch_first (bool, optional): Whether the input is in (batch_size, seq_len, input_size) format. Defaults to True.
        input_indices (Optional[str | torch.Tensor], optional): Path to a file containing the input indices or the tensor itself (specifying which neurons receive input). Defaults to None.
        output_indices (Optional[str | torch.Tensor], optional): Path to a file containing the output indices or the tensor itself (specifying which neurons contribute to the output). Defaults to None.
        out_nonlinearity (str, optional): Nonlinearity applied to the output layer. Defaults to "relu".
        rnn_nonlinearity (str, optional): Nonlinearity used in the SparseRNN. Defaults to "relu".
        use_layernorm (bool, optional): Whether to use layer normalization in the SparseRNN. Defaults to False.
        bias (bool, optional): Whether to use bias in the SparseRNN. Defaults to True.
    """

    def __init__(
        self,
        num_classes: int,
        sheet_size: tuple[int, int] = (150, 300),
        synapse_std: float = 10,
        synapses_per_neuron: int = 32,
        self_recurrence: bool = True,
        connectivity_hh: Optional[str | torch.Tensor] = None,
        connectivity_ih: Optional[str | torch.Tensor] = None,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        batch_first: bool = True,
        input_indices: Optional[str | torch.Tensor] = None,
        output_indices: Optional[str | torch.Tensor] = None,
        out_nonlinearity: str = "relu",
        rnn_nonlinearity: str = "relu",
        use_layernorm: bool = False,
        bias: bool = True,
    ):
        super().__init__(
            sheet_size=sheet_size,
            synapse_std=synapse_std,
            synapses_per_neuron=synapses_per_neuron,
            self_recurrence=self_recurrence,
            connectivity_hh=connectivity_hh,
            connectivity_ih=connectivity_ih,
            num_classes=num_classes,
            batch_first=batch_first,
            input_indices=input_indices,
            output_indices=output_indices,
            out_nonlinearity=out_nonlinearity,
        )

        self.rnn = SparseRNN(
            input_size=self.num_neurons,
            hidden_size=self.num_neurons,
            connectivity_ih=self.connectivity_ih,
            connectivity_hh=self.connectivity_hh,
            num_layers=1,
            sparse_format=sparse_format,
            mm_function=mm_function,
            batch_first=batch_first,
            use_layernorm=use_layernorm,
            nonlinearity=rnn_nonlinearity,
            bias=bias,
        )


class SparseRKAN(SparseRKANBase):
    """
    Sparse Recurrent Kernel Approximation Network (RKAN) layer using spline basis functions.

    Extends the SparseRKANBase class.

    Args:
        input_size (int | list[int]): Size of the input.
        hidden_size (int | list[int]): Size of the hidden state.
        connectivity_ih (torch.Tensor | list[torch.Tensor]): Connectivity matrix for input-to-hidden connections.
        connectivity_hh (torch.Tensor | list[torch.Tensor]): Connectivity matrix for hidden-to-hidden connections.
        num_layers (int, optional): Number of layers. Defaults to 1.
        sparse_format (str, optional): Sparse format. Defaults to "torch_sparse".
        mm_function (str, optional): Matrix multiplication function. Defaults to "torch_sparse".
        batch_first (bool, optional): Whether the input is in (batch_size, seq_len, input_size) format. Defaults to True.
        grid_min (float, optional): Minimum value of the RBF grid. Defaults to -2.0.
        grid_max (float, optional): Maximum value of the RBF grid. Defaults to 2.0.
        num_grids (int, optional): Number of RBF grid points. Defaults to 8.
        use_base_update (bool, optional): Whether to use a base update. Defaults to True.
        base_nonlinearity (str, optional): Nonlinearity for the base update. Defaults to "silu".
        spline_weight_init_scale (float, optional): Standard deviation for spline weight initialization. Defaults to 0.1.
        use_layernorm (bool, optional): Whether to use layer normalization. Defaults to True.
    """

    def __init__(
        self,
        input_size: int | list[int],
        hidden_size: int | list[int],
        connectivity_ih: torch.Tensor | list[torch.Tensor],
        connectivity_hh: torch.Tensor | list[torch.Tensor],
        num_layers: int = 1,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        batch_first: bool = True,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_nonlinearity: str = "silu",
        spline_weight_init_scale: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__(
            hidden_size=hidden_size,
            connectivity_ih=connectivity_ih,
            connectivity_hh=connectivity_hh,
            num_layers=num_layers,
            batch_first=batch_first,
            use_layernorm=use_layernorm,
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Module()
            layer.ih = SparseKANLayer(
                in_features=input_size if i == 0 else self.hidden_size[i - 1],
                out_features=self.hidden_size[i],
                connectivity=self.connectivity_ih[i],
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_nonlinearity=base_nonlinearity,
                spline_weight_init_scale=spline_weight_init_scale,
                sparse_format=sparse_format,
                mm_function=mm_function,
            )
            layer.hh = SparseKANLayer(
                in_features=self.hidden_size[i],
                out_features=self.hidden_size[i],
                connectivity=self.connectivity_hh[i],
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_nonlinearity=base_nonlinearity,
                spline_weight_init_scale=spline_weight_init_scale,
                sparse_format=sparse_format,
                mm_function=mm_function,
            )
            self.layers.append(layer)


class SparseRChebyKAN(SparseRKANBase):
    """
    Sparse Recurrent Kernel Approximation Network (RKAN) layer using Chebyshev polynomials.

    Extends the SparseRKANBase class.

    Args:
        input_size (int | list[int]): Size of the input.
        hidden_size (int | list[int]): Size of the hidden state.
        connectivity_ih (torch.Tensor | list[torch.Tensor]): Connectivity matrix for input-to-hidden connections.
        connectivity_hh (torch.Tensor | list[torch.Tensor]): Connectivity matrix for hidden-to-hidden connections.
        num_layers (int, optional): Number of layers. Defaults to 1.
        batch_first (bool, optional): Whether the input is in (batch_size, seq_len, input_size) format. Defaults to True.
        sparse_format (str, optional): Sparse format. Defaults to "torch_sparse".
        mm_function (str, optional): Matrix multiplication function. Defaults to "torch_sparse".
        degree (int, optional): Degree of the Chebyshev polynomials. Defaults to 5.
        use_layernorm (bool, optional): Whether to use layer normalization. Defaults to False.
    """

    def __init__(
        self,
        input_size: int | list[int],
        hidden_size: int | list[int],
        connectivity_ih: torch.Tensor | list[torch.Tensor],
        connectivity_hh: torch.Tensor | list[torch.Tensor],
        num_layers: int = 1,
        batch_first: bool = True,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        degree: int = 5,
        use_layernorm: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            connectivity_ih=connectivity_ih,
            connectivity_hh=connectivity_hh,
            num_layers=num_layers,
            batch_first=batch_first,
            use_layernorm=use_layernorm,
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Module()
            layer.ih = SparseChebyKANLayer(
                in_features=input_size if i == 0 else self.hidden_size[i - 1],
                out_features=self.hidden_size[i],
                connectivity=self.connectivity_ih[i],
                degree=degree,
                sparse_format=sparse_format,
                mm_function=mm_function,
            )
            layer.hh = SparseChebyKANLayer(
                in_features=self.hidden_size[i],
                out_features=self.hidden_size[i],
                connectivity=self.connectivity_hh[i],
                degree=degree,
                sparse_format=sparse_format,
                mm_function=mm_function,
            )
            self.layers.append(layer)


class TopographicalRKAN(TopographicalRNNBase):
    def __init__(
        self,
        sheet_size: tuple[int, int] = (150, 300),
        synapse_std: float = 10,
        synapses_per_neuron: int = 32,
        self_recurrence: bool = True,
        connectivity_hh: Optional[str | torch.Tensor] = None,
        connectivity_ih: Optional[str | torch.Tensor] = None,
        input_indices: Optional[str | torch.Tensor] = None,
        output_indices: Optional[str | torch.Tensor] = None,
        num_classes: int = 10,
        batch_first: bool = True,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
        use_layernorm: bool = True,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        base_nonlinearity: str = "silu",
        out_nonlinearity: str = "relu",
    ):
        super().__init__(
            sheet_size=sheet_size,
            synapse_std=synapse_std,
            synapses_per_neuron=synapses_per_neuron,
            self_recurrence=self_recurrence,
            connectivity_hh=connectivity_hh,
            connectivity_ih=connectivity_ih,
            num_classes=num_classes,
            batch_first=batch_first,
            input_indices=input_indices,
            output_indices=output_indices,
            out_nonlinearity=out_nonlinearity,
        )

        self.rnn = SparseRKAN(
            input_size=self.num_neurons,
            hidden_size=self.num_neurons,
            connectivity_ih=self.connectivity_ih,
            connectivity_hh=self.connectivity_hh,
            num_layers=1,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_nonlinearity=base_nonlinearity,
            spline_weight_init_scale=spline_weight_init_scale,
            use_layernorm=use_layernorm,
            sparse_format=sparse_format,
            mm_function=mm_function,
            batch_first=batch_first,
        )


class TopographicalRChebyKAN(TopographicalRNNBase):
    def __init__(
        self,
        sheet_size: tuple[int, int] = (150, 300),
        synapse_std: float = 10,
        synapses_per_neuron: int = 32,
        self_recurrence: bool = True,
        connectivity_hh: Optional[str | torch.Tensor] = None,
        connectivity_ih: Optional[str | torch.Tensor] = None,
        input_indices: Optional[str | torch.Tensor] = None,
        output_indices: Optional[str | torch.Tensor] = None,
        num_classes: int = 10,
        batch_first: bool = True,
        degree: int = 5,
        use_layernorm: bool = False,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        out_nonlinearity: str = "relu",
    ):
        super().__init__(
            sheet_size=sheet_size,
            synapse_std=synapse_std,
            synapses_per_neuron=synapses_per_neuron,
            self_recurrence=self_recurrence,
            connectivity_hh=connectivity_hh,
            connectivity_ih=connectivity_ih,
            num_classes=num_classes,
            batch_first=batch_first,
            input_indices=input_indices,
            output_indices=output_indices,
            out_nonlinearity=out_nonlinearity,
        )

        self.rnn = SparseRChebyKAN(
            input_size=self.num_neurons,
            hidden_size=self.num_neurons,
            connectivity_ih=self.connectivity_ih,
            connectivity_hh=self.connectivity_hh,
            num_layers=1,
            degree=degree,
            use_layernorm=use_layernorm,
            sparse_format=sparse_format,
            mm_function=mm_function,
            batch_first=batch_first,
        )
