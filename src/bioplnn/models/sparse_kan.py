import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import torchsparsegradutils as tsgu

from bioplnn.utils import extend_for_multilayer, get_activation_class


class SplineLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: list[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FastKANLayer(
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class SparseLinearKAN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity: torch.Tensor,
        sparse_layout: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        features_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.features_first = features_first
        self.mm_function = mm_function

        if connectivity.layout != "coo":
            raise ValueError("connectivity must be in COO format.")

        if mm_function == "torch_sparse":
            if sparse_layout != "torch_sparse":
                raise ValueError(
                    "mm_function must be 'torch_sparse' when sparse_layout is 'torch_sparse'."
                )
            indices, values = torch_sparse.coalesce(
                connectivity.indices().clone(),
                connectivity.values().clone(),
                self.out_features,
                self.in_features,
            )
            self.indices = nn.Parameter(indices, requires_grad=False)
            self.values = nn.Parameter(values.float())
        elif mm_function not in ("native", "tsgu"):
            if sparse_layout not in ("coo", "csr"):
                raise ValueError(
                    "mm_function must be 'native' or 'tsgu' when sparse_layout is 'coo' or 'csr'."
                )
            weight = connectivity.clone().coalesce().float()
            if sparse_layout == "csr":
                weight = weight.to_sparse_csr()
            self.weight = nn.Parameter(weight)
        else:
            raise ValueError(
                f"Invalid mm_function: {mm_function}. Choose from 'torch_sparse', 'native', 'tsgu'."
            )
        self.bias = nn.Parameter(torch.zeros(self.out_features, 1)) if bias else None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (H, *) if features_first, otherwise (*, H).

        Returns:
            torch.Tensor: Output tensor.
        """
        shape = x.shape
        if self.features_first:
            x = x.flatten(start_dim=1)
        else:
            x = x.flatten(start_dim=0, end_dim=-2)
            x = x.t()

        if self.mm_function == "torch_sparse":
            x = torch_sparse.spmm(
                self.indices,
                self.values,
                self.in_features,
                self.out_features,
                x,
            )
        elif self.mm_function == "native":
            x = torch.sparse.mm(self.weight, x)
        elif self.mm_function == "tsgu":
            x = tsgu.sparse_mm(self.weight, x)

        if self.bias is not None:
            x = x + self.bias

        if self.features_first:
            x = x.view(self.out_features, *shape[1:])
        else:
            x = x.t()
            x = x.view(*shape[:-1], self.out_features)

        return x


class SparseRKAN(nn.Module):
    def __init__(
        self,
        input_size: int | list[int],
        hidden_size: int | list[int],
        connectivity_ih: torch.Tensor | list[torch.Tensor],
        connectivity_hh: torch.Tensor | list[torch.Tensor],
        num_layers: int = 1,
        sparse_layout: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        batch_first: bool = True,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.nonlinearity = get_activation_class(nonlinearity)()

        self.input_size = extend_for_multilayer(input_size, num_layers)
        self.hidden_size = extend_for_multilayer(hidden_size, num_layers)
        connectivity_ih = extend_for_multilayer(connectivity_ih, num_layers)
        connectivity_hh = extend_for_multilayer(connectivity_hh, num_layers)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Module()
            layer.ih = SparseLinearKAN(
                input_size if i == 0 else hidden_size[i - 1],
                hidden_size[i],
                connectivity_ih[i],
                connectivity_hh[i],
                sparse_layout=sparse_layout,
                mm_function=mm_function,
                features_first=True,
                bias=bias,
            )
            layer.hh = SparseLinearKAN(
                hidden_size[i],
                hidden_size[i],
                connectivity_ih[i],
                connectivity_hh[i],
                sparse_layout=sparse_layout,
                mm_function=mm_function,
                features_first=True,
                bias=bias,
            )
            self.layers.append(layer)

    def forward(self, x, num_steps=None, return_activations=False):
        """
        Forward pass of the TopographicalCorticalCell.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        if x.dim() == 2:
            if self.batch_first:  # x.shape == (batch_size, input_size)
                x = x.t()
            if num_steps is None:
                raise ValueError("num_steps must be provided for 2D input.")
            x = [x] * num_steps
            # x.shape = (num_steps, input_size, batch_size)
        elif x.dim() == 3:
            if self.batch_first:  # x.shape == (batch_size, num_steps, input_size)
                x = x.permute(1, 2, 0)
            else:  # x.shape == (num_steps, batch_size, input_size)
                x = x.permute(0, 2, 1)
            # x.shape = (num_steps, input_size, batch_size)
            if num_steps is not None and x.shape[0] != num_steps:
                raise ValueError(
                    "num_steps must be None or equal to the length of the first dimension of x"
                )
            num_steps = x.shape[0]
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D, but got {x.dim()} dimensions."
            )

        batch_size = x.shape[-1]
        out = []

        if return_activations:
            activations = []

        h = torch.zeros(self.num_layers, self.hidden_size, batch_size)
        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                h[i] = self.nonlinearity(
                    layer.ih(x[t] if i == 0 else h[i - 1]) + layer.hh(h[i])
                )
            out.append(h[-1])
            if return_activations:
                activations.append(h.clone())

        out = torch.stack(out)
        if return_activations:
            activations = torch.stack(activations)

        if self.batch_first:
            out = out.permute(2, 0, 1)
            h = h.permute(2, 0, 1)
            if return_activations:
                activations = activations.permute(3, 0, 1, 2)
        else:
            out = out.permute(0, 2, 1)
            h = h.permute(0, 2, 1)
            if return_activations:
                activations = activations.permute(0, 3, 1, 2)

        if return_activations:
            return out, h, activations
        return out, h
