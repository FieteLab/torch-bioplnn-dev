import torch
import torch.nn as nn
import torch_sparse
import torchsparsegradutils as tsgu

from bioplnn.utils import expand_list, get_activation_class


class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity: torch.Tensor,
        sparse_format: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        feature_dim: int = -1,
        bias: bool = True,
        requires_grad: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.feature_dim = feature_dim
        self.mm_function = mm_function

        if connectivity.layout != torch.sparse_coo:
            raise ValueError("connectivity must be in COO format.")

        if in_features != connectivity.shape[1]:
            raise ValueError(
                f"Input size ({in_features}) must be equal to the number of columns in connectivity ({connectivity.shape[1]})."
            )
        if out_features != connectivity.shape[0]:
            raise ValueError(
                f"Output size ({out_features}) must be equal to the number of rows in connectivity ({connectivity.shape[0]})."
            )

        if mm_function == "torch_sparse":
            if sparse_format != "torch_sparse":
                raise ValueError(
                    "mm_function must be 'torch_sparse' when sparse_format is 'torch_sparse'."
                )
            indices, values = torch_sparse.coalesce(
                connectivity.indices().clone(),
                connectivity.values().clone(),
                self.out_features,
                self.in_features,
            )
            self.indices = nn.Parameter(indices, requires_grad=False)
            self.values = nn.Parameter(values.float(), requires_grad=requires_grad)
        elif mm_function not in ("native", "tsgu"):
            if sparse_format not in ("coo", "csr"):
                raise ValueError(
                    "mm_function must be 'native' or 'tsgu' when sparse_format is 'coo' or 'csr'."
                )
            weight = connectivity.clone().coalesce().float()
            if sparse_format == "csr":
                weight = weight.to_sparse_csr()
            self.weight = nn.Parameter(weight, requires_grad=requires_grad)
        else:
            raise ValueError(
                f"Invalid mm_function: {mm_function}. Choose from 'torch_sparse', 'native', 'tsgu'."
            )
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features, 1), requires_grad=requires_grad)
            if bias
            else None
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (H, *) if feature_first, otherwise (*, H).

        Returns:
            torch.Tensor: Output tensor.
        """
        shape = list(x.shape)
        permutation = torch.arange(x.dim())
        permutation[self.feature_dim] = 0
        permutation[0] = self.feature_dim
        if self.feature_dim != 0:
            x = x.permute(*permutation)
        x = x.flatten(start_dim=1)

        if self.mm_function == "torch_sparse":
            x = torch_sparse.spmm(
                self.indices,
                self.values,
                self.out_features,
                self.in_features,
                x,
            )
        elif self.mm_function == "native":
            x = torch.sparse.mm(self.weight, x)
        elif self.mm_function == "tsgu":
            x = tsgu.sparse_mm(self.weight, x)

        if self.bias is not None:
            x = x + self.bias

        if self.feature_dim != 0:
            x = x.permute(*permutation)
        shape[self.feature_dim] = self.out_features
        x = x.view(*shape)

        return x


class SparseSplineLinear(SparseLinear):
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


class SparseKANLayer(nn.Module):
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
                "({out_features}) must be equal to the number of rows in connectivity ({connectivity.shape[0]})"
            )
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.use_base_update = use_base_update

        offset = torch.arange(num_grids) * in_features
        offset = torch.stack((torch.zeros_like(offset), offset)).unsqueeze(1)
        indices_spline = (
            connectivity.indices().clone().unsqueeze(-1).expand(-1, -1, num_grids)
        )
        indices_spline = indices_spline + offset
        indices_spline = indices_spline.flatten(1)
        values_spline = connectivity.values().clone().repeat(num_grids)
        connectivity_spline = torch.sparse_coo_tensor(
            indices_spline,
            values_spline,
            (connectivity.shape[0], in_features * num_grids),
        ).coalesce()

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
        spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.flatten(1))
        if self.use_base_update:
            base = self.base_linear(self.base_nonlinearity(x))
            ret = ret + base
        return ret


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class SparseChebyKANLayer(nn.Module):
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

        offset = torch.arange(degree + 1) * in_features
        offset = torch.stack((torch.zeros_like(offset), offset)).unsqueeze(1)
        indices = (
            connectivity.indices().clone().unsqueeze(-1).expand(-1, -1, degree + 1)
        )
        indices = indices + offset
        indices = indices.flatten(1)
        values = connectivity.values().clone().repeat(degree + 1)
        connectivity = torch.sparse_coo_tensor(
            indices,
            values,
            (connectivity.shape[0], in_features * (degree + 1)),
        ).coalesce()

        self.linear = SparseLinear(
            in_features * (degree + 1),
            out_features,
            connectivity,
            sparse_format,
            mm_function,
            feature_dim=-1,
            bias=False,
        )

        self.arange = nn.Parameter(torch.arange(0, degree + 1), requires_grad=False)

    def forward(self, x, eps=1e-7):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        x = torch.clamp(x, -1 + eps, 1 - eps)
        # View and repeat input degree + 1 times
        x = x.unsqueeze(-1).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = self.linear(x.flatten(1))
        return y


class SparseRNN(nn.Module):
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
        use_layernorm: bool = True,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.nonlinearity = get_activation_class(nonlinearity)()

        self.hidden_size = expand_list(hidden_size, num_layers)
        connectivity_ih = expand_list(connectivity_ih, num_layers)
        connectivity_hh = expand_list(connectivity_hh, num_layers)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Module()
            layer.ih = SparseLinear(
                input_size if i == 0 else self.hidden_size[i - 1],
                self.hidden_size[i],
                connectivity_ih[i],
                sparse_format=sparse_format,
                mm_function=mm_function,
                feature_dim=0,
                bias=bias,
            )
            layer.hh = SparseLinear(
                self.hidden_size[i],
                self.hidden_size[i],
                connectivity_hh[i],
                sparse_format=sparse_format,
                mm_function=mm_function,
                feature_dim=0,
                bias=bias,
            )
            self.layers.append(layer)

        self.layernorms = nn.ModuleList(
            [
                nn.LayerNorm(self.hidden_size[i]) if use_layernorm else nn.Identity()
                for i in range(num_layers)
            ]
        )

    def forward(self, x, num_steps=None, return_activations=False):
        """
        Forward pass of the TopographicalCorticalCell.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        device = x.device

        if x.dim() == 2:
            if not self.batch_first:
                raise ValueError("batch_first must be False for 2D input.")
            if num_steps is None:
                raise ValueError("num_steps must be provided for 2D input.")
            x = x.t()
            x = [x] * num_steps
        elif x.dim() == 3:
            if self.batch_first:
                x = x.permute(
                    1, 2, 0
                )  # (batch_size, num_steps, input_size) -> (num_steps, input_size, batch_size)
            else:
                x = x.permute(
                    0, 2, 1
                )  # (num_steps, batch_size, input_size) -> (num_steps, input_size, batch_size)
            if num_steps is not None and x.shape[0] != num_steps:
                raise ValueError(
                    "num_steps must be None or equal to the length of the first dimension of x"
                )
            num_steps = x.shape[0]
        else:
            raise ValueError(
                f"Input tensor must be 2D or 3D, but got {x.dim()} dimensions."
            )

        batch_size = x[0].shape[-1]
        out = []

        if return_activations:
            activations = [[] for _ in range(len(self.layers))]

        h = [
            torch.zeros(self.hidden_size[i], batch_size).to(device)
            for i in range(len(self.layers))
        ]

        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                h[i] = self.nonlinearity(
                    layer.ih(x[t] if i == 0 else h[i - 1]) + layer.hh(h[i])
                )
                h[i] = self.layernorms[i](h[i].t()).t()
                if return_activations:
                    activations.append([h_i.clone() for h_i in h])
            out.append(h[-1])

        out = torch.stack(out)

        if return_activations:
            activations = [torch.stack([h_i for h_i in h_t]) for h_t in activations]

        h = [h_i.t() for h_i in h]
        if self.batch_first:
            out = out.permute(2, 0, 1)
            if return_activations:
                activations = [x.permute(2, 0, 1) for h_t in activations]
        else:
            out = out.permute(0, 2, 1)
            if return_activations:
                activations = [x.permute(0, 2, 1) for h_t in activations]

        if return_activations:
            return out, h, activations
        return out, h


class SparseRKANBase(nn.Module):
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
                nn.LayerNorm(self.hidden_size[i]) if use_layernorm else nn.Identity()
                for i in range(num_layers)
            ]
        )

    def forward(self, x, num_steps=None, return_activations=False):
        """
        Forward pass of the TopographicalCorticalCell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size) [same input across timesteps, no batch_first allowed],
                or (batch_size, num_steps, input_size) if batch_first,
                otherwise (num_steps, batch_size, input_size) [different input across timesteps].

        Returns:
            torch.Tensor: Output tensor.
        """
        device = x.device

        if x.dim() == 2:
            if not self.batch_first:  # x.shape == (batch_size, input_size)
                raise ValueError("batch_first must be True for 2D input.")
            if num_steps is None:
                raise ValueError("num_steps must be provided for 2D input.")
            batch_size = x.shape[0]
            x = [x] * num_steps
        elif x.dim() == 3:
            if self.batch_first:  # x.shape == (batch_size, num_steps, input_size)
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
        # x.shape == (num_steps, batch_size, input_size)
        # Note difference from SparseRNN
        out = []
        h = [
            torch.zeros(batch_size, self.hidden_size[i]).to(device)
            for i in range(len(self.layers))
        ]
        if return_activations:
            activations = [[] for _ in range(len(self.layers))]

        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                h[i] = layer.ih(x[t] if i == 0 else h[i - 1]) + layer.hh(h[i])
                h[i] = self.layernorms[i](h[i])
                if return_activations:
                    activations[i].append(h[i].clone())
            out.append(h[-1])

        out = torch.stack(out)
        if return_activations:
            activations = [torch.stack(activations_i) for activations_i in activations]

        if self.batch_first:
            out = out.transpose(
                0, 1
            )  # (num_steps, batch_size, hidden_size) -> (batch_size, num_steps, hidden_size)
            if return_activations:
                activations = [
                    activations_i.transpose(0, 1) for activations_i in activations
                ]  # (num_layers, num_steps, batch_size, hidden_size) -> (num_layers, batch_size, num_steps, hidden_size)

        if return_activations:
            return out, h, activations
        return out, h


class SparseRKAN(SparseRKANBase):
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
