import torch
import torch.nn as nn
import torch_sparse
import torchsparsegradutils as tsgu

from bioplnn.utils import extend_for_multilayer, get_activation_class


class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        connectivity_indices: torch.Tensor,
        connectivity_values: torch.Tensor,
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

        if mm_function == "torch_sparse":
            if sparse_layout != "torch_sparse":
                raise ValueError(
                    "mm_function must be 'torch_sparse' when sparse_layout is 'torch_sparse'."
                )
            indices, values = torch_sparse.coalesce(
                connectivity_indices.clone(),
                connectivity_values.clone(),
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
            weight = (
                torch.sparse_coo_tensor(
                    connectivity_indices.clone(),
                    connectivity_values.clone(),
                    (self.out_features, self.in_features),
                )
                .coalesce()
                .float()
            )
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


class SparseRNN(nn.Module):
    def __init__(
        self,
        input_size: int | list[int],
        hidden_size: int | list[int],
        connectivity_values_ih: torch.Tensor | list[torch.Tensor],
        connectivity_indices_ih: torch.Tensor | list[torch.Tensor],
        connectivity_values_hh: torch.Tensor | list[torch.Tensor],
        connectivity_indices_hh: torch.Tensor | list[torch.Tensor],
        num_layers: int = 1,
        sparse_layout: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        batch_first: bool = True,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ):
        super(SparseRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.nonlinearity = get_activation_class(nonlinearity)()

        self.input_size = extend_for_multilayer(input_size, num_layers)
        self.hidden_size = extend_for_multilayer(hidden_size, num_layers)
        connectivity_values_ih = extend_for_multilayer(
            connectivity_values_ih, num_layers
        )
        connectivity_values_ih = extend_for_multilayer(
            connectivity_indices_ih, num_layers
        )
        connectivity_values_hh = extend_for_multilayer(
            connectivity_values_hh, num_layers
        )
        connectivity_values_hh = extend_for_multilayer(
            connectivity_indices_hh, num_layers
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Module()
            layer.ih = SparseLinear(
                input_size if i == 0 else hidden_size[i - 1],
                hidden_size[i],
                connectivity_values_ih[i],
                connectivity_indices_ih[i],
                sparse_layout=sparse_layout,
                mm_function=mm_function,
                features_first=True,
                bias=bias,
            )
            layer.hh = SparseLinear(
                hidden_size[i],
                hidden_size[i],
                connectivity_values_hh[i],
                connectivity_indices_hh[i],
                sparse_layout=sparse_layout,
                mm_function=mm_function,
                features_first=True,
                bias=bias,
            )
            self.layers.append(layer)

    def forward(self, x, num_steps=None):
        """
        Forward pass of the TopographicalCorticalCell.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x: Dense (strided) tensor of shape (batch_size, num_neurons) if
        # batch_first, otherwise (num_neurons, batch_size)
        # assert self.weight.is_coalesced()

        if x.dim() == 2:
            if self.batch_first:
                x = x.t()
            if num_steps is None:
                raise ValueError("num_steps must be provided for 2D input.")
            x = [x] * num_steps
        elif x.dim() == 3:
            if self.batch_first:
                x = x.permute(1, 2, 0)
            else:
                x = x.permute(0, 2, 1)
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

        h = torch.zeros(self.num_layers, self.hidden_size, batch_size)
        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                h[i] = self.nonlinearity(
                    layer.ih(x[t] if i == 0 else h[i - 1]) + layer.hh(h[i])
                )
            out.append(h[-1])

        out = torch.stack(out)

        if self.batch_first:
            out = out.permute(2, 0, 1)
        else:
            out = out.permute(0, 2, 1)

        return out
