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
        connectivity: torch.Tensor,
        sparse_layout: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        feature_dim: int = -1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.feature_dim = feature_dim
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
            x (torch.Tensor): Input tensor of shape (H, *) if feature_first, otherwise (*, H).

        Returns:
            torch.Tensor: Output tensor.
        """
        shape = x.shape
        permutation = list(range(x.dim()))
        permutation[self.feature_dim] = 0
        permutation[0] = self.feature_dim
        if self.feature_dim != 0:
            x = x.permute(*permutation)
        x = x.flatten(start_dim=1)

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

        if self.feature_dim != 0:
            x = x.permute(*permutation)
        shape[self.feature_dim] = self.out_features
        x = x.view(*shape)

        return x


class SparseRNN(nn.Module):
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
            layer.ih = SparseLinear(
                input_size if i == 0 else hidden_size[i - 1],
                hidden_size[i],
                connectivity_ih[i],
                connectivity_hh[i],
                sparse_layout=sparse_layout,
                mm_function=mm_function,
                feature_dim=0,
                bias=bias,
            )
            layer.hh = SparseLinear(
                hidden_size[i],
                hidden_size[i],
                connectivity_ih[i],
                connectivity_hh[i],
                sparse_layout=sparse_layout,
                mm_function=mm_function,
                feature_dim=0,
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

        batch_size = x[0].shape[-1]
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
