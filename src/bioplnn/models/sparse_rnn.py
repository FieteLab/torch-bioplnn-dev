import math
from typing import Any, Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_sparse
import torchsparsegradutils as tsgu
from matplotlib import animation

from bioplnn.utils import get_activation_class, idx_2D_to_1D

class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        values: torch.Tensor,
        indices: torch.Tensor,
        mm_function: str = "torch_sparse",
        bias: bool = True,
        activation: str = "tanh",
        batch_first: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = get_activation_class(activation)
        self.batch_first = batch_first
        self.mm_function = mm_function

        if mm_function == "torch_sparse":
            indices, weight = torch_sparse.coalesce(
                indices, values, self.out_features, self.in_features
            )
            self.indices = nn.Parameter(indices, requires_grad=False)
        elif mm_function in ("native", "tsgu"):
            weight = torch.sparse_coo_tensor(
                indices,
                values,
                (self.out_features, self.in_features),
                check_invariants=True,
            ).coalesce().to_sparse_csr()
        else:
            raise ValueError(
                f"Invalid mm_function: {mm_function}. Choose from 'torch_sparse', 'native', 'tsgu'."
            )
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.out_features, 1)) if bias else None
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x: Dense (strided) tensor of shape (batch_size, num_neurons)
        if self.batch_first:
            x = x.t()

        # Perform sparse matrix multiplication with or without bias
        if self.mm_function == "torch_sparse":
            x = (
                torch_sparse.spmm(
                    self.indices,
                    self.weight,
                    self.in_features,  # type: ignore
                    self.out_features,
                    x,
                )
                + self.bias
            )
        elif self.mm_function == "native":
            x = torch.sparse.mm(self.weight, x) + self.bias
        elif self.mm_function == "tsgu":
            x = tsgu.sparse_mm(self.weight, x) + self.bias
            

        # Transpose output back to batch first
        if self.batch_first:
            x = x.t()

        return x


class SparseRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        mm_function: str,
        values: torch.Tensor,
        indices: torch.Tensor,
        num_layers: int,
        bias: bool = True,
        activation: str = "tanh",
        batch_first: bool = False,
    ):
        super(SparseRNN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.bias = bias
        self.activation = get_activation_class(activation)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SparseLinear(
                in_features if i == 0 else out_features,
                out_features,
                values,
                indices,
                mm_function=mm_function,
                bias=bias,
                activation=activation,
                batch_first=False,
            ))
            
    def forward(self, x):
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
        
        if self.dim() == 3:
            if self.batch_first:
                x = x.permute(1, 2, 0)
            else:
                x = x.permute(0, 2, 1)
        elif self.dim() == 2:
            x = x.unsqueeze(-1)
        else:
            raise ValueError("Input tensor must be 2D or 3D.")
        
        L, H, B = x.shape

        # Perform sparse matrix multiplication with or without bias
        h = torch.zeros(self.num_layers, self.in_features, B)
        for t in range(L):
            for i, layer in enumerate(self.layers):
                x = layer(x[t] if t == i == 0 else x, h[i])
                

        # Transpose output back to batch first
        if self.batch_first:
            x = x.t()

        return x