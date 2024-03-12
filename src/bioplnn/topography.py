import math
from typing import Any, Optional
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch_sparse
import torchsparsegradutils as tsgu

from bioplnn.utils import idx_2D_to_1D


class TopographicalCorticalCell(nn.Module):
    def __init__(
        self,
        sheet_size: tuple[int, int] = (100, 100),
        connectivity_std: float = 10,
        synapses_per_neuron: int = 32,
        bias: bool = True,
        mm_function: str = "torch_sparse",
        sparse_format: str = "torch_sparse",
        batch_first: bool = True,
        adjacency_matrix_path: str | None = None,
        self_recurrence: bool = False,
        initialization: str = "identity",
    ):
        """
        Initialize the TopographicalCorticalSheet object.

        Args:
            sheet_size (tuple): The size of the sheet (number of rows, number of columns).
            connectivity_std (float): The standard deviation of the connectivity weights.
            synapses_per_neuron (int): The number of synapses per neuron.
            bias (bool, optional): Whether to include bias or not. Defaults to True.
            mm_function (str, optional): The sparse matrix multiplication function to use.
                Possible values are "native", "torch_sparse", and "tsgu". Defaults to "native".
            sparse_format (str, optional): The sparse format to use.
                Possible values are "coo", "csr", and "torch_sparse". Defaults to "coo".
            batch_first (bool, optional): Whether the batch dimension is the first dimension. Defaults to False.
            **kwargs: Additional keyword arguments (unused)

        Raises:
            ValueError: If an invalid mm_function or sparse_format is provided.
        """
        super().__init__()
        # Save the sparse matrix multiplication function
        self.sparse_format = sparse_format
        self.batch_first = batch_first

        # Select the sparse matrix multiplication function
        if mm_function == "native":
            self.mm_function = torch.sparse.mm
        elif mm_function == "torch_sparse":
            if sparse_format != "torch_sparse":
                raise ValueError(
                    "sparse_format must be 'torch_sparse' if mm_function is 'torch_sparse'"
                )
            self.mm_function = torch_sparse.spmm
        elif mm_function == "tsgu":
            self.mm_function = tsgu.sparse_mm
        else:
            raise ValueError(f"Invalid mm_function: {mm_function}")

        # Load adjacency matrix if provided
        if adjacency_matrix_path is not None:
            adj = torch.load(adjacency_matrix_path).coalesce()
            indices = adj.indices().long()
            # add identity connection to indices
            if self_recurrence:
                identity = indices.unique().tile(2, 1)
                indices = torch.cat([indices, identity], 1)
            _, inv, fan_in = indices[0].unique(
                return_inverse=True, return_counts=True
            )
            scale = torch.sqrt(2 / fan_in.float())
            values = torch.randn(indices.shape[1]) * scale[inv]

        # Create adjacency matrix with normal distribution randomized weights
        else:
            indices = []
            if initialization == "identity":
                values = []
            for i in range(sheet_size[0]):
                for j in range(sheet_size[1]):
                    synapses = (
                        torch.randn(2, synapses_per_neuron) * connectivity_std
                        + torch.tensor((i, j))[:, None]
                    ).long()
                    synapses = torch.cat(
                        [synapses, torch.tensor((i, j))[:, None]], dim=1
                    )
                    synapses = synapses.clamp(
                        torch.tensor((0, 0))[:, None],
                        torch.tensor((sheet_size[0] - 1, sheet_size[1] - 1))[
                            :, None
                        ],
                    )
                    synapses = idx_2D_to_1D(
                        synapses, sheet_size[0], sheet_size[1]
                    )
                    synapse_root = torch.full_like(
                        synapses,
                        int(
                            idx_2D_to_1D(
                                torch.tensor((i, j)),
                                sheet_size[0],
                                sheet_size[1],
                            )
                        ),
                    )
                    indices.append(torch.stack((synapses, synapse_root)))
                    if initialization == "identity":
                        values.append(
                            torch.cat(
                                [
                                    torch.zeros(synapses_per_neuron),
                                    torch.ones(1),
                                ]
                            )
                        )
            indices = torch.cat(indices, dim=1)

            # He initialization of values (synapses_per_neuron is the fan_in)
            if initialization == "he":
                values = torch.randn(indices.shape[1]) * math.sqrt(
                    2 / synapses_per_neuron
                )
            elif initialization == "identity":
                values = torch.cat(values)
            else:
                raise ValueError(f"Invalid initialization: {initialization}")

        self.num_neurons = indices.max().item() + 1

        if sparse_format in ("coo", "csr"):
            weight = torch.sparse_coo_tensor(
                indices,
                values,
                (self.num_neurons, self.num_neurons),  # type: ignore
                check_invariants=True,
            ).coalesce()
            if sparse_format == "csr":
                weight = weight.to_sparse_csr()
        elif sparse_format == "torch_sparse":
            indices, weight = torch_sparse.coalesce(
                indices, values, self.num_neurons, self.num_neurons
            )
            self.indices = nn.Parameter(indices, requires_grad=False)

        else:
            raise ValueError(f"Invalid sparse_format: {sparse_format}")
        self.weight = nn.Parameter(weight)  # type: ignore
        # self.weight.register_hook(lambda grad: grad.coalesce())
        # self.weight.register_hook(lambda grad: print(grad))

        # Initialize the bias vector
        self.bias = (
            nn.Parameter(torch.zeros(self.num_neurons, 1)) if bias else None
        )

    def coalesce(self):
        """
        Coalesce the weight matrix.
        """
        self.weight.data = self.weight.data.coalesce()

    def forward(self, x):
        """
        Forward pass of the TopographicalCorticalSheet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x: Dense (strided) tensor of shape (batch_size, num_neurons) if
        # batch_first, otherwise (num_neurons, batch_size)
        # assert self.weight.is_coalesced()

        # Transpose input if batch_first
        if self.batch_first:
            x = x.t()

        # Perform sparse matrix multiplication with or without bias
        if self.sparse_format == "torch_sparse":
            x = self.mm_function(
                self.indices,
                self.weight,
                self.num_neurons,  # type: ignore
                self.num_neurons,
                x,
            )
        else:
            x = (
                self.mm_function(self.weight, x)  # type: ignore
                if self.bias is None
                else self.mm_function(self.weight, x)  # type: ignore
            )

        # Transpose output back to batch first
        if self.batch_first:
            x = x.t()

        return x


class TopographicalRNN(nn.Module):
    def __init__(
        self,
        sheet_size: tuple[int, int] = (256, 256),
        connectivity_std: float = 10,
        synapses_per_neuron: int = 32,
        bias: bool = True,
        mm_function: str = "torch_sparse",
        sparse_format: str = "torch_sparse",
        batch_first: bool = True,
        adjacency_matrix_path: str = None,
        self_recurrence: bool = False,
        num_timesteps: int = 100,
        input_indices: str | torch.Tensor = None,
        output_indices: str | torch.Tensor = None,
        activation: str = "relu",
        initialization: str = "identity",
    ):
        """
        Initialize the TopographicalCorticalRNN object.

        Args:
            sheet_size (tuple): The size of the cortical sheet (number of rows, number of columns).
            connectivity_std (float): The standard deviation of the connectivity weights.
            synapses_per_neuron (int): The number of synapses per neuron.
            num_timesteps (int): The number of timesteps for the recurrent processing.
            pool_stride (int): The stride for the max pooling operation.
            activation (torch.nn.Module, optional): The activation function to use. Defaults to nn.GELU.
            sheet_bias (bool, optional): Whether to include bias in the cortical sheet. Defaults to True.
            sheet_mm_function (str, optional): The sparse matrix multiplication function to use in the cortical sheet.
                Possible values are "native", "torch_sparse", and "tsgu". Defaults to "native".
            sheet_sparse_format (str, optional): The sparse format to use in the cortical sheet.
                Possible values are "coo", "csr", and "torch_sparse". Defaults to "coo".
            sheet_batch_first (bool, optional): Whether the batch dimension is the first dimension in the cortical sheet. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        if (
            sheet_size is not None
            or connectivity_std is not None
            or synapses_per_neuron is not None
        ) and adjacency_matrix_path is not None:
            warn(
                "If adjacency_matrix_path is provided, sheet_size, connectivity_std, and synapses_per_neuron will be ignored"
            )
        self.num_timesteps = num_timesteps
        self.batch_first = batch_first
        if activation == "gelu":
            activation = nn.GELU
            self.activation = activation()
        elif activation == "relu":
            activation = nn.ReLU
            self.activation = activation()
        else:
            raise ValueError(f"Invalid activation: {activation}")

        if isinstance(input_indices, str):
            if "npy" in input_indices:
                input_indices = np.load(input_indices)
                input_indices = torch.tensor(input_indices)
            else:
                input_indices = torch.load(input_indices)
        elif input_indices is not None or not isinstance(
            input_indices, torch.Tensor
        ):
            raise ValueError(
                "input_indices must be a torch.Tensor or a path to a .npy or .pt file"
            )

        if isinstance(output_indices, str):
            if "npy" in output_indices:
                output_indices = np.load(output_indices)
                output_indices = torch.tensor(output_indices)
            else:
                output_indices = torch.load(output_indices)
        else:
            if output_indices is not None or not isinstance(
                output_indices, torch.Tensor
            ):
                raise ValueError(
                    "output_indices must be a torch.Tensor or a path to a .npy or .pt file"
                )

        self.input_indices = input_indices
        self.output_indices = output_indices

        # Create the CorticalSheet layer
        self.cortical_sheet = TopographicalCorticalCell(
            sheet_size=sheet_size,
            connectivity_std=connectivity_std,
            synapses_per_neuron=synapses_per_neuron,
            bias=bias,
            mm_function=mm_function,
            sparse_format=sparse_format,
            batch_first=False,
            adjacency_matrix_path=adjacency_matrix_path,
            self_recurrence=self_recurrence,
            initialization=initialization,
        )

        num_out_neurons = (
            self.cortical_sheet.num_neurons
            if output_indices is None
            else output_indices.shape[0]
        )

        # Create output block
        self.out_block = nn.Sequential(
            nn.Linear(num_out_neurons, 64),
            activation(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        """
        Forward pass of the TopographicalCorticalRNN.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, num_neurons) or (batch_size, num_channels, num_neurons).

        Returns:
            torch.Tensor: Output tensor.
        """
        # Coallesce weight matrix
        # self.cortical_sheet.coalesce()

        # Average out channel dimension if it exists
        if len(x.shape) > 2:
            x = x.flatten(2)
            x = x.mean(dim=1)

        if self.input_indices is not None:
            input_x = torch.zeros(
                x.shape[0],
                self.cortical_sheet.num_neurons,
                device=x.device,
                dtype=x.dtype,
            )
            input_x[:, self.input_indices] = x
            x = input_x

        # To avoid tranposing x before and after every iteration, we tranpose
        # before and after ALL iterations and do not tranpose within forward()
        # of self.cortical_sheet
        if self.batch_first:
            x = x.t()

        input_x = x

        # Pass the input through the CorticalSheet layer num_timesteps times
        for _ in range(self.num_timesteps):
            x = self.activation(self.cortical_sheet(input_x + x))

        # Transpose back
        if self.batch_first:
            x = x.t()

        if self.output_indices is not None:
            x = x[:, self.output_indices]

        # Return classification from out_block
        return self.out_block(x)
