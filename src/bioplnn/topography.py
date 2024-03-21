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


class TopographicalCorticalCell(nn.Module):
    def __init__(
        self,
        sheet_size: tuple[int, int] = (100, 100),
        connectivity_std: float = 10,
        synapses_per_neuron: int = 32,
        bias: bool = True,
        mm_function: str = "torch_sparse",
        sparse_format: str = None,
        batch_first: bool = True,
        adjacency_matrix_path: str | None = None,
        self_recurrence: bool = False,
    ):
        """
        Initialize the TopographicalCorticalCell object.

        Args:
            sheet_size (tuple): The size of the sheet (number of rows, number of columns).
            connectivity_std (float): The standard deviation of the connectivity weights.
            synapses_per_neuron (int): The number of synapses per neuron.
            bias (bool, optional): Whether to include bias or not. Defaults to True.
            mm_function (str, optional): The sparse matrix multiplication function to use.
                Possible values are  'torch_sparse', 'native', and 'tsgu'. Defaults to 'torch_sparse'.
            sparse_format (str, optional): The sparse format to use.
                Possible values are 'coo' and 'csr'. Defaults to 'coo'.
            batch_first (bool, optional): Whether the batch dimension is the first dimension. Defaults to False.
            adjacency_matrix_path (str, optional): The path to the adjacency matrix file. Defaults to None.
            self_recurrence (bool, optional): Whether to include self-recurrence connections. Defaults to False.

        Raises:
            ValueError: If an invalid mm_function or sparse_format is provided.
        """
        super().__init__()
        # Save the sparse matrix multiplication function
        self.sheet_size = sheet_size
        self.sparse_format = sparse_format
        self.batch_first = batch_first

        # Select the sparse matrix multiplication function
        if mm_function == "torch_sparse":
            if sparse_format is not None and sparse_format != "torch_sparse":
                raise ValueError(
                    "sparse_format must not be specified if mm_function is 'torch_sparse'"
                )
            self.sparse_format = "torch_sparse"
            self.mm_function = torch_sparse.spmm
        elif mm_function in ("native", "tsgu"):
            if sparse_format not in ("coo", "csr"):
                raise ValueError(
                    "sparse_format must be 'coo' or 'csr' if mm_function is 'native' or 'tsgu'"
                )
            self.mm_function = torch.sparse.mm
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
            _, inv, fan_in = indices[0].unique(return_inverse=True, return_counts=True)
            scale = torch.sqrt(2 / fan_in.float())
            values = torch.randn(indices.shape[1]) * scale[inv]

        # Create adjacency matrix with normal distribution randomized weights
        else:
            indices = []
            # if initialization == "identity":
            #     values = []
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
                        torch.tensor((sheet_size[0] - 1, sheet_size[1] - 1))[:, None],
                    )
                    synapses = idx_2D_to_1D(synapses, sheet_size[0], sheet_size[1])
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
                    # if initialization == "identity":
                    #     values.append(
                    #         torch.cat(
                    #             [
                    #                 torch.zeros(synapses_per_neuron),
                    #                 torch.ones(1),
                    #             ]
                    #         )
                    #     )
            indices = torch.cat(indices, dim=1)

            # He initialization of values (synapses_per_neuron is the fan_in)
            # if initialization == "he":
            values = torch.randn(indices.shape[1]) * math.sqrt(2 / synapses_per_neuron)
            # elif initialization == "identity":
            #     values = torch.cat(values)
            # else:
            #     raise ValueError(f"Invalid initialization: {initialization}")

        self.num_neurons = self.sheet_size[0] * self.sheet_size[1]

        if mm_function == "torch_sparse":
            indices, weight = torch_sparse.coalesce(
                indices, values, self.num_neurons, self.num_neurons
            )
            self.indices = nn.Parameter(indices, requires_grad=False)
        else:
            weight = torch.sparse_coo_tensor(
                indices,
                values,
                (self.num_neurons, self.num_neurons),  # type: ignore
                check_invariants=True,
            ).coalesce()
            if sparse_format == "csr":
                weight = weight.to_sparse_csr()
        self.weight = nn.Parameter(weight)  # type: ignore

        # Initialize the bias vector
        self.bias = nn.Parameter(torch.zeros(self.num_neurons, 1)) if bias else None

    def coalesce(self):
        """
        Coalesce the weight matrix.
        """
        self.weight.data = self.weight.data.coalesce()

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

        # Transpose input if batch_first
        if self.batch_first:
            x = x.t()

        # Perform sparse matrix multiplication with or without bias
        if self.sparse_format == "torch_sparse":
            x = (
                self.mm_function(
                    self.indices,
                    self.weight,
                    self.num_neurons,  # type: ignore
                    self.num_neurons,
                    x,
                )
                + self.bias
            )
        else:
            x = self.mm_function(self.weight, x) + self.bias

        # Transpose output back to batch first
        if self.batch_first:
            x = x.t()

        return x


class TopographicalRNN(nn.Module):
    def __init__(
        self,
        sheet_size: tuple[int, int] = (150, 300),
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
    ):
        """
        Initialize the TopographicalCorticalRNN object.

        Args:
            sheet_size (tuple[int, int], optional): The size of the cortical sheet (number of rows, number of columns). Defaults to (256, 256).
            connectivity_std (float, optional): The standard deviation of the connectivity weights. Defaults to 10.
            synapses_per_neuron (int, optional): The number of synapses per neuron. Defaults to 32.
            bias (bool, optional): Whether to include bias in the cortical sheet. Defaults to True.
            mm_function (str, optional): The sparse matrix multiplication function to use in the cortical sheet.
                Possible values are "native", "torch_sparse", and "tsgu". Defaults to "torch_sparse".
            sparse_format (str, optional): The sparse format to use in the cortical sheet.
                Possible values are "coo", "csr", and "torch_sparse". Defaults to "torch_sparse".
            batch_first (bool, optional): Whether the batch dimension is the first dimension in the cortical sheet. Defaults to True.
            adjacency_matrix_path (str, optional): The path to the adjacency matrix file. Defaults to None.
            self_recurrence (bool, optional): Whether to include self-recurrence in the cortical sheet. Defaults to False.
            num_timesteps (int, optional): The number of timesteps for the recurrent processing. Defaults to 100.
            input_indices (str | torch.Tensor, optional): The input indices for the cortical sheet.
                Can be a path to a .npy or .pt file or a torch.Tensor. Defaults to None.
            output_indices (str | torch.Tensor, optional): The output indices for the cortical sheet.
                Can be a path to a .npy or .pt file or a torch.Tensor. Defaults to None.
            activation (str, optional): The activation function to use. Possible values are "relu" and "gelu". Defaults to "relu".
            initialization (str, optional): The initialization method for the cortical sheet. Defaults to "identity".
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
        self.activation = get_activation_class(activation)()

        if isinstance(input_indices, str):
            if input_indices.endswith("npy"):
                input_indices = np.load(input_indices)
                input_indices = torch.tensor(input_indices)
            else:
                input_indices = torch.load(input_indices)

        if isinstance(output_indices, str):
            if output_indices.endswith("npy"):
                output_indices = np.load(output_indices)
                output_indices = torch.tensor(output_indices)
            else:
                output_indices = torch.load(output_indices)

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
        )

        num_out_neurons = (
            self.cortical_sheet.num_neurons
            if output_indices is None
            else output_indices.shape[0]
        )

        # Create output block
        self.out_block = nn.Sequential(
            nn.Linear(num_out_neurons, 64),
            self.activation,
            nn.Linear(64, 10),
        )

    def visualize(self, activations, save_path=None, fps=4, frames=None):
        """
        Visualize the forward pass of the TopographicalCorticalRNN.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, num_neurons) or (batch_size, num_channels, num_neurons).

        Returns:
            torch.Tensor: Output tensor.
        """
        if frames is not None:
            activations = activations[frames[0] : frames[1]]
        for i in range(len(activations)):
            activations[i] = activations[i][0].reshape(*self.cortical_sheet.sheet_size)

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(8, 8))

        im = plt.imshow(
            activations[0], interpolation="none", aspect="auto", vmin=0, vmax=1
        )

        def animate_func(i):
            if i % fps == 0:
                print(".", end="")

            im.set_array(activations[i])
            return [im]

        anim = animation.FuncAnimation(
            fig,
            animate_func,
            frames=len(activations),
            interval=1000 / fps,  # in ms
        )

        if save_path is not None:
            anim.save(
                save_path,
                fps=fps,
            )
        else:
            plt.show()

    def forward(
        self,
        x,
        visualize=False,
        visualization_save_path=None,
        visualization_fps=4,
        visualization_frames=None,
        return_activations=False,
    ):
        """
        Forward pass of the TopographicalCorticalRNN.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, num_neurons) or (batch_size, num_channels, num_neurons).

        Returns:
            torch.Tensor: Output tensor.
        """

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
        if visualize or return_activations:
            activations = [x.t().detach().cpu()]

        for _ in range(self.num_timesteps):
            x = self.activation(self.cortical_sheet(input_x + x))
            if visualize or return_activations:
                activations.append(x.t().detach().cpu())

        # Transpose back
        if self.batch_first:
            x = x.t()

        # Select output indices if provided
        if self.output_indices is not None:
            x = x[:, self.output_indices]

        # Visualize if required
        if visualize:
            self.visualize(
                activations,
                visualization_save_path,
                visualization_fps,
                visualization_frames,
            )

        # Return classification from out_block
        if return_activations:
            return self.out_block(x), activations
        else:
            return self.out_block(x)


if __name__ == "__main__":
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "topography_trainer.py")) as file:
        exec(file.read())
