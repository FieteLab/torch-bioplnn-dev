import math
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_sparse
import torchsparsegradutils as tsgu
from matplotlib import animation

from bioplnn.models.sparse_rnn import SparseRNN
from bioplnn.utils import get_activation_class, idx_1D_to_2D, idx_2D_to_1D


class TopographicalRNN(nn.Module):
    def __init__(
        self,
        sheet_size: tuple[int, int] = (150, 300),
        synapse_std: float = 10,
        synapses_per_neuron: int = 32,
        self_recurrence: bool = True,
        connectivity_hh: Optional[str | torch.Tensor] = None,
        connectivity_ih: Optional[str | torch.Tensor] = None,
        sparse_layout: str = "torch_sparse",
        mm_function: str = "torch_sparse",
        num_classes: int = 10,
        batch_first: bool = True,
        input_indices: Optional[str | torch.Tensor] = None,
        output_indices: Optional[str | torch.Tensor] = None,
        nonlinearity: str = "relu",
        bias: bool = True,
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
            sparse_layout (str, optional): The sparse format to use in the cortical sheet.
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

        self.batch_first = batch_first
        self.nonlinearity = get_activation_class(nonlinearity)()

        use_random = synapse_std is not None and synapses_per_neuron is not None
        use_connectivity = connectivity_hh is not None and connectivity_ih is not None
        if use_connectivity:
            if use_random:
                warn(
                    "Both random initialization and connectivity initialization are provided. Using connectivity initialization."
                )
                use_random = False
            try:
                connectivity_hh = torch.load(connectivity_hh)
                connectivity_ih = torch.load(connectivity_ih)
            except Exception:
                pass

            if connectivity_ih.layout != "coo" or connectivity_hh.layout != "coo":
                raise ValueError("Connectivity matrices must be in COO format")
            if (
                connectivity_ih.shape[0]
                != connectivity_ih.shape[1]
                != connectivity_hh.shape[0]
                != connectivity_hh.shape[1]
            ):
                raise ValueError("Connectivity matrices must be square")

            self.num_neurons = connectivity_ih.shape[0]

        elif use_random:
            connectivity_ih, connectivity_hh = self.random_connectivity(
                sheet_size, synapse_std, synapses_per_neuron, self_recurrence
            )
            self.num_neurons = sheet_size[0] * sheet_size[1]
        else:
            raise ValueError(
                "Either connectivity or random initialization must be provided"
            )

        try:
            try:
                input_indices = torch.load(input_indices)
            except Exception:
                input_indices = np.load(input_indices)
                input_indices = torch.tensor(input_indices)
        except Exception:
            pass

        try:
            try:
                output_indices = torch.load(output_indices)
            except Exception:
                output_indices = np.load(output_indices)
                output_indices = torch.tensor(output_indices)
        except Exception:
            pass

        if (input_indices is not None and input_indices.dim() > 1) or (
            output_indices is not None and output_indices.dim() > 1
        ):
            raise ValueError("Input and output indices must be 1D tensors")

        self.input_indices = input_indices
        self.output_indices = output_indices

        # Create the CorticalSheet layer

        self.rnn = SparseRNN(
            self.num_neurons,
            self.num_neurons,
            connectivity_ih,
            connectivity_hh,
            num_layers=1,
            sparse_layout=sparse_layout,
            mm_function=mm_function,
            batch_first=batch_first,
            nonlinearity=nonlinearity,
            bias=bias,
        )
        num_out_neurons = (
            self.num_neurons if output_indices is None else output_indices.shape[0]
        )

        # Create output block
        self.out_block = nn.Sequential(
            nn.Linear(num_out_neurons, 64),
            self.activation,
            nn.Linear(64, num_classes),
        )

    def random_connectivity(
        self, sheet_size, synapse_std, synapses_per_neuron, self_recurrence
    ):
        num_neurons = sheet_size[0] * sheet_size[1]

        idx_1d = torch.arange(num_neurons)
        idx = idx_1D_to_2D(idx_1d, sheet_size[0], sheet_size[1]).t()
        synapses = (
            torch.randn(num_neurons, 2, synapses_per_neuron) * synapse_std
            + idx.unsqueeze(-1)
        ).long()
        if self_recurrence:
            synapses = torch.cat([synapses, idx.unsqueeze(-1)], dim=2)
        synapses = synapses.clamp(
            torch.zeros(2).view(1, 2, 1),
            torch.tensor((sheet_size[0] - 1, sheet_size[1] - 1)).view(1, 2, 1),
        )
        synapses = idx_2D_to_1D(
            synapses.transpose(0, 1).flatten(1), sheet_size[0], sheet_size[1]
        ).view(num_neurons, -1)

        synapse_root = idx_1d.expand(-1, synapses_per_neuron + 1)

        indices = torch.stack((synapses, synapse_root)).flatten(1)

        # He initialization of values (synapses_per_neuron is the fan_in)
        values_ih = torch.randn(indices.shape[1]) * math.sqrt(2 / synapses_per_neuron)
        values_hh = torch.randn(indices.shape[1]) * math.sqrt(2 / synapses_per_neuron)

        connectivity_ih = torch.sparse_coo_tensor(
            indices,
            values_ih,
            (math.prod(sheet_size), math.prod(sheet_size)),  # type: ignore
            check_invariants=True,
        ).coalesce()

        connectivity_hh = torch.sparse_coo_tensor(
            indices,
            values_hh,
            (math.prod(sheet_size), math.prod(sheet_size)),  # type: ignore
            check_invariants=True,
        ).coalesce()

        return connectivity_ih, connectivity_hh

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
        num_steps=None,
        return_activations=False,
    ):
        """
        Forward pass of the TopographicalCorticalRNN.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, num_neurons) or (batch_size, num_channels, num_neurons).

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.batch_first:
            B, L, H = x.shape
        else:
            L, B, H = x.shape

        if self.input_indices is not None:
            input_x = torch.zeros(
                B,
                self.num_neurons,
                device=x.device,
                dtype=x.dtype,
            )
            input_x[:, self.input_indices] = x
            x = input_x

        ret = self.rnn(x, num_steps, return_activations=return_activations)
        if return_activations:
            out, _, activations = ret
        else:
            out, _ = ret

        # Select output indices if provided
        if self.output_indices is not None:
            x = x[:, self.output_indices]

        # Return classification from out_block
        if return_activations:
            return self.out_block(x), activations
        return self.out_block(x)
