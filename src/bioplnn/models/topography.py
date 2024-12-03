import math
from os import PathLike
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import animation

from bioplnn.models.sparse import SparseRNN
from bioplnn.utils import idx_1D_to_2D, idx_2D_to_1D


class TopographicalRNN(SparseRNN):
    """
    Base class for Topographical Recurrent Neural Networks (TRNNs).

    TRNNs are a type of recurrent neural network designed to model spatial dependencies on a sheet-like topology.
    This base class provides common functionalities for all TRNN variants.

    Args:
        sheet_size (tuple[int, int]): Size of the sheet-like topology (height, width).
        synapse_std (float): Standard deviation for random synapse initialization.
        synapses_per_neuron (int): Number of synapses per neuron.
        self_recurrence (bool): Whether to include self-recurrent connections.
        connectivity_hh (Optional[str | torch.Tensor]): Path to a file containing the hidden-to-hidden connectivity matrix or the matrix itself.
        connectivity_ih (Optional[str | torch.Tensor]): Path to a file containing the input-to-hidden connectivity matrix or the matrix itself.
        num_classes (int): Number of output classes.
        batch_first (bool): Whether the input is in (batch_size, seq_len, input_size) format.
        input_indices (Optional[str | torch.Tensor]): Path to a file containing the input indices or the tensor itself (specifying which neurons receive input).
        output_indices (Optional[str | torch.Tensor]): Path to a file containing the output indices or the tensor itself (specifying which neurons contribute to the output).
        out_nonlinearity (str): Nonlinearity applied to the output layer.
    """

    def __init__(
        self,
        sheet_size: tuple[int, int] = (150, 300),
        synapse_std: float = 10,
        synapses_per_neuron: int = 32,
        self_recurrence: bool = True,
        connectivity_hh: Optional[PathLike | torch.Tensor] = None,
        connectivity_ih: Optional[PathLike | torch.Tensor] = None,
        batch_first: bool = True,
        input_indices: Optional[PathLike | torch.Tensor] = None,
        output_indices: Optional[PathLike | torch.Tensor] = None,
        hidden_init_mode: str = "zeros",
        nonlinearity: str = "tanh",
        use_layernorm: bool = False,
        bias: bool = True,
    ):
        self.sheet_size = sheet_size
        self.batch_first = batch_first

        # Handle input and output indices
        self.input_indices, self.output_indices = self._init_indices(
            input_indices, output_indices
        )

        self.input_size = (
            len(self.input_indices)
            if self.input_indices is not None
            else self.num_neurons
        )

        self.connectivity_ih, self.connectivity_hh = self._init_connectivity(
            connectivity_ih,
            connectivity_hh,
            sheet_size,
            synapse_std,
            synapses_per_neuron,
            self_recurrence,
        )
        # Check for consistency in connectivity matrix usage

        self.num_neurons = self.connectivity_hh.shape[0]

        super().__init__(
            input_size=self.input_size,
            hidden_size=self.num_neurons,
            connectivity_ih=self.connectivity_ih,
            connectivity_hh=self.connectivity_hh,
            hidden_init_mode=hidden_init_mode,
            use_layernorm=use_layernorm,
            nonlinearity=nonlinearity,
            batch_first=batch_first,
            bias=bias,
        )

        # Time constant
        self.tau = nn.Parameter(
            torch.ones(self.num_neurons), requires_grad=True
        )

    def _init_indices(self, input_indices, output_indices):
        if input_indices is not None:
            try:
                input_indices = torch.load(input_indices).squeeze()
            except AttributeError:
                pass
            if input_indices.dim() > 1:
                raise ValueError("Input indices must be a 1D tensor")
        if output_indices is not None:
            try:
                output_indices = torch.load(output_indices).squeeze()
            except AttributeError:
                pass
            if output_indices.dim() > 1:
                raise ValueError("Output indices must be a 1D tensor")

        return input_indices, output_indices

    def _init_connectivity(
        self,
        connectivity_ih,
        connectivity_hh,
        sheet_size,
        synapse_std,
        synapses_per_neuron,
        self_recurrence,
    ):
        if (connectivity_ih is None) != (connectivity_hh is None):
            raise ValueError(
                "Both connectivity matrices must be provided if one is provided"
            )

        # Initialize connectivity or randomize based on arguments
        use_random = (
            sheet_size is not None
            and synapse_std is not None
            and synapses_per_neuron is not None
        )
        use_connectivity = (
            connectivity_ih is not None and connectivity_hh is not None
        )

        if use_connectivity:
            if use_random:
                warn(
                    "Both random initialization and connectivity initialization are provided. Using connectivity initialization."
                )
            try:
                connectivity_ih = torch.load(connectivity_ih)
            except AttributeError:
                pass
            try:
                connectivity_hh = torch.load(connectivity_hh)
            except AttributeError:
                pass
            if (
                connectivity_ih.layout != torch.sparse_coo
                or connectivity_hh.layout != torch.sparse_coo
            ):
                raise ValueError("Connectivity matrices must be in COO format")
        elif use_random:
            connectivity_ih, connectivity_hh = self.random_connectivity(
                sheet_size, synapse_std, synapses_per_neuron, self_recurrence
            )
        else:
            raise ValueError(
                "Either connectivity or random initialization must be provided"
            )

        # Validate connectivity matrix format
        if (
            self.input_indices is not None
            and connectivity_ih.shape[1] != self.input_indices.shape[0]
        ):
            raise ValueError(
                "connectivity_ih.shape[1] and input_indices.shape[0] do not match"
            )
        if not (
            connectivity_ih.shape[0]
            == connectivity_hh.shape[0]
            == connectivity_hh.shape[1]
        ):
            raise ValueError(
                "connectivity_ih.shape[0], connectivity_hh.shape[0], and connectivity_hh.shape[1] must be equal"
            )

        return connectivity_ih, connectivity_hh

    def random_connectivity(
        self, sheet_size, synapse_std, synapses_per_neuron, self_recurrence
    ):
        """
        Generates random connectivity matrices for the TRNN.

        Args:
            sheet_size (tuple[int, int]): Size of the sheet-like topology.
            synapse_std (float): Standard deviation for random synapse initialization.
            synapses_per_neuron (int): Number of synapses per neuron.
            self_recurrence (bool): Whether to include self-recurrent connections.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Connectivity matrices for input-to-hidden and hidden-to-hidden connections.
        """
        # Generate random connectivity for hidden-to-hidden connections
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

        synapse_root = idx_1d.unsqueeze(-1).expand(-1, synapses.shape[1])

        indices_hh = torch.stack((synapses, synapse_root)).flatten(1)

        ## He initialization of values (synapses_per_neuron is the fan_in)
        values_hh = torch.randn(indices_hh.shape[1]) * math.sqrt(
            2 / synapses_per_neuron
        )

        connectivity_hh = torch.sparse_coo_tensor(
            indices_hh,
            values_hh,
            (num_neurons, num_neurons),
            check_invariants=True,
        ).coalesce()

        # Generate random connectivity for input-to-hidden connections

        indices_ih = torch.stack(
            (
                self.input_indices
                if self.input_indices is not None
                else torch.arange(self.input_size),
                torch.arange(self.input_size),
            )
        )

        values_ih = torch.randn(indices_ih.shape[1]) * math.sqrt(
            2 / synapses_per_neuron
        )

        connectivity_ih = torch.sparse_coo_tensor(
            indices_ih,
            values_ih,
            (num_neurons, self.input_size),
            check_invariants=True,
        ).coalesce()

        return connectivity_ih, connectivity_hh

    def visualize(self, activations, save_path=None, fps=4, frames=None):
        """
        Visualizes the activations of the TopographicalRNN as an animation.

        Args:
            activations (torch.Tensor): Tensor of activations.
            save_path (str, optional): Path to save the animation. Defaults to None.
            fps (int, optional): Frames per second for the animation. Defaults to 4.
            frames (tuple[int, int], optional): Range of frames to visualize. Defaults to None.
        """
        if frames is not None:
            activations = activations[frames[0] : frames[1]]
        for i in range(len(activations)):
            activations[i] = activations[i][0].reshape(*self.sheet_size)

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
            plt.show(block=False)
            plt.pause(5)
            plt.close()

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        h_0: Optional[torch.Tensor] = None,
    ):
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

        hs = self._init_state(
            h_0,
            num_steps,
            batch_size,
            device=device,
        )
        # Process input sequence
        for t in range(num_steps):
            hs[t] = self.nonlinearity(self.ih(x[t]) + self.hh(hs[t - 1]))
            hs[t] = self.layernorm(hs[t])
            tau = self.sigmoid(self.tau)
            hs[t] = tau * hs[t] + (1 - tau) * hs[t - 1]

        # Stack outputs and adjust dimensions if necessary
        hs = self._format_result(hs)

        # Select output indices if provided
        if self.output_indices is not None:
            outs = hs[..., self.output_indices]

        return outs, hs
