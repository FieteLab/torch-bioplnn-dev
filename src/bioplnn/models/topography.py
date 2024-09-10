import math
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import animation

from bioplnn.models.sparse import SparseRChebyKAN, SparseRKAN, SparseRNN
from bioplnn.utils import get_activation_class, idx_1D_to_2D, idx_2D_to_1D


class TopographicalRNNBase(nn.Module):
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
        connectivity_hh: Optional[str | torch.Tensor] = None,
        connectivity_ih: Optional[str | torch.Tensor] = None,
        num_classes: int = 10,
        batch_first: bool = True,
        input_indices: Optional[str | torch.Tensor] = None,
        output_indices: Optional[str | torch.Tensor] = None,
        out_nonlinearity: str = "relu",
    ):
        super().__init__()

        self.sheet_size = sheet_size
        self.batch_first = batch_first
        self.out_nonlinearity = get_activation_class(out_nonlinearity)()

        # Check for consistency in connectivity matrix usage
        if (connectivity_ih is None) != (connectivity_hh is None):
            raise ValueError(
                "Both connectivity matrices must be provided if one is provided"
            )

        # Initialize connectivity or randomize based on arguments
        use_random = synapse_std is not None and synapses_per_neuron is not None
        use_connectivity = connectivity_ih is not None and connectivity_hh is not None

        if use_connectivity:
            if use_random:
                warn(
                    "Both random initialization and connectivity initialization are provided. Using connectivity initialization."
                )
            try:
                self.connectivity_hh = torch.load(connectivity_hh)
                self.connectivity_ih = torch.load(connectivity_ih)
            except AttributeError:
                pass
        elif use_random:
            self.connectivity_ih, self.connectivity_hh = self.random_connectivity(
                sheet_size, synapse_std, synapses_per_neuron, self_recurrence
            )
        else:
            raise ValueError(
                "Either connectivity or random initialization must be provided"
            )

        # Validate connectivity matrix format
        if (
            self.connectivity_ih.layout != torch.sparse_coo
            or self.connectivity_hh.layout != torch.sparse_coo
        ):
            raise ValueError("Connectivity matrices must be in COO format")
        if (
            self.connectivity_ih.shape[0]
            != self.connectivity_ih.shape[1]
            != self.connectivity_hh.shape[0]
            != self.connectivity_hh.shape[1]
        ):
            raise ValueError("Connectivity matrices must be square")

        self.num_neurons = self.connectivity_ih.shape[0]

        # Handle input and output indices
        if input_indices is not None:
            try:
                self.input_indices = torch.load(input_indices)
            except AttributeError:
                pass
            if self.input_indices.dim() > 1:
                raise ValueError("Input indices must be a 1D tensor")
        if output_indices is not None:
            try:
                self.output_indices = torch.load(output_indices).squeeze()
            except AttributeError:
                pass
            if self.output_indices.dim() > 1:
                raise ValueError("Output indices must be a 1D tensor")

        if (input_indices is not None and self.input_indices.dim() > 1) or (
            output_indices is not None and self.output_indices.dim() > 1
        ):
            raise ValueError("Input and output indices must be 1D tensors")

        num_out_neurons = (
            self.num_neurons if output_indices is None else self.output_indices.shape[0]
        )

        # Create output block
        self.out_layer = nn.Sequential(
            nn.Linear(num_out_neurons, 64),
            self.out_nonlinearity,
            nn.Linear(64, num_classes),
        )

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

        indices = torch.stack((synapses, synapse_root)).flatten(1)

        # He initialization of values (synapses_per_neuron is the fan_in)
        values_ih = torch.randn(indices.shape[1]) * math.sqrt(2 / synapses_per_neuron)
        values_hh = torch.randn(indices.shape[1]) * math.sqrt(2 / synapses_per_neuron)

        m = n = math.prod(sheet_size)

        connectivity_ih = torch.sparse_coo_tensor(
            indices,
            values_ih,
            (m, n),
            check_invariants=True,
        ).coalesce()

        connectivity_hh = torch.sparse_coo_tensor(
            indices,
            values_hh,
            (m, n),
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
        x,
        num_steps=None,
        loss_all_timesteps=False,
    ):
        """
        Forward pass of the TopographicalRNNBase.

        Args:
            x (torch.Tensor): Input tensor.
            num_steps (int, optional): Number of time steps. Defaults to None.
            loss_all_timesteps (bool, optional): Whether to calculate loss for all timesteps or only the last one. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and hidden state.
        """
        # TODO: Add sparse-dense hybrid functionality for channels
        if self.batch_first:
            if x.dim() == 2:
                T = 1
            else:
                T = x.shape[1]
            batch_size = x.shape[0]
        else:
            if x.dim() == 2:
                T = 1
            else:
                T = x.shape[0]
            batch_size = x.shape[1]

        if self.input_indices is not None:
            xs = []
            for t in range(T):
                x_t = torch.zeros(
                    batch_size,
                    self.num_neurons,
                    device=x.device,
                    dtype=x.dtype,
                )
                if x.dim() == 3:
                    if self.batch_first:
                        x_t[:, self.input_indices] = x[:, t]
                    else:
                        x_t[:, self.input_indices] = x[t]
                else:
                    x_t[:, self.input_indices] = x
                xs.append(x_t)
            x = torch.stack(xs).squeeze()

        x, h = self.rnn(x, num_steps)

        if self.batch_first:
            x = x.transpose(0, 1)

        # Select output indices if provided
        if self.output_indices is not None:
            x = x[..., self.output_indices]

        if loss_all_timesteps:
            return [self.out_layer(out) for out in x]

        return self.out_layer(x[-1]), h


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
