import torch
import torch.nn as nn
import torch_sparse
import torchsparsegradutils as tsgu
import math


class TopographicalCorticalSheet(nn.Module):
    def __init__(
        self,
        sheet_size,
        connectivity_std,
        synapses_per_neuron,
        bias=True,
        mm_function="native",
        sparse_format="coo",
        batch_first=False,
        **kwargs,
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
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If an invalid mm_function or sparse_format is provided.
        """
        super().__init__()
        # Save the sparse matrix multiplication function
        self.sheet_size = sheet_size
        self.num_neurons = sheet_size[0] * sheet_size[1]
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

        # Create adjacency matrix with normal distribution randomized weights
        indices = []
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
                synapses = self.idx_2D_to_1D(synapses)
                synapse_root = torch.full_like(
                    synapses, self.idx_2D_to_1D(torch.tensor((i, j)))
                )
                indices.append(torch.stack((synapses, synapse_root)))
        indices = torch.cat(indices, dim=1)
        # Sort indices by synapses
        # indices = indices[:, torch.argsort(indices[0])]
        # Xavier initialization of values (synapses_per_neuron is the fan-in/out)
        values = torch.randn(indices.shape[1]) * math.sqrt(
            1 / synapses_per_neuron
        )
        if sparse_format in ("coo", "csr"):
            weight = torch.sparse_coo_tensor(
                indices,
                values,
                (self.num_neurons, self.num_neurons),
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

    def idx_1D_to_2D(self, x):
        """
        Convert a 1D index to a 2D index.

        Args:
            x (torch.Tensor): 1D index.

        Returns:
            torch.Tensor: 2D index.
        """
        return torch.stack((x // self.sheet_size[1], x % self.sheet_size[1]))

    def idx_2D_to_1D(self, x):
        """
        Convert a 2D index to a 1D index.

        Args:
            x (torch.Tensor): 2D index.

        Returns:
            torch.Tensor: 1D index.
        """
        return x[0] * self.sheet_size[1] + x[1]

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


class TopographicalCorticalRNN(nn.Module):
    def __init__(
        self,
        sheet_size,
        connectivity_std,
        synapses_per_neuron,
        num_timesteps,
        pool_stride,
        activation=nn.GELU,
        sheet_bias=True,
        sheet_mm_function="native",
        sheet_sparse_format="coo",
        sheet_batch_first=False,
        **kwargs,
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
        self.num_neurons = sheet_size[0] * sheet_size[1]
        self.sheet_size = sheet_size
        self.num_timesteps = num_timesteps
        self.activation = activation()
        self.sheet_batch_first = sheet_batch_first

        # Create the CorticalSheet layer
        self.cortical_sheet = TopographicalCorticalSheet(
            sheet_size,
            connectivity_std,
            synapses_per_neuron,
            sheet_bias,
            sheet_mm_function,
            sheet_sparse_format,
            sheet_batch_first,
        )

        self.pool = nn.MaxPool2d(pool_stride, pool_stride)

        # Create output block
        self.out_block = nn.Sequential(
            nn.Linear(self.num_neurons // (pool_stride**2), 1024),
            activation(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        """
        Forward pass of the TopographicalCorticalRNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x: Dense (strided) tensor of shape (batch_size, 1, 32, 32)
        batch_size = x.shape[0]

        # Coallesce weight matrix
        # self.cortical_sheet.coalesce()

        # Flatten spatial and channel dimensions
        x = x.view(batch_size, -1)

        # To avoid tranposing x before and after every iteration, we tranpose
        # before and after ALL iterations and do not tranpose within forward()
        # of self.cortical_sheet
        if not self.sheet_batch_first:
            x = x.t()

        # Pass the input through the CorticalSheet layer num_timesteps times
        for _ in range(self.num_timesteps):
            x = self.activation(self.cortical_sheet(x))

        # Transpose back
        if not self.sheet_batch_first:
            x = x.t()

        # Apply pooling
        x = self.pool(x.view(batch_size, 1, *self.sheet_size)).view(
            batch_size, -1
        )

        # Return classification from out_block
        return self.out_block(x.flatten(1))
