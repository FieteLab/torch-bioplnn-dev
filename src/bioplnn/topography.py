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
        self.weight.data = self.weight.data.coalesce()

    def idx_1D_to_2D(self, x):
        return torch.stack((x // self.sheet_size[1], x % self.sheet_size[1]))

    def idx_2D_to_1D(self, x):
        return x[0] * self.sheet_size[1] + x[1]

    def forward(self, x):
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
                self.num_neurons,
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
