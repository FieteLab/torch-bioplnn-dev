import torch
import torch.nn as nn


class CorticalSheet(nn.Module):
    def __init__(self, num_neurons, synapses_per_neuron):
        super().__init__()
        # Create a sparse tensor for the weight matrix
        indices = []

        for i in range(num_neurons):
            synapses = torch.randint(0, num_neurons, (synapses_per_neuron,))
            synapse_root = torch.full_like(synapses, i)
            indices.append(torch.stack((synapses, synapse_root)))
        indices = torch.cat(indices, dim=1)
        values = torch.randn(num_neurons * synapses_per_neuron)

        coo_matrix = torch.sparse_coo_tensor(
            indices, values, (num_neurons, num_neurons)
        )
        csr_matrix = coo_matrix.coalesce().to_sparse_csr()
        self.weight = nn.Parameter(csr_matrix)

        # Initialize the bias vector
        self.bias = nn.Parameter(torch.zeros(num_neurons))

    def forward(self, x):
        # Perform sparse matrix multiplication
        output = torch.sparse.mm(self.weight, x)

        # Add the bias vector
        output += self.bias

        return output


class CorticalRNN(nn.Module):
    def __init__(
        self,
        num_neurons,
        synapses_per_neuron,
        num_timesteps,
        activation=nn.GELU,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.activation = activation()

        # Define the CorticalSheet layer
        self.cortical_sheet = CorticalSheet(num_neurons, synapses_per_neuron)

    def forward(self, x):
        # Pass the input through the CorticalSheet layer num_timesteps times
        for _ in range(self.num_timesteps):
            x = self.activation(self.cortical_sheet(x))

        return x
