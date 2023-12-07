
import torch
from torch import nn

class LIF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
    
class Excitatory(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class Inhibitory(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class CorticalSheet(nn.Module):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        
        self.connectivity = torch.sparse_coo_tensor()
        

    def forward(self, x):
        pass
    

