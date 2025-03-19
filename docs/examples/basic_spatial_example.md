# Basic API usage example

## Overview

The goal of this tutorial is to wire up a simple model that is comprised of a spatially embedded brain area, specify a simple learning objective, and optimize model parameters.

Before we begin, here are a few notes pertaining to the nomenclature we have adopted.

- A neuron ***class*** is defined by its synaptic affiliation. In `torch-biopl` you can configure types to be `Excitatory`/`Inhibitory` (where synapses have a postive/negative sign), or `Hybrid` which defaults to standard machine learning-style synapses that are unconstrained.
- Within each neuron class, you can instantiate neuron ***types***. We employ the definition of neuron types with an eye to be able to specify inter-type local connectivity rules. 
- Within each neuron type, are ***subtypes***. Neurons within a subtype can (but don't have to) share properties like time constants, nonlinearities, and biases.
- Each neural area can be configured independently by specifying its classes, types, subtypes, and inter-type connectivity rules.
- Areas can be stitched together to form larger networks.
- Learnable parameters in `torch-biopl` are usually the synaptic weights, neural time constants, and biases.


```python
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np

from bioplnn.models import SpatiallyEmbeddedClassifier
```


```python
# Torch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
```

Let us wire up a simple one-area network with two neural classes. Let one of them be an `Excitatory` cell class and one be `Inhibitory` each with `16` subtypes. Now, we have our neural populations. All that's left to do is to specify the inter-type connectivity. 

In `torch-biopl` we adopt the following convention:

- Inter-celltype connectivity (within a given area) is specified through an adjacency matrix.
- In addition to the neuron types within an area, we also have to account for projections into and out of the area. Keeping this in mind, we use a schema where rows in the adjacency matrix represent the ***pre-synaptic*** neuron type and columns represent the ***post-synaptic*** neuron type. 
- The **first row** always denotes projections into the area, and the **last column** always denotes feedforward projections out of the area.

For example, if our neuron_type_1 is E and neuron_type_2 is I, then `inter_neuron_type_connectivity`= $\begin{bmatrix} 1 & 1 & 0 \cr 1 & 1 & 1 \cr 1 & 1 & 0 \end{bmatrix}$ represents a standard recurrent inhibitory circuit motif (ala [Wong et al. (2006)](https://pubmed.ncbi.nlm.nih.gov/16436619/)), where both the E and I populations receive input, and only the E population projects downstream.

Since we plan to train on grayscale images in this example, `in_channels` = 1


```python
# Model setup
model = SpatiallyEmbeddedClassifier(
    rnn_kwargs={
        "num_areas": 1,
        "area_kwargs": [
            {
                "num_neuron_types": 2,
                "num_neuron_subtypes": np.array([16, 16]),
                "neuron_type_class": np.array(['excitatory', 'inhibitory']),
                "inter_neuron_type_connectivity": np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]]),
                "in_size": [28, 28],
                "in_channels": 1,
                "out_channels": 32,
            },
        ],
    },
    num_classes = 10,
    fc_dim = 256,
    dropout = 0.5,
).to(device)
```


```python
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.CrossEntropyLoss()
```


```python
# Dataloader setup
transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train_data = MNIST(root="data", train=True, transform=transform)
train_loader = DataLoader(
    train_data, batch_size=256, num_workers=8, shuffle=True
)
```

    /scratch2/weka/mcdermott/lakshmin/conda_envs/test_bioplnn_env/lib/python3.12/site-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(



```python
# Define the training loop
model.train()
n_epochs = 10
log_frequency = 100

running_loss, running_correct, running_total = 0, 0, 0
for epoch in range(n_epochs):
    for i, (x, labels) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        labels = labels.to(device)
        torch._inductor.cudagraph_mark_step_begin()
        logits = model(x, num_steps=5)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate running accuracy and loss
        _, predicted = torch.max(logits, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        running_acc = running_correct / running_total
        if (i + 1)%log_frequency == 0:
            print(
                f"Training | Epoch: {epoch} | " +
                f"Loss: {running_loss:.4f} | " +
                f"Acc: {running_acc:.2%}"
            )
            running_loss, running_correct, running_total = 0, 0, 0
```

     43%|████▎     | 101/235 [00:10<00:06, 21.68it/s]

    Training | Epoch: 0 | Loss: 230.5550 | Acc: 11.30%


     86%|████████▌ | 202/235 [00:16<00:01, 18.90it/s]

    Training | Epoch: 0 | Loss: 220.1151 | Acc: 20.44%


    100%|██████████| 235/235 [00:17<00:00, 13.13it/s]
     43%|████▎     | 102/235 [00:10<00:06, 21.96it/s]

    Training | Epoch: 1 | Loss: 257.5806 | Acc: 26.85%


     86%|████████▌ | 201/235 [00:15<00:01, 20.88it/s]

    Training | Epoch: 1 | Loss: 180.8878 | Acc: 28.72%


    100%|██████████| 235/235 [00:16<00:00, 13.97it/s]
     43%|████▎     | 102/235 [00:09<00:06, 22.11it/s]

    Training | Epoch: 2 | Loss: 237.9631 | Acc: 30.28%


     87%|████████▋ | 204/235 [00:14<00:01, 20.53it/s]

    Training | Epoch: 2 | Loss: 171.1650 | Acc: 32.17%


    100%|██████████| 235/235 [00:15<00:00, 14.71it/s]
     44%|████▍     | 104/235 [00:09<00:06, 21.30it/s]

    Training | Epoch: 3 | Loss: 222.0076 | Acc: 35.64%


     87%|████████▋ | 204/235 [00:14<00:01, 21.45it/s]

    Training | Epoch: 3 | Loss: 157.9773 | Acc: 38.38%


    100%|██████████| 235/235 [00:16<00:00, 14.59it/s]
     44%|████▍     | 104/235 [00:09<00:06, 20.57it/s]

    Training | Epoch: 4 | Loss: 205.7286 | Acc: 40.15%


     86%|████████▌ | 202/235 [00:14<00:01, 20.17it/s]

    Training | Epoch: 4 | Loss: 148.0692 | Acc: 42.49%


    100%|██████████| 235/235 [00:15<00:00, 15.02it/s]
     45%|████▍     | 105/235 [00:10<00:05, 22.12it/s]

    Training | Epoch: 5 | Loss: 194.4819 | Acc: 43.67%


     87%|████████▋ | 204/235 [00:14<00:01, 20.49it/s]

    Training | Epoch: 5 | Loss: 139.6240 | Acc: 45.91%


    100%|██████████| 235/235 [00:16<00:00, 14.60it/s]
     44%|████▍     | 103/235 [00:10<00:06, 20.21it/s]

    Training | Epoch: 6 | Loss: 186.2790 | Acc: 46.58%


     86%|████████▌ | 202/235 [00:14<00:01, 19.63it/s]

    Training | Epoch: 6 | Loss: 134.6118 | Acc: 48.28%


    100%|██████████| 235/235 [00:16<00:00, 14.54it/s]
     44%|████▍     | 103/235 [00:09<00:06, 20.21it/s]

    Training | Epoch: 7 | Loss: 180.0221 | Acc: 48.97%


     86%|████████▌ | 201/235 [00:14<00:01, 21.88it/s]

    Training | Epoch: 7 | Loss: 131.4528 | Acc: 49.70%


    100%|██████████| 235/235 [00:16<00:00, 14.62it/s]
     44%|████▍     | 103/235 [00:09<00:06, 21.15it/s]

    Training | Epoch: 8 | Loss: 173.1868 | Acc: 51.24%


     86%|████████▋ | 203/235 [00:14<00:01, 21.70it/s]

    Training | Epoch: 8 | Loss: 130.3644 | Acc: 49.56%


    100%|██████████| 235/235 [00:16<00:00, 14.67it/s]
     45%|████▍     | 105/235 [00:09<00:05, 22.82it/s]

    Training | Epoch: 9 | Loss: 171.3067 | Acc: 51.74%


     87%|████████▋ | 204/235 [00:14<00:01, 21.01it/s]

    Training | Epoch: 9 | Loss: 126.4864 | Acc: 51.59%


    100%|██████████| 235/235 [00:15<00:00, 15.27it/s]

