## Setting up your environment

### Using conda

```bash
conda create -n bioplnn python=3.12
conda activate bioplnn
```
### Using venv

```bash
python -m venv venv
source venv/bin/activate
```

## Installation

### Using pip

The recommended installation method is via pip (not yet available):

```bash
pip install bioplnn
```

### Building from source

1. Clone the BioPlNN repository:

```bash
git clone https://github.com/valmikikothare/bioplnn.git
```

2. Navigate to the cloned directory:

```bash
cd bioplnn
```

3. Install specific dependencies:

```bash
pip install -r requirements/[cu124.txt|cpu.txt]
pip install -r requirements/[sparse_cu124.txt|sparse_cpu.txt]
```
where `cu124*` is for systems with CUDA 12.4-compatible GPUs and `cpu*` is for systems without CUDA.
These need to be installed in this order because of build dependencies and version conflicts.

4. [Optional] Install the development dependencies:

```bash
pip install -r requirements/dev.txt
```

5. Build and install the package:

```bash
pip install -e .
```
where `-e` installs the package in editable mode.


## Usage

### Using the CLI

Provided in the `examples` directory is a script for training the models.
The model, data, and training parameters are configured using Hydra configs,
which are stored in the `config` directory. See Hydra's
[docs](https://hydra.cc/docs/intro) for more information on the directory
structure and syntax.
Suppose we want to use the `e1l.yaml` model config in `config/model` and
the `mnist.yaml` data config in `config/data`. To specify these from the
command line, run
```bash
python examples/trainer.py model=e1l data=mnist
```
This relies on the `config/config.yaml` file, which contains
the following:
```yaml
defaults:
  - model: null
  - data: null
  ...
```
This means that the `model` and `data` keys must be overridden in the command
line, as shown above. If you want to set these to the default values, you can
edit the `config/config.yaml` file as follows:
```yaml
defaults:
  - model: e1l
  - data: mnist
  ...
```

### Using the API

#### ConnectomeRNN

```python
import torch
from bioplnn.models import ConnectomeRNN

connectivity_hh = torch.load("path/to/connectivity_hh.pt")
connectivity_ih = torch.load("path/to/connectivity_ih.pt")
output_neurons = torch.load("path/to/output_neurons.pt")
input_size = connectivity_hh.shape[1]
hidden_size = connectivity_hh.shape[2]

# Define the model
model = ConnectomeRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    connectivity_hh=connectivity_hh,
    connectivity_ih=connectivity_ih,
    output_neurons=output_neurons,
    nonlinearity="Sigmoid",
    batch_first=False,
    compile_solver_kwargs={
        "mode": "max-autotune",
        "dynamic": False,
        "fullgraph": True,
    },
)

# Define the input
num_steps = 10
batch_size = 8
inputs = torch.randn(num_steps, batch_size, input_size)

# Set the model to evaluation mode
model.eval()

# Perform a forward pass
outputs = model(inputs)

print(outputs.shape)
# (num_steps, batch_size, hidden_size)
```

#### SpatiallyEmbeddedRNN

```python
import torch
from bioplnn.models import SpatiallyEmbeddedRNN, SpatiallyEmbeddedAreaConfig

# Define the model
area_configs = [
    SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32),
        in_channels=3,
        out_channels=16,
    )
]
model = SpatiallyEmbeddedRNN(num_areas=1, area_configs=area_configs, batch_first=False)

# Define the input (num_steps, batch size, channels, height, width)
num_steps = 10
batch_size = 8
x = torch.randn(num_steps, batch_size, 3, 32, 32)

# Set the model to evaluation mode
model.eval()

# Perform a forward pass
outputs = model(x, num_steps=num_steps)

print(outputs.shape)
# (num_steps, batch_size, 16, 32, 32)
```