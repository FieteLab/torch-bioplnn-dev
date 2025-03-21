# BioPlNN: Biologically Plausible Neural Network Package

**BioPlNN** is a PyTorch package designed to bridge the gap between traditional Artificial Neural Networks (ANNs) and biologically-inspired models. It provides modules that allow researchers to:

* Simulate large-scale populations of neurons with realistic biological properties.
* Explore the impact of network topology on neural function.
* Train models using standard machine learning techniques while incorporating biological constraints.

## Key Features

* **ConnectomeRNN:** This module simulates a population of rate-based neurons with arbitrary connectivity patterns. It utilizes sparse tensors for efficient memory usage, enabling simulations of large-scale networks.
* **SpatiallyEmbeddedRNN:** This module simulates a series of spatially embedded areas of the brain, each with their configurable neuron types, circuit motifs, and transfer functions.

## Installing Python
This project requires Python 3.9 or higher. Most modern Linux distributions and
macOS come with Python 3.9 or higher pre-installed. If you are on a system that
does not have Python 3.9 or higher installed, there are a couple of options to
install it.

We recommend using [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main),
an environment and package manager that allows you to create and manage multiple
isolated environments, each with their own Python version and set of installed packages.
You can use miniconda even if you already have a Python installation.

To install miniconda, follow the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

Alternatively, you can install Python 3.9 or higher directly, following the instructions
[here](https://www.python.org/downloads/).

## Setting up your environment
Once you have Python 3.9 or higher installed, you can create a new environment
in which to install the required packages. If you have miniconda installed,
follow [Using conda](#using-conda) below. Otherwise, follow
[Using venv](#using-venv) below (or use any other environment manager you prefer).

### Using conda

If you have not already done so, initialize conda in your terminal:
```bash
conda init
```

Then, create a new environment and activate it:
```bash
conda create -n bioplnn python=3.12
conda activate bioplnn
```

To verify that you are in the `bioplnn` environment, run:
```bash
conda env list
```

To deactivate the environment, run:
```bash
conda deactivate
```

For more information on conda, see the [conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html).

### Using venv

```bash
python -m venv venv
source venv/bin/activate
```

## Requirements
This project depends on certain packages that are not available on PyPI. You must install these manually
before installing BioPlNN.

### PyTorch and Torchvision
Currently, the latest supported version of PyTorch is 2.5.1. To install a
specific version of PyTorch (and its corresponding Torchvision version), follow
the instructions for your system [here](https://pytorch.org/get-started/previous-versions/).

For example, below are the installation commands for PyTorch 2.5.1 and
Torchvision 0.20.1 on the following systems:
1. System with a CUDA 12.4-compatible GPU
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```
2. CPU-only system (not macOS)
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```
3. CPU-only system (macOS)
```bash
pip install torch==2.5.1 torchvision==0.20.1
```

### PyTorch Sparse
You must install PyTorch ([see above](#pytorch-and-torchvision)) before installing PyTorch Sparse.

To install PyTorch Sparse for your specific system and PyTorch version, follow the instructions [here](https://github.com/rusty1s/pytorch_sparse).

For example, to install PyTorch Sparse for PyTorch 2.5.1 and a system with a
CUDA 12.4-compatible GPU, you would run:
```bash
pip install torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```
And for a CPU-only system (any OS), you would run:
```bash
pip install torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

## Installation
Make sure you have installed the requirements as described [above](#requirements).
Then, you can install the package using one of the following methods:

### From PyPI (recommended) (not yet available)

```bash
pip install bioplnn
```

### From source

1. Clone the BioPlNN repository:

```bash
git clone https://github.com/valmikikothare/bioplnn.git
```

2. Navigate to the cloned directory:

```bash
cd bioplnn
```

3. Install the package:

```bash
pip install -e .
```
where `-e` installs the package in editable mode.


## Usage

### Using the CLI

Provided in the `examples` directory is `trainer.py`, a sample script for
training the models on classification tasks.

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

connectome = torch.load("path/to/connectome.pt")
input_projection = torch.load("path/to/input_projection.pt")
output_projection = torch.load("path/to/output_projection.pt")
input_size = input_projection.shape[1]
hidden_size = input_projection.shape[2]
output_size = output_projection.shape[1]

# Define the model
model = ConnectomeODERNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    connectome=connectome,
    input_projection=input_projection,
    output_projection=output_projection,
    neuron_nonlinearity="Sigmoid",
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

**Further Documentation:**

This README provides a basic introduction to BioPlNN. More detailed documentation, including advanced usage examples and configuration options, will be available soon. Example are provided in the `examples` directory.

**Contributing:**

We welcome contributions to BioPlNN! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on submitting code and documentation changes.
