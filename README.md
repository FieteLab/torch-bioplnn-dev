# BioPlNN: Biologically Plausible Neural Network Package

**BioPlNN** is a PyTorch package designed to bridge the gap between traditional Artificial Neural Networks (ANNs) and biologically-inspired models. It provides modules that allow researchers to:

* Simulate large-scale populations of neurons with realistic biological properties.
* Explore the impact of network topology on neural function.
* Train models using standard machine learning techniques while incorporating biological constraints.

## Key Features

* **TopographicalRNN:** This module simulates a population of rate-based neurons with arbitrary connectivity patterns. It utilizes sparse tensors for efficient memory usage, enabling simulations of large-scale networks.
* **Conv2dEIRNN:** This module builds upon PyTorch's existing Conv2d and RNN modules by introducing separate excitatory and inhibitory neural populations within each layer. It allows users to define the connectivity between these populations within and across layers.

## Installation

### Using pip

The recommended installation method is via pip (not yet available):

```bash
pip install bioplnn
```

### Building from source

1. Clone the BioPlNN repository:

```bash
git clone https://github.com/hpvok13/bioplnn.git
```

2. Navigate to the cloned directory:

```bash
cd bioplnn
```

3. Build and install the package:

```bash
pip install [-e] .
```
where `-e` is optional and will install the package in editable mode.

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
There means that the `model` and `data` keys must be overridden in the command
line, as shown above. If you want to set these to the default values, you can
edit the `config/config.yaml` file as follows:
```yaml
defaults:
  - model: e1l
  - data: mnist
  ...
```

### Using the API

#### TopographicalRNN

```python
import torch
from bioplnn.models import TopographicalRNN

config = # Get config dictionary using yaml, hydra, etc.

# Create RNN layer
rnn = TopographicalRNN(**config)

# Define input data
inputs = torch.rand(batch_size, num_neurons)

# Run forward pass
outputs = rnn(inputs)
```

#### Conv2dEIRNN

```python
import torch
from bioplnn.models import Conv2dEIRNN

config = # Get config dictionary using yaml, hydra, etc.

# Create Conv2dEIRNN layer
rnn = Conv2dEIRNN(**config)

# Define input data
inputs = torch.rand(batch_size, in_channels, height, width)

# Run forward pass
outputs = rnn(inputs)
```

**Further Documentation:**

This README provides a basic introduction to BioPlNN. More detailed documentation, including advanced usage examples and configuration options, will be available soon. Example are provided in the `examples` directory.

**Contributing:**

We welcome contributions to BioPlNN! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on submitting code and documentation changes.
