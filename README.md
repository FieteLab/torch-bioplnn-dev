## `torch-biopl`: Biologically-plausible neural networks made easy
<p align="center" style="text-align: center">
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.7-3776AB.svg?style=flat&amp;logo=python&amp;logoColor=white" alt="python"></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-1.8.0-EE4C2C.svg?style=flat&amp;logo=pytorch" alt="pytorch"></a>
<a href="https://serre-lab.github.io/rnn_rts_site/"><img alt="RNN RTS" src="https://img.shields.io/badge/Project%20page-RNN%20RTs-green"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

`torch-biopl` is a PyTorch package designed to bridge the gap between traditional Artificial Neural Networks (ANNs) and biologically-inspired models. It provides modules that allow researchers to:

* Train models using standard machine learning techniques while incorporating biological constraints.
* Simulate large-scale populations of neurons with realistic biological properties.
* Explore the impact of network topology on neural function.

### Core functionalities

* `ConnectomeRNN`: Handles rate-based neural populations whose recurrent connectivity matrix is specified from a biologically recoded (or synthetically initialized) connectome. It includes under-the-hood application of sparse tensor arithmetic for efficient memory usage, enabling simulation and training of large-scale networks. We support the ability to flexibly spin up probabilistic connectomes, define celltypes and associated synaptic variables, and tune user-defined parameters via gradient descent.

* `SpatiallyEmbeddedRNN`: A library of functions that support

This module simulates a series of spatially embedded areas of the brain, each with their configurable neuron types, circuit motifs, and transfer functions.

### Installation and usage
For instructions to install the right dependencies and use either the API or CLI (Command Line Interface) of `torch-biopl` please refer to our [quick start guide](https://fietelab.github.io/torch-biopl-dev/install) and [API documentation](#).

### Tutorials and examples

This README provides a basic introduction to BioPlNN. More detailed documentation, including advanced usage examples and configuration options, will be available soon. Example are provided in the `examples` directory.

### Contributing

We welcome contributions to `torch-biopl`! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on submitting code and documentation changes.
