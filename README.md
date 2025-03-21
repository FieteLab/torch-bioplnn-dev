## `torch-biopl`: Biologically-plausible neural networks made easy
<p align="center" style="text-align: center">
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&amp;logo=python&amp;logoColor=white" alt="python"></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&amp;logo=pytorch" alt="pytorch"></a>
<a href="https://fietelab.github.io/torch-biopl-dev/"><img alt="torch-biopl" src="https://img.shields.io/badge/Project%20page-torch--biopl-black"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

`torch-biopl` is a PyTorch package designed to bridge the gap between traditional Artificial Neural Networks (ANNs) and biologically-inspired models. It provides modules that allow researchers to:

* Train models using standard machine learning techniques while incorporating biological constraints.
* Simulate large-scale populations of neurons with realistic biological properties.
* Explore the impact of network topology on neural function.

### Core functionalities

* `ConnectomeRNN`
    - Handles rate-based neural populations whose recurrent connectivity matrix is specified from a biologically recoded (or synthetically initialized) connectome.
    - Includes under-the-hood application of sparse tensor arithmetic for efficient memory usage, enabling simulation and training of large-scale networks.
    - Supports the ability to flexibly spin up probabilistic connectomes, define celltypes and associated synaptic variables, and tune user-defined parameters via gradient descent.

* `SpatiallyEmbeddedRNN`: A library of functions that support
    - Model constructors and helpers to wire up cortical architectures with varying levels of biological specification.
    - Configurable aspects include cell classes, cell types, cell subtypes, local connectivity rules, synaptic and neuronal nonlinearities, time constants, feedback wiring, and lots more.

### Installation and usage
For instructions to install the right dependencies and use either the API or CLI (Command Line Interface) of `torch-biopl` please refer to our [Quick start guide](https://fietelab.github.io/torch-biopl-dev/install) and [API documentation](https://fietelab.github.io/torch-biopl-dev/reference).

### Tutorials and examples

For [basic](https://fietelab.github.io/torch-biopl-dev/examples/basic_spatial_example) and [advanced](https://fietelab.github.io/torch-biopl-dev/examples/advance_configs) examples please refer to the webpage.

### Contributing

We welcome contributions to `torch-biopl`. For guidelines on submitting code and documentation changes, please refer to [contributing](#).
