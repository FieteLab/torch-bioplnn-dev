## `torch-biopl`: Biologically-plausible neural networks made easy

`torch-biopl` is a PyTorch package designed to bridge the gap between traditional Artificial Neural Networks (ANNs) and biologically-inspired models. It provides modules that allow researchers to:

* Train models using standard machine learning techniques while incorporating biological constraints.
* Simulate large-scale populations of neurons with realistic biological properties.
* Explore the impact of network topology on neural function.

### Core functionalities

* `ConnectomeRNN`: This module simulates a population of rate-based neurons whose recurrent connectivity matrix is specified from a biologically recoded (or synthetically initialized) connectome. It includes under-the-hood application of sparse tensor arithmetic for efficient memory usage, enabling simulation and training of large-scale networks. We support the ability to define celltypes which specify details about synapse.

* `SpatiallyEmbeddedRNN`: This module simulates a series of spatially embedded areas of the brain, each with their configurable neuron types, circuit motifs, and transfer functions.

### Installation and usage
For instructions to install the right dependencies and use either the API or CLI (Command Line Interface) of `torch-biopl` please refer to our [quick start guide](https://fietelab.github.io/torch-bioplnn-dev/install) and [API documentation](#).

### Tutorials and examples

This README provides a basic introduction to BioPlNN. More detailed documentation, including advanced usage examples and configuration options, will be available soon. Example are provided in the `examples` directory.

### Contributing

We welcome contributions to `torch-biopl`! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on submitting code and documentation changes.
