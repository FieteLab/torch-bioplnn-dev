# Exploring advanced configurations

## Overview

In addition to the essential capabilities exposed in the first tutorial, we can exercise fine-grained control in how we wire up biologically-plausible networks. These are some of the functionality we will explore in this section:

- Synapse vs neuron nonlinearities
- Microcircuit archetypes
- Parameter sharing capabilities
- Hierarchically constructed neural areas
- Inter-areal feedback connectivity


```python
import torch
import numpy as np
from bioplnn.models import SpatiallyEmbeddedRNN, SpatiallyEmbeddedAreaConfig
```

## Synapse vs neuron nonlinearities

What do we mean by this? In an attempt to distinguish synaptic transfer functions from post-aggregation neuronal transfer functions, we give users the ability to specify pre- and post-integration nonlinearities. 

Let us consider the same example model from the previous tutorial: A simple one-area network with two neural classes with the following `inter_neuron_type_connectivity`: $\begin{bmatrix}1&1&0\cr1&1&1\cr1&1&0\end{bmatrix}$. Following the same convention as the connectivity matrix, you can specify the transfer function for each of those synapse groups by setting the `inter_neuron_type_nonlinearity` parameter. Similarly, `neuron_type_nonlinearity` can be used to control the post-aggregation transfer function for each neuron type.

If you were to construe a scenario where synapses have unbounded transfer functions while the neuron as whole is bounded from above (and for the sake of argument: bounded differently for the E and I subpopulations), then you'd do something like this:


```python
# writing a custom activation function that works similarly to a ReLU but is bounded from above!
from torch.nn.modules.activation import Hardtanh
class ModRelu(Hardtanh):
    def __init__(self, _ub: float, _lb: float = 0., inplace: bool = False):
        super().__init__(_lb, _ub, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
upper_bounded_relu = ModRelu(_ub = 5.)
```


```python
area_configs = [
    SpatiallyEmbeddedAreaConfig(
                num_neuron_types = 2,
                num_neuron_subtypes = np.array([16, 16]),
                neuron_type_class = np.array(['excitatory', 'inhibitory']),
                neuron_type_nonlinearity = ['sigmoid', upper_bounded_relu],
                inter_neuron_type_connectivity = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]]),
                inter_neuron_type_nonlinearity = np.array([['relu', 'relu', ''], ['relu', 'relu', 'relu'], ['relu', 'relu', '']]),
                in_size = [28, 28],
                in_channels =  1,
                out_channels = 32,
    )
]
model = SpatiallyEmbeddedRNN(num_areas=1, area_configs=area_configs, batch_first=False)
```

## Microcircuit archetypes

It must be evident that `inter_neuron_type_connectivity` is a powerful option that can be used to dictate a wide variety of microcircuit motifs. We provide some examples below for inspiration.

#### Feedback inhibition
Parvalbumin-positive inhibitory cells in Layer 2/3 of the cortex are known to interact with Pyramidal cells through some form of divisive inhibition [Jonke et al. (2017)](https://www.jneurosci.org/content/37/35/8511). To instantiate this microcircuit, you'd set `inter_neuron_type_connectivity` $= \begin{bmatrix}1&0&0\cr0&1&1\cr1&1&0\end{bmatrix}$ (conventions same as in the original example).

#### Feedforward inhibition
Feedforward inhibition is another essential mechanism within the brain, to regulate neuronal firing and prevent runaway excitation ([Panthi and Leitch (2019)](https://pubmed.ncbi.nlm.nih.gov/31494287/), [Large et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4776521/)). To implement the microcircuit presented in these (and related) papers, you can set `inter_neuron_type_connectivity` $= \begin{bmatrix}1&1&0\cr1&0&1\cr1&1&1\end{bmatrix}$

#### Pyr-PV-SST-VIP motif
Interneuron subtypes play a critical role in several aspects of cortical function ([Guo and Kumar (2023)](https://www.nature.com/articles/s42003-023-05231-0), [Condylis et al. (2022)](https://www.science.org/doi/10.1126/science.abl5981), etc.). Of particular interest is a motif that involves one excitatory and three inhibitory interneuron populations (PV, SST, VIP). Please refer to these papers for pictorial depictions of the microcircuits. To realise this in `torch-biopl`, we would do the following:


```python
area_configs = [
    SpatiallyEmbeddedAreaConfig(
                num_neuron_types = 4,
                num_neuron_subtypes = np.array([16, 8, 8, 8]),
                neuron_type_class = np.array(['excitatory', 'inhibitory', 'inhibitory', 'inhibitory']),
                inter_neuron_type_connectivity = np.array([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 0]]),
                in_size = [28, 28],
                in_channels =  1,
                out_channels = 32,
    )
]
model = SpatiallyEmbeddedRNN(num_areas=1, area_configs=area_configs, batch_first=False)
```

We remark that this is not an exhaustive list, but merely a window into endless possibilities :)

## Parameter sharing capabilities for $ \tau_{mem} $

We provide the user the option to tie neural time constants:
- Across space, but unique for each cell subtype (`tau_mode` = 'subtype')
- Across cell subtype, but unique for each spatial location (`tau_mode` = 'spatial')
- Each neuron learns its own time constant (`tau_mode` = 'subtype_spatial')
- Across ***types*** (`tau_mode` = 'type')

To go hand in hand with this, we also allow the user to provide an initialization for these time constants. This can be done via `tau_init_fn`. As with the nonlinearities, users can either provide torch initializers or custom functions to accomplish this.

## Hierarchically constructed neural areas

Intuitive and expressive. For reasons more than one, you may want to wire up brain areas that are connected to each other via long-range synapses. This is quite easy to accomplish in `torch-biopl`. Note that each area can be configured independently!


```python
area_configs = [
    SpatiallyEmbeddedAreaConfig(
                num_neuron_types = 2,
                num_neuron_subtypes = np.array([16, 16]),
                neuron_type_class = np.array(['excitatory', 'inhibitory']),
                inter_neuron_type_connectivity = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]]),
                in_size = [28, 28],
                in_channels =  1,
                out_channels = 32,
    ),
    SpatiallyEmbeddedAreaConfig(
                num_neuron_types = 2,
                num_neuron_subtypes = np.array([32, 32]),
                neuron_type_class = np.array(['excitatory', 'inhibitory']),
                inter_neuron_type_connectivity = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]]),
                in_size = [14, 14],
                in_channels =  32,
                out_channels = 32,
    )
]

model = SpatiallyEmbeddedRNN(num_areas=2, area_configs=area_configs, batch_first=False)
```

## Inter-areal feedback connectivity

Finally, when you have multiple interacting areas, you'd want to ability to feedback information from downstream areas back up to early areas. `torch-biopl` provides an easy way to configure the flow of information. In simple terms, users can provide an adjacency matrix where rows are presynaptic ***areas*** and columns are postsynaptic ***areas***.


```python
conn = SpatiallyEmbeddedAreaConfig.inter_neuron_type_connectivity_template_df(use_feedback=True, num_neuron_types=2)
# this prints out the format of the connectivity adjacency matrix that you can follow
print(conn)
```

              neuron_0  neuron_1  output
    input        False     False   False
    feedback     False     False   False
    neuron_0     False     False   False
    neuron_1     False     False   False



```python
area_configs_feedback_model = [
    SpatiallyEmbeddedAreaConfig(
                num_neuron_types = 2,
                num_neuron_subtypes = np.array([16, 16]),
                neuron_type_class = np.array(['excitatory', 'inhibitory']),
                inter_neuron_type_connectivity = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]]),
                feedback_channels = 16,
                in_size = [28, 28],
                in_channels =  1,
                out_channels = 32,
    ),
    SpatiallyEmbeddedAreaConfig(
                num_neuron_types = 2,
                num_neuron_subtypes = np.array([32, 32]),
                neuron_type_class = np.array(['excitatory', 'inhibitory']),
                inter_neuron_type_connectivity = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]]),
                in_size = [14, 14],
                in_channels =  32,
                out_channels = 32,
    )
]

model_wFeedback = SpatiallyEmbeddedRNN(
                            num_areas = 2, 
                            area_configs = area_configs_feedback_model, 
                            batch_first = False,
                            inter_area_feedback_connectivity = np.array([[0, 0],[1, 0]])
                    )
```
