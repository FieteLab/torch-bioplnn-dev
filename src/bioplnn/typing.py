from collections.abc import Callable, Sequence
from typing import Any, TypeVar, Union

import torch
from numpy.typing import NDArray

TensorInitFnType = Callable[..., torch.Tensor]
"""Type alias for a function that initializes a tensor.

A function that takes a variable number of positional arguments describing the
shape of the tensor to initialize. Can optionally take a `device` keyword 
argument, in which case the function is expected to return a tensor on that 
device.

Returns:
    torch.Tensor: The initialized tensor.
"""
ActivationFnType = Callable[[torch.Tensor], torch.Tensor]
"""Type alias for a function that applies an activation to a tensor.

A function that takes a single torch.Tensor as input and returns a transformed
torch.Tensor of the same shape. Used for neural network activation functions.

Args:
    tensor: Input tensor to be transformed.

Returns:
    torch.Tensor: The transformed tensor.
"""

T = TypeVar("T")

Param1dType = Union[T, Sequence[T], NDArray[Any]]
"""Type alias for a 1D parameter.

Used to annotate function arguments that can be a single value, a Sequence, or a 
NumPy array. Generic over the element type T.

Type:
    Union[T, Sequence[T], NDArray[T]]: A single value, Sequence, or NumPy array.
"""

CellTypeParam = Param1dType[T]
"""Type alias for a cell type parameter.

Used to annotate function arguments that can be a single value, a Sequence, or a 
NumPy array. Generic over the element type T.
"""

NeuronTypeParam = Param1dType[T]
"""Type alias for a neuron type parameter.

Used to annotate function arguments that can be a single value, a Sequence, or a 
NumPy array. Generic over the element type T.
"""

Param2dType = Union[T, Sequence[Sequence[T]], NDArray[Any]]
"""Type alias for a 2D parameter.

Used to annotate function arguments that can be a single value, a nested Sequence,
or a NumPy array. Generic over the element type T.

Type:
    Union[T, Sequence[Sequence[T]], NDArray[T]]: A single value, nested Sequence, or 
    NumPy array.
"""

InterCellTypeParam = Param2dType[T]
"""Type alias for an inter-cell type parameter.

Used to annotate function arguments that can be a single value, a Sequence, or a 
NumPy array. Generic over the element type T.
"""

InterAreaParam = Param2dType[T]
"""Type alias for an inter-area parameter.

Used to annotate function arguments that can be a single value, a Sequence, or a 
NumPy array. Generic over the element type T.
"""
