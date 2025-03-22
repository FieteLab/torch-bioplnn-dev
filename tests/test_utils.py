import numpy as np
import torch

from bioplnn.utils.common import is_list_like, without_keys
from bioplnn.utils.torch import manual_seed, manual_seed_deterministic


def test_manual_seed():
    # Test setting manual seed
    manual_seed(42)
    assert torch.initial_seed() == 42


def test_manual_seed_deterministic():
    # Test setting manual seed for deterministic execution
    manual_seed_deterministic(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_without_keys():
    # Test removing keys from a dictionary
    d = {"a": 1, "b": 2, "c": 3}
    result = without_keys(d, ["b"])
    assert result == {"a": 1, "c": 3}


def test_is_list_like():
    # Test checking if an object is list-like
    assert is_list_like([1, 2, 3]) is True
    assert is_list_like("string") is False


def test_is_list_like_edge_cases():
    # Test edge cases for is_list_like
    assert is_list_like((1, 2, 3)) is True  # Tuple
    assert is_list_like({"key": "value"}) is False  # Dict
    assert is_list_like(np.array([1, 2, 3])) is True  # Numpy array


def test_without_keys_edge_cases():
    # Test edge cases for without_keys
    d = {"a": 1, "b": 2, "c": 3}
    result = without_keys(d, ["d"])  # Key not in dict
    assert result == {"a": 1, "b": 2, "c": 3}

    result = without_keys(d, [])  # No keys to remove
    assert result == d

    result = without_keys({}, ["a"])  # Empty dict
    assert result == {}


# TODO: More extensive testing needs to be done for the utils
