import torch

from bioplnn.models.connectome import ConnectomeODERNN, ConnectomeRNN


def test_connectome_rnn_initialization():
    # Test initialization with default parameters
    connectome = torch.eye(10)
    rnn = ConnectomeRNN(input_size=10, num_neurons=10, connectome=connectome)
    assert rnn is not None


def test_connectome_odernn_initialization():
    # Test initialization with default parameters
    connectome = torch.eye(10)
    odenn = ConnectomeODERNN(
        input_size=10, num_neurons=10, connectome=connectome
    )
    assert odenn is not None


def test_connectome_rnn_update_fn():
    # Test update function
    batch_size = 10
    input_size = 10
    num_neurons = 10
    connectome = torch.eye(num_neurons)

    rnn = ConnectomeRNN(
        input_size=input_size, num_neurons=num_neurons, connectome=connectome
    )

    x_t = torch.rand(input_size, batch_size)
    h = torch.rand(num_neurons, batch_size)

    h_new = rnn.update_fn(x_t, h)
    assert h_new.shape == h.shape


def test_connectome_odernn_update_fn():
    # Test update function
    batch_size = 16
    num_steps = 5
    input_size = 10
    num_neurons = 10
    connectome = torch.eye(num_neurons)

    rnn = ConnectomeODERNN(
        input_size=input_size, num_neurons=num_neurons, connectome=connectome
    )

    t = torch.zeros(batch_size)
    h = torch.rand(batch_size, num_neurons)
    x = torch.rand(num_steps, input_size, batch_size)
    args = {"x": x, "start_time": 0.0, "end_time": 1.0}

    dhdt = rnn.update_fn(t, h, args)
    assert dhdt.shape == h.shape


def test_connectome_rnn_forward():
    # Test forward pass
    batch_size = 16
    num_steps = 5
    input_size = 10
    num_neurons = 10
    connectome = torch.eye(num_neurons)

    rnn = ConnectomeRNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome,
        batch_first=False,
    )

    x = torch.rand(num_steps, batch_size, input_size)

    output = rnn.forward(x)
    assert output is not None


def test_connectome_odernn_forward():
    # Test forward pass
    batch_size = 16
    num_steps = 5
    input_size = 10
    num_neurons = 10
    connectome = torch.eye(num_neurons)

    odenn = ConnectomeODERNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome,
        batch_first=False,
    )
    x = torch.rand(num_steps, batch_size, input_size)
    output = odenn.forward(x)
    assert output is not None


# TODO: More extensive testing needs to be done for the connectome RNN and ODERNN
