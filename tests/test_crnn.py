import torch

from bioplnn.models.ei_crnn import Conv2dEIRNNLayer, Conv2dEIRNNLayerConfig


def test_layer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load config
    config = Conv2dEIRNNLayerConfig(
        spatial_size=(32, 32),
        in_channels=3,
        out_channels=10,
        neuron_channels=16,
        neuron_type="excitatory",
        neuron_spatial_mode="same",
        fb_channels=8,
        conv_connectivity=[[1, 0], [0, 1]],
        conv_rectify=False,
        conv_kernel_size=(3, 3),
        conv_activation="relu",
        conv_bias=True,
        post_agg_activation="relu",
        tau_mode="channel",
        tau_init_fn="ones",
        default_hidden_init_fn="zeros",
        default_fb_init_fn="zeros",
        default_out_init_fn="zeros",
    )
    model = Conv2dEIRNNLayer(config).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(16, 3, 32, 32).to(device)
    label = torch.randint(0, 10, (16,)).to(device)
    h_neuron = model.init_hidden(16, device=device)
    fb = model.init_fb(16, device=device)
    out, h_neuron_new = model(x, h_neuron, fb)
    assert out.shape == (16, 10, 32, 32)
    assert len(h_neuron_new) == 1
    assert h_neuron_new[0].shape == (16, 16, 32, 32)

    # Test forward pass with label
    pred = torch.nn.functional.adaptive_max_pool2d(out, (1, 1)).squeeze()
    loss = criterion(pred, label)
    assert loss.shape == ()
    assert loss.item() < 0.0

    # Test backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
