from typing import Union
import torch
import torch.nn as nn
from torch.autograd import Variable


class LinearExc(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def positive_rectifier(self, weight):
        """
        Applies the positive rectifier activation function to the convolutional layer weights.

        Args:
            conv (torch.nn.Conv2d): The convolutional layer.
        Returns:
            torch.Tensor: The rectified weights.
        """
        return torch.relu(weight)

    def forward(self, x):
        self.weight.data = self.positive_rectifier(self.weight.data)
        return super().forward(x)


class LinearInh(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def non_positive_rectifier(self, weight):
        """
        Applies the positive rectifier activation function to the convolutional layer weights in-place.

        Args:
            conv (torch.nn.Conv2d): The convolutional layer.
        Returns:
            torch.Tensor: The rectified weights.
        """
        return -torch.relu(-weight)

    def forward(self, x):
        self.weight.data = self.non_positive_rectifier(self.weight.data)
        return super().forward(x)


class Conv2dExc(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def positive_rectifier(self, weight):
        """
        Applies the positive rectifier activation function to the convolutional layer weights in-place.

        Args:
            conv (torch.nn.Conv2d): The convolutional layer.
        Returns:
            torch.Tensor: The rectified weights.
        """
        return torch.relu(weight)

    def forward(self, x):
        self.weight.data = self.positive_rectifier(self.weight.data)
        return super().forward(x)


class Conv2dInh(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def non_positive_rectifier(self, weight):
        """
        Applies the positive rectifier activation function to the convolutional layer weights in-place.

        Args:
            conv (torch.nn.Conv2d): The convolutional layer.
        Returns:
            torch.Tensor: The rectified weights.
        """
        return -torch.relu(-weight)

    def forward(self, x):
        self.weight.data = self.non_positive_rectifier(self.weight.data)
        return super().forward(x)


class ConvRNNEICell(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        exc_dim: int,
        inh_dim: int,
        kernel_size: tuple[int, int],
        inhib_conv_kernel_sizes: Union[list[int], tuple[int]],
        bias: bool = True,
        euler: bool = False,
        dt: int = 1,
        activation: str = "tanh",
    ):
        """
        Initialize the ConvGRU cell

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            exc_dim (int): Number of channels of excitatory column tensor.
            inh_dim (int): Number of channels of inhibitory class tensor.
            kernel_size (tuple[int, int]): Size of the convolutional kernel.
            inhib_conv_kernel_sizes (Union[list[int], tuple[int]]): Sizes of the convolutional kernels for inhibitory convolutions.
            bias (bool): Whether or not to add the bias.
            euler (bool, optional): Whether to use Euler updates for the cell state. Default is False.
            dt (int, optional): Time step for Euler updates. Default is 1.
        """
        super(ConvRNNEICell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.exc_dim = exc_dim
        self.inh_dim = inh_dim
        self.bias = bias
        self.euler = euler
        self.dt = dt
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(
                "Only 'tanh' and 'relu' activations are supported."
            )
        # Learnable membrane time constants for excitatory and inhibitory cell populations
        self.tau_exc = nn.Parameter(
            torch.randn(
                (self.exc_dim, self.height, self.width), requires_grad=True
            )
        ).unsqueeze(0)
        self.tau_inh = nn.Parameter(
            torch.randn(
                (self.inh_dim, self.height, self.width), requires_grad=True
            )
            + 0.5
        ).unsqueeze(0)
        # self.tau_exc = nn.Parameter(
        #     torch.randn(self.exc_dim, requires_grad=True)
        # ).unsqueeze(0)
        # self.tau_inh = nn.Parameter(
        #     torch.randn(self.inh_dim, requires_grad=True) + 0.5
        # ).unsqueeze(0)

        self.conv_exc = Conv2dExc(
            in_channels=input_dim + self.exc_dim,
            out_channels=self.exc_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_inh = Conv2dInh(
            in_channels=input_dim + self.exc_dim + self.inh_dim,
            out_channels=self.inh_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        # Inhibitory convs
        if not (
            isinstance(inhib_conv_kernel_sizes, list)
            or isinstance(inhib_conv_kernel_sizes, tuple)
        ):
            raise ValueError(
                "inhib_conv_kernel_sizes must be a list or tuple of integers."
            )

        self.inhib_conv_kernel_sizes = inhib_conv_kernel_sizes
        self.inhib_convs = nn.ModuleList()

        for kernel_size in self.inhib_conv_kernel_sizes:
            self.inhib_convs.append(
                Conv2dInh(
                    in_channels=self.inh_dim,
                    out_channels=self.exc_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    bias=False,
                )
            )

        self.out_pool = nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2))

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.

        Returns:
            torch.Tensor: The initialized hidden state tensor.
        """
        return Variable(
            torch.zeros(
                batch_size,
                (self.exc_dim + self.inh_dim),
                self.height,
                self.width,
            )
        )

    def forward(self, input_tensor: torch.Tensor, h_cur: torch.Tensor):
        """
        Performs forward pass of the cRNN_EI model.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (b, c, h, w).
                The input is actually the target_model.
            h_cur (torch.Tensor): Current hidden and cell states respectively
                of shape (b, c_hidden, h, w).

        Returns:
            torch.Tensor: Next hidden state of shape (b, c_hidden*2, h, w).
            torch.Tensor: Output tensor after pooling of shape (b, c_hidden*2, h', w').
        """

        exc_cells, inh_cells = torch.split(
            h_cur, [self.exc_dim, self.inh_dim], dim=1
        )

        cnm_exc = torch.tanh(
            self.conv_exc(torch.cat([input_tensor, exc_cells], dim=1))
        )

        cnm_inh = torch.tanh(
            self.conv_inh(
                torch.cat([input_tensor, exc_cells, inh_cells], dim=1)
            )
        )

        # candidate neural memories after inhibition of varying distance
        total_inhs = torch.zeros_like(cnm_inh)
        for conv in self.inhib_convs:
            # non-positive rectification of each conv weight
            total_inhs += conv(cnm_inh)

        # subtract contribution of inhibitory conv's from the cnm
        cnm_exc_with_inh = cnm_exc - total_inhs

        if self.euler:
            self.tau_exc = torch.sigmoid(self.tau_exc)
            h_next_exc = (
                1
                - self.tau_exc.unsqueeze(1)
                .unsqueeze(2)
                .repeat(1, self.height, self.width)
            ) * exc_cells + (self.tau_exc) * cnm_exc_with_inh

            self.tau_inh = torch.sigmoid(self.tau_inh)
            h_next_inh = (
                1
                - self.tau_inh.unsqueeze(1)
                .unsqueeze(2)
                .repeat(1, self.height, self.width)
            ) * inh_cells + (self.tau_inh) * cnm_inh

            h_next = torch.cat([h_next_exc, h_next_inh], dim=1)
        else:
            raise NotImplementedError("Please use euler updates for now.")

        out = self.out_pool(h_next)

        return h_next, out
