import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvRNNEICell(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        exc_column_dim,
        inh_class_dim,
        kernel_size,
        inhib_conv_kernel_sizes,
        bias,
        dtype,
        euler=False,
        dt=1,
    ):
        """
        Initialize the ConvGRU cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvRNNEICell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.exc_column_dim = exc_column_dim
        self.inh_class_dim = inh_class_dim
        self.bias = bias
        self.dtype = dtype
        self.euler = euler
        self.dt = dt

        # Learnable membrane time constants for excitatory and inhibitory cell populations
        # self.tau_exc = nn.Parameter(torch.randn((self.exc_column_dim, self.height, self.width), requires_grad=True)).type(self.dtype).unsqueeze(0)
        # self.tau_inh = nn.Parameter(torch.randn((self.inh_class_dim, self.height, self.width), requires_grad=True) + 0.5).type(self.dtype).unsqueeze(0)
        self.tau_exc = (
            nn.Parameter(torch.randn(self.exc_column_dim, requires_grad=True))
            .type(self.dtype)
            .unsqueeze(0)  # type: ignore
        )
        self.tau_inh = (
            nn.Parameter(
                torch.randn(self.inh_class_dim, requires_grad=True) + 0.5
            )
            .type(self.dtype)
            .unsqueeze(0)  # type: ignore
        )

        self.conv_cnm_exc = nn.Conv2d(
            in_channels=input_dim + self.exc_column_dim,
            out_channels=self.exc_column_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_cnm_inh = nn.Conv2d(
            in_channels=input_dim + self.exc_column_dim + self.inh_class_dim,
            out_channels=self.inh_class_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        # inhibitory convs
        assert isinstance(inhib_conv_kernel_sizes, list) or isinstance(
            inhib_conv_kernel_sizes, tuple
        )

        self.inhib_conv_kernel_sizes = inhib_conv_kernel_sizes
        self.inhib_convs = nn.ModuleList()

        for kernel_size in self.inhib_conv_kernel_sizes:
            self.inhib_convs.append(
                nn.Conv2d(
                    in_channels=self.inh_class_dim,
                    out_channels=self.exc_column_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    bias=False,
                )
            )

        self.out_pool = nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2))

    def init_hidden(self, batch_size):
        return Variable(
            torch.zeros(
                batch_size,
                (self.exc_column_dim + self.inh_class_dim),
                self.height,
                self.width,
            )
        ).type(self.dtype)

    def non_positive_rectifier(self, conv):
        # same as min(0, conv.weight.data)
        conv.weight.data = -torch.relu(-conv.weight.data)

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """

        exc_cells, inh_cells = torch.split(h_cur, self.exc_column_dim, dim=1)

        cnm_exc = torch.tanh(
            self.conv_cnm_exc(torch.cat([input_tensor, exc_cells], dim=1))
        )

        cnm_inh = torch.tanh(
            self.conv_cnm_inh(
                torch.cat([input_tensor, exc_cells, inh_cells], dim=1)
            )
        )

        # candidate neural memories after inhibition of varying distance
        total_inhs = []
        for conv in self.inhib_convs:
            # non-positive rectification of each conv weight
            self.non_positive_rectifier(conv)

            total_inhs.append(conv(cnm_inh))
        # subtract contribution of inhibitory conv's from the cnm
        cnm_exc_with_inh = cnm_exc - sum(total_inhs)

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
            raise NotImplementedError("please use euler updates for now.")

        out = self.out_pool(h_next)

        return h_next, out
