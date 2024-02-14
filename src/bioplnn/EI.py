import torch
import torch.nn as nn
from torch.autograd import Variable


class LinearExc(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight = torch.relu(self.weight)
        return super().forward(x)


class LinearInh(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight = -torch.relu(-self.weight)
        return super().forward(x)


class Conv2dExc(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight = torch.relu(self.weight)
        if self.bias is not None:
            self.bias = torch.relu(self.bias)
        return super().forward(x)


class Conv2dInh(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight = -torch.relu(-self.weight)
        if self.bias is not None:
            self.bias = -torch.relu(-self.bias)
        return super().forward(x)


class Conv2dEIRNNCell(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        cur_exc_dim: int,
        cur_inh_dim: int,
        exc_kernel_size: tuple[int, int],
        inhib_kernel_sizes: list[tuple[int, int]],
        use_h_prev: bool = False,
        prev_exc_dim: int = None,
        prev_inh_dim: int = None,
        use_feedback: bool = False,
        feedback_exc_dim: int = None,
        feedback_inh_dim: int = None,
        pool_kernel_size: tuple[int, int] = (5, 5),
        pool_stride: tuple[int, int] = (2, 2),
        bias: bool = True,
        euler: bool = True,
        dt: int = 1,
        activation: str = "tanh",
    ):
        """
        Initialize the ConvRNNEICell.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            prev_exc_dim (int): Number of channels of previous excitatory tensor.
            prev_inh_dim (int): Number of channels of previous inhibitory tensor.
            cur_exc_dim (int): Number of channels of current excitatory tensor.
            cur_inh_dim (int): Number of channels of current inhibitory tensor.
            feedback_exc_dim (int): Number of channels of feedback excitatory tensor.
            feedback_inh_dim (int): Number of channels of feedback inhibitory tensor.
            exc_kernel_size (tuple[int, int]): Size of the kernel for excitatory convolution.
            inhib_kernel_sizes (list[tuple[int, int]]): Sizes of the kernels for inhibitory convolutions.
            bias (bool, optional): Whether or not to add the bias. Default is True.
            euler (bool, optional): Whether to use Euler updates for the cell state. Default is True.
            dt (int, optional): Time step for Euler updates. Default is 1.
            activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported. Default is "tanh".
        """
        super().__init__()
        self.input_size = input_size

        self.input_dim = input_dim
        self.cur_exc_dim = cur_exc_dim
        self.cur_inh_dim = cur_inh_dim
        if use_h_prev and (prev_exc_dim is None or prev_inh_dim is None):
            raise ValueError(
                "If use_h_prev is True, prev_exc_dim and prev_inh_dim must be provided."
            )
        if use_feedback and (
            feedback_exc_dim is None or feedback_inh_dim is None
        ):
            raise ValueError(
                "If use_feedback is True, feedback_exc_dim and feedback_inh_dim must be provided."
            )
        self.use_h_prev = use_h_prev
        self.use_feedback = use_feedback
        padding = exc_kernel_size[0] // 2, exc_kernel_size[1] // 2
        self.bias = bias
        self.euler = euler
        self.dt = dt
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                "Only 'tanh' and 'relu' activations are supported."
            )

        # Learnable membrane time constants for excitatory and inhibitory cell populations
        self.tau_exc = nn.Parameter(
            torch.randn((1, cur_exc_dim, *input_size), requires_grad=True)
        )
        self.tau_inh = nn.Parameter(
            torch.randn((1, cur_inh_dim, *input_size), requires_grad=True)
            + 0.5
        )

        # Initialize excitatory convolutional layers
        exc_in_channels = (
            input_dim
            + cur_exc_dim
            + (prev_exc_dim if use_h_prev else 0)
            + (feedback_exc_dim if use_feedback else 0)
        )
        self.conv_exc = Conv2dExc(
            in_channels=exc_in_channels,
            out_channels=cur_exc_dim + cur_inh_dim,
            kernel_size=exc_kernel_size,
            padding=padding,
            bias=bias,
        )

        # Initialize inhibitory convolutional layers with different kernel sizes
        self.convs_inh = nn.ModuleList()
        inh_in_channels = (
            cur_inh_dim
            + (prev_inh_dim if use_h_prev else 0)
            + (feedback_inh_dim if use_feedback else 0)
        )
        for kernel_size in inhib_kernel_sizes:
            self.convs_inh.append(
                Conv2dInh(
                    in_channels=inh_in_channels,
                    out_channels=cur_exc_dim + cur_inh_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    bias=False,
                )
            )

        # Initialize output pooling layer
        self.out_pool = nn.AvgPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=(pool_kernel_size[0] // 2, pool_kernel_size[1] // 2),
        )

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.

        Returns:
            torch.Tensor: The initialized excitatory hidden state tensor.
            torch.Tensor: The initialized inhibitory hidden state tensor.
        """
        return (
            Variable(
                torch.zeros(batch_size, (self.cur_exc_dim), *self.input_size)
            ),
            Variable(
                torch.zeros(batch_size, (self.cur_inh_dim), *self.input_size)
            ),
        )

    def forward(
        self,
        input: torch.Tensor,
        h_cur_exc: torch.Tensor,
        h_cur_inh: torch.Tensor,
        h_prev_exc: torch.Tensor | None = None,
        h_prev_inh: torch.Tensor | None = None,
        feedback_exc: torch.Tensor | None = None,
        feedback_inh: torch.Tensor | None = None,
    ):
        """
        Performs forward pass of the cRNN_EI model.

        Args:
            input (torch.Tensor): Input tensor of shape (b, c, h, w).
                The input is actually the target_model.
            h_cur (torch.Tensor): Current hidden and cell states respectively
                of shape (b, c_hidden, h, w).

        Returns:
            torch.Tensor: Next hidden state of shape (b, c_hidden*2, h, w).
            torch.Tensor: Output tensor after pooling of shape (b, c_hidden*2, h', w').
        """
        exc_input = [input, h_cur_exc]
        if self.use_h_prev:
            if h_prev_exc is None:
                raise ValueError(
                    "If use_h_prev is True, h_prev_exc must be provided."
                )
            exc_input.append(h_prev_exc)
        if self.use_feedback:
            if feedback_exc is None:
                raise ValueError(
                    "If use_feedback is True, feedback_exc must be provided."
                )
            exc_input.append(feedback_exc)
        exc_input = torch.cat([*exc_input], dim=1)
        cnm = self.activation(self.conv_exc(exc_input, dim=1))

        inhibitions = []
        inh_input = []
        if self.use_h_prev:
            if h_prev_inh is None:
                raise ValueError(
                    "If use_h_prev is True, h_prev_inh must be provided."
                )
            inh_input.append(h_prev_inh)
        if self.use_feedback:
            if feedback_inh is None:
                raise ValueError(
                    "If use_feedback is True, feedback_inh must be provided."
                )
            inh_input.append(feedback_inh)
        inh_input = torch.cat(inh_input, dim=1)
        for conv_inh in self.convs_inh:
            inhibitions.append(self.activation(conv_inh(inh_input, dim=1)))

        # subtract contribution of inhibitory conv's from the cnm
        cnm_with_inh = cnm + sum(inhibitions)
        cnm_exc_with_inh, cnm_inh_with_inh = torch.split(
            cnm_with_inh, [self.cur_exc_dim, self.cur_inh_dim], dim=1
        )

        if self.euler:
            self.tau_exc = torch.sigmoid(self.tau_exc)
            h_next_exc = (
                1
                - self.tau_exc.unsqueeze(1)
                .unsqueeze(2)
                .repeat(1, *self.input_size)
            ) * h_cur_exc + (self.tau_exc) * cnm_exc_with_inh

            self.tau_inh = torch.sigmoid(self.tau_inh)
            h_next_inh = (
                1
                - self.tau_inh.unsqueeze(1)
                .unsqueeze(2)
                .repeat(1, self.height, self.width)
            ) * h_cur_inh + (self.tau_inh) * cnm_inh_with_inh
        else:
            raise NotImplementedError("Please use euler updates for now.")

        out = self.out_pool(torch.cat([h_next_exc, h_next_inh], dim=1))

        return h_next_exc, h_next_inh, out

    class Conv2dEIRNN(nn.Module):
        def __init__(
            self,
            input_size: tuple[int, int],
            input_dim: int,
            exc_dim: int | list[int],
            inh_dim: int | list[int],
            exc_kernel_size: tuple[int, int] | list[tuple[int, int]],
            inhib_kernel_sizes: (
                list[tuple[int, int]] | list[list[tuple[int, int]]]
            ),
            inhib_sclae_factors: list[int] | list[list[int]],
            num_layers: int,
            num_steps: int,
            num_classes: int,
            use_h_prev: bool = False,
            use_feedback: bool = False,
            feedback_adjacency: torch.Tensor | None = None,
            pool_kernel_size: tuple[int, int] = (5, 5),
            pool_stride: tuple[int, int] = (2, 2),
            bias: bool = True,
            euler: bool = True,
            dt: int = 1,
            activation: str = "tanh",
        ):
            """
            Initialize the Conv2dEIRNN.

            Args:
                input_size (tuple[int, int]): Height and width of input tensor as (height, width).
                input_dim (int): Number of channels of input tensor.
                hidden_dim (int): Number of channels of hidden tensor.
                num_layers (int): Number of layers in the RNN.
                num_iterations (int): Number of iterations to perform in each layer.
                exc_kernel_size (tuple[int, int]): Size of the kernel for excitatory convolution.
                inhib_kernel_sizes (list[tuple[int, int]]): Sizes of the kernels for inhibitory convolutions.
                use_h_prev (bool, optional): Whether to use previous hidden states as input. Default is False.
                use_feedback (bool, optional): Whether to use feedback from previous layers as input. Default is False.
                pool_kernel_size (tuple[int, int], optional): Size of the kernel for pooling. Default is (5, 5).
                pool_stride (tuple[int, int], optional): Stride of the pooling operation. Default is (2, 2).
                bias (bool, optional): Whether or not to add the bias. Default is True.
                euler (bool, optional): Whether to use Euler updates for the cell state. Default is True.
                dt (int, optional): Time step for Euler updates. Default is 1.
                activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported. Default is "tanh".
            """
            super().__init__()
            self.input_size = input_size
            self.input_dim = input_dim
            self.hidden_dim = self._extend_for_multilayer(hidden_dim)
            self.num_layers = num_layers
            self.num_steps = num_steps
            self.use_h_prev = use_h_prev
            self.use_feedback = use_feedback
            self.pool_kernel_size = pool_kernel_size
            self.pool_stride = pool_stride
            self.bias = bias
            self.euler = euler
            self.dt = dt
            self.activation = activation

            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(
                    Conv2dEIRNNCell(
                        input_size=input_size,
                        input_dim=input_dim,
                        cur_exc_dim=hidden_dim,
                        cur_inh_dim=hidden_dim,
                        exc_kernel_size=exc_kernel_size,
                        inhib_kernel_sizes=inhib_kernel_sizes,
                        use_h_prev=use_h_prev,
                        prev_exc_dim=hidden_dim,
                        prev_inh_dim=hidden_dim,
                        use_feedback=use_feedback,
                        feedback_exc_dim=hidden_dim,
                        feedback_inh_dim=hidden_dim,
                        pool_kernel_size=pool_kernel_size,
                        pool_stride=pool_stride,
                        bias=bias,
                        euler=euler,
                        dt=dt,
                        activation=activation,
                    )
                )

        @staticmethod
        def _check_kernel_size_consistency(kernel_size):
            if not (
                isinstance(kernel_size, tuple)
                or (
                    isinstance(kernel_size, list)
                    and all([isinstance(elem, tuple) for elem in kernel_size])
                )
            ):
                raise ValueError(
                    "`kernel_size` must be tuple or list of tuples"
                )

        @staticmethod
        def _extend_for_multilayer(param, num_layers):
            if not isinstance(param, list):
                param = [param] * num_layers
            return param

        def forward(self, input: torch.Tensor):
            """
            Performs forward pass of the Conv2dEIRNN.

            Args:
                input (torch.Tensor): Input tensor of shape (b, c, h, w).

            Returns:
                torch.Tensor: Output tensor after pooling of shape (b, hidden_dim*2, h', w').
            """
            batch_size = input.size(0)
            h_exc, h_inh = self._init_hidden(batch_size)
            prev_exc = [
                torch.zeros(batch_size, self.hidden_dim[i], *self.input_size)
            ] * self.num_layers
            prev_inh = [0] * self.num_layers
            feedback_exc = [0] * self.num_layers
            feedback_inh = [0] * self.num_layers

            for _ in range(self.num_iterations):
                for layer in self.layers:
                    h_exc, h_inh = layer(
                        input,
                        h_cur_exc=h_exc,
                        h_cur_inh=h_inh,
                        h_prev_exc=h_exc if self.use_h_prev else None,
                        h_prev_inh=h_inh if self.use_h_prev else None,
                        feedback_exc=h_exc if self.use_feedback else None,
                        feedback_inh=h_inh if self.use_feedback else None,
                    )
            return h_exc


class RecAttnModel(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        exc_column_dims,
        inh_class_dims,
        kernel_size,
        inhib_conv_kernel_sizes,
        inhib_scale_factors,
        num_layers,
        n_timesteps,
        dtype,
        num_classes,
        fc_size,
        batch_first=False,
        bias=True,
        return_all_layers=False,
        euler=False,
        dt=10,
    ):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param inhib_conv_kernel_sizes: List[(int, int)] or tuple((int, int))
            Size of inhibitory convolutional kernels.
        :param inhib_scale_factors: List[int]
            the larger the kernel size, the lower the loss coefficient/scale:
            || local inhibition || > || less local inhibition ||
            where local = smaller kernel size and less local = larger kernel size.
        :param num_layers: int
            Number of recurrent layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(RecAttnModel, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

        # self.height, self.width = input_size
        self.input_size = input_size
        self.input_dim = input_dim
        self.exc_column_dims = exc_column_dims
        self.inh_class_dims = inh_class_dims
        self.hidden_dim = [
            n_e + n_i for n_e, n_i in zip(exc_column_dims, inh_class_dims)
        ]
        self.kernel_size = kernel_size

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        self.kernel_size = self._extend_for_multilayer(
            self.kernel_size, num_layers
        )
        self.hidden_dim = self._extend_for_multilayer(
            self.hidden_dim, num_layers
        )
        if not len(self.kernel_size) == len(self.hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.inhib_conv_kernel_sizes = inhib_conv_kernel_sizes
        self.inhib_scale_factors = inhib_scale_factors
        assert len(self.inhib_scale_factors) == len(
            self.inhib_conv_kernel_sizes
        )

        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.n_timesteps = n_timesteps

        cell_list = []
        attn_blocks = []

        # TODO: readout only from the exc cells, not both exc and inh
        self.fc = nn.Linear(
            self.hidden_dim[-1]
            * self.input_size[-1]
            // 2
            * self.input_size[-1]
            // 2,
            fc_size,
        )
        self.classification = nn.Linear(fc_size, num_classes)

        self.readout_layer = nn.Sequential(
            self.fc,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            self.classification,
        )

        # input_size seems to be constant here? -- need to fix

        for i in range(0, self.num_layers):
            # cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cur_input_dim = self.input_dim[i]
            xh = xw = self.input_size[i]
            cell_list.append(
                ConvRNNEICell(
                    input_size=(xh, xw),
                    input_dim=cur_input_dim,
                    exc_column_dim=self.exc_column_dims[i],
                    inh_class_dim=self.inh_class_dims[i],
                    kernel_size=self.kernel_size[i],
                    inhib_conv_kernel_sizes=self.inhib_conv_kernel_sizes,
                    bias=self.bias,
                    dtype=self.dtype,
                    euler=euler,
                    dt=dt,
                )
            )
            attn_blocks.append(
                SimpleAttentionalGain(xh // 2, self.hidden_dim[i])
            )

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)
        self.attn_blocks = nn.ModuleList(attn_blocks)

    def forward(self, cue, mixture, hidden_state=None):
        """
        :param cue: (b, c, h, w)
        :param mixture: (b, c, h, w)
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """

        # Implement stateful ConvGRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=cue.size(0))

        ######
        #  process the cue (prime the cRNN)
        ######
        cue_activities = []
        seq_len = self.n_timesteps
        cur_layer_input = cue

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute
                # the next hidden and cell state through ConvGRUCell forward function
                if layer_idx == 0:
                    h, out = self.cell_list[layer_idx](
                        input_tensor=cur_layer_input, h_cur=h
                    )
                else:
                    h, out = self.cell_list[layer_idx](
                        input_tensor=cur_layer_input[:, t, ...], h_cur=h
                    )
                # output_inner.append(h)
                output_inner.append(out)

            cur_layer_input = torch.stack(output_inner, dim=1)
            # cue_activities.append([h])
            cue_activities.append([out])

        ######
        #  process the mixture (get cRNN to do the task)
        ######
        cur_layer_input = mixture
        layer_output_list, last_state_list = [], []

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                if layer_idx == 0:
                    h, out = self.cell_list[layer_idx](
                        input_tensor=cur_layer_input, h_cur=h
                    )
                else:
                    h, out = self.cell_list[layer_idx](
                        input_tensor=cur_layer_input[:, t, ...], h_cur=h
                    )

                ###################
                # Attention block #
                ###################
                # h = self.attn_blocks[layer_idx](cue_activities[layer_idx], h)
                out = self.attn_blocks[layer_idx](
                    cue_activities[layer_idx][0], out
                )

                output_inner.append(out)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def loss_function(self, cue, mixture, label):
        # mse = nn.MSELoss()

        out, hidden = self.forward(cue, mixture=mixture)

        # pick last layer, last timepoint
        out = out[-1][:, -1, ...]
        out = out.view(out.size(0), -1)
        out = self.readout_layer(out)

        inhib_loss = 0
        for cell in self.cell_list:
            for scale, conv in zip(self.inhib_scale_factors, cell.inhib_convs):
                inhib_loss += (
                    scale
                    * torch.linalg.norm(
                        conv.weight, dim=(-2, -1), ord=2
                    ).mean()
                )

        # calculate cross entropy loss
        loss = self.criterion(out, label)

        # total loss includes inhibitory loss
        loss += inhib_loss

        # log both losses for tracking purposes
        return {
            "loss": loss,
            "inhib_loss": inhib_loss,
        }
