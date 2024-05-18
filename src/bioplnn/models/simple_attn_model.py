import numpy as np
import torch
import torch.nn as nn
from layers import conv2d_same
from layers.hannpooling2d import HannPooling2d
from torch.autograd import Variable
from torchvision import models as M


class SimpleAttentionalGain(nn.Module):
    def __init__(self, spatial_dim, cnn_channels):
        super(SimpleAttentionalGain, self).__init__()

        # outsize is N X C X SD X SD
        self.spatial_average = nn.AdaptiveAvgPool2d((spatial_dim, spatial_dim))

        self.bias = nn.Parameter(torch.zeros(1))  # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1))  # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1))  # init threshold to zero
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.slope, 1)
        nn.init.constant_(self.threshold, 0)

    def forward(self, cue, mixture):
        ## Process cue
        cue = self.spatial_average(cue)

        # apply threshold shift
        cue = cue - self.threshold

        # apply slope
        cue = cue * self.slope

        # apply sigmoid & bias
        cue = self.bias + (1 - self.bias) * torch.sigmoid(cue)

        # Apply to mixture (element mult)
        mixture = torch.mul(mixture, cue)

        return mixture


class attnCNN(nn.Module):
    def __init__(self, num_classes=1000, fc_size=2048):
        super(attnCNN, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

        # hard-coding the input representational space here...
        self.norm_coch_rep = nn.LayerNorm([3, 128, 128])
        self.attn_block_in = SimpleAttentionalGain(128, 3)

        self.conv0 = nn.Sequential(
            nn.LayerNorm([3, 128, 128]),
            conv2d_same.create_conv2d_pad(
                3, 8, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [5, 5], padding = [2, 2])
            nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2)),
        )
        self.attn_block0 = SimpleAttentionalGain(64, 8)

        self.conv1 = nn.Sequential(
            nn.LayerNorm([8, 64, 64]),
            conv2d_same.create_conv2d_pad(
                8, 16, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [5, 5], padding = [2, 2])
            nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2)),
        )
        self.attn_block1 = SimpleAttentionalGain(32, 16)

        self.conv2 = nn.Sequential(
            nn.LayerNorm([16, 32, 32]),
            conv2d_same.create_conv2d_pad(
                16, 32, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [3, 3], padding = [1, 1])
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.attn_block2 = SimpleAttentionalGain(16, 32)

        self.conv3 = nn.Sequential(
            nn.LayerNorm([32, 16, 16]),
            conv2d_same.create_conv2d_pad(
                32, 64, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [3, 3], padding = [1, 1])
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.attn_block3 = SimpleAttentionalGain(8, 64)

        self.conv4 = nn.Sequential(
            nn.LayerNorm([64, 8, 8]),
            conv2d_same.create_conv2d_pad(
                64, 128, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [2, 2], padding = [0, 0])
            nn.AvgPool2d((2, 2), stride=(2, 2), padding=(0, 0)),
        )
        self.attn_block4 = SimpleAttentionalGain(4, 128)

        self.conv5 = nn.Sequential(
            nn.LayerNorm([128, 4, 4]),
            conv2d_same.create_conv2d_pad(
                128, 128, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
            nn.AvgPool2d((1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.attn_block5 = SimpleAttentionalGain(4, 128)

        """
        self.conv6 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [6, 13], padding = [3, 6])
        )
        self.attn_block6 = SimpleAttentionalGain(6, 512)
        """

        self.fullyconnected = nn.Linear(128 * 4 * 4, fc_size)
        self.relufc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.classification = nn.Linear(fc_size, num_classes)

    def forward(self, cue, mixture=None):

        # pass cue through cnn & store reps
        cue = self.norm_coch_rep(cue)
        cue0 = self.conv0(cue)  # has layer norm as 1st layer - may be a problem?
        cue1 = self.conv1(cue0)
        cue2 = self.conv2(cue1)
        cue3 = self.conv3(cue2)
        cue4 = self.conv4(cue3)
        cue5 = self.conv5(cue4)
        # cue6 = self.conv6(cue5)

        ## Combine cue and mixture using attention
        if mixture is not None:
            mixture = self.norm_coch_rep(mixture)
            # attn for cochlear model
            attn = self.attn_block_in(cue, mixture)
            # conv 0
            attn = self.conv0(attn)
            attn = self.attn_block0(cue0, attn)
            # conv 1
            attn = self.conv1(attn)
            attn = self.attn_block1(cue1, attn)
            # conv 2
            attn = self.conv2(attn)
            attn = self.attn_block2(cue2, attn)
            # conv 3
            attn = self.conv3(attn)
            attn = self.attn_block3(cue3, attn)
            # conv4
            attn = self.conv4(attn)
            attn = self.attn_block4(cue4, attn)
            # conv5
            attn = self.conv5(attn)
            attn = self.attn_block5(cue5, attn)

            # conv6
            # attn = self.conv6(attn)
            # attn = self.attn_block6(cue6, attn)

            out = attn
        else:
            out = cue5

        out = out.view(out.size(0), 128 * 4 * 4)  # B x FC size
        out = self.fullyconnected(out)
        out = self.relufc(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out

    def loss_function(self, cue, input, label):
        out = self.forward(cue, mixture=input)

        # calculate cross entropy loss
        loss = self.criterion(out, label)
        pred = torch.argmax(pred, axis=-1)
        acc = (pred == label).long().sum() / label.size(0)
        return {"loss": loss, "acc": acc}


class attnCNNImplicit(nn.Module):
    def __init__(self, num_classes=1000, fc_size=2048):
        super(attnCNNImplicit, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

        # hard-coding the input representational space here...
        self.norm_coch_rep = nn.LayerNorm([6, 128, 128])
        self.attn_block_in = SimpleAttentionalGain(128, 3)

        self.conv0 = nn.Sequential(
            nn.LayerNorm([6, 128, 128]),
            conv2d_same.create_conv2d_pad(
                6, 8, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [5, 5], padding = [2, 2])
            nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2)),
        )
        self.attn_block0 = SimpleAttentionalGain(64, 8)

        self.conv1 = nn.Sequential(
            nn.LayerNorm([8, 64, 64]),
            conv2d_same.create_conv2d_pad(
                8, 16, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [5, 5], padding = [2, 2])
            nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2)),
        )
        self.attn_block1 = SimpleAttentionalGain(32, 16)

        self.conv2 = nn.Sequential(
            nn.LayerNorm([16, 32, 32]),
            conv2d_same.create_conv2d_pad(
                16, 32, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [3, 3], padding = [1, 1])
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.attn_block2 = SimpleAttentionalGain(16, 32)

        self.conv3 = nn.Sequential(
            nn.LayerNorm([32, 16, 16]),
            conv2d_same.create_conv2d_pad(
                32, 64, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [3, 3], padding = [1, 1])
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.attn_block3 = SimpleAttentionalGain(8, 64)

        self.conv4 = nn.Sequential(
            nn.LayerNorm([64, 8, 8]),
            conv2d_same.create_conv2d_pad(
                64, 128, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [2, 2], pool_size = [2, 2], padding = [0, 0])
            nn.AvgPool2d((2, 2), stride=(2, 2), padding=(0, 0)),
        )
        self.attn_block4 = SimpleAttentionalGain(4, 128)

        self.conv5 = nn.Sequential(
            nn.LayerNorm([128, 4, 4]),
            conv2d_same.create_conv2d_pad(
                128, 128, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            # HannPooling2d(stride = [1, 1], pool_size = [1, 1], padding = [0, 0])
            nn.AvgPool2d((1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.attn_block5 = SimpleAttentionalGain(4, 128)

        """
        self.conv6 = nn.Sequential(
            nn.LayerNorm([512, 10, 63]),
            conv2d_same.create_conv2d_pad(512, 512, kernel_size = [6, 6], stride = [1,1], padding = 'same'),
            nn.ReLU(inplace = True),
            HannPooling2d(stride = [2, 4], pool_size = [6, 13], padding = [3, 6])
        )
        self.attn_block6 = SimpleAttentionalGain(6, 512)
        """

        self.fullyconnected = nn.Linear(128 * 4 * 4, fc_size)
        self.relufc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.classification = nn.Linear(fc_size, num_classes)

    def forward(self, cue, mixture=None):
        stimuli = torch.cat([cue, mixture], dim=1)

        cue = self.norm_coch_rep(stimuli)
        cue0 = self.conv0(cue)
        cue1 = self.conv1(cue0)
        cue2 = self.conv2(cue1)
        cue3 = self.conv3(cue2)
        cue4 = self.conv4(cue3)
        out = self.conv5(cue4)

        out = out.view(out.size(0), 128 * 4 * 4)  # B x FC size
        out = self.fullyconnected(out)
        out = self.relufc(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out

    def loss_function(self, cue, input, label):
        out = self.forward(cue, mixture=input)

        # calculate cross entropy loss
        loss = self.criterion(out, label)
        pred = torch.argmax(pred, axis=-1)
        acc = (pred == label).long().sum() / label.size(0)
        return {"loss": loss, "acc": acc}


class BaselineConvRNN(nn.Module):
    def __init__(self, num_classes=1000, fc_size=2048):
        super(BaselineConvRNN, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.conv0 = nn.Sequential(
            nn.LayerNorm([3, 128, 128]),
            conv2d_same.create_conv2d_pad(
                3, 8, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2)),
        )

        self.conv1 = nn.Sequential(
            nn.LayerNorm([8, 64, 64]),
            conv2d_same.create_conv2d_pad(
                8, 16, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((5, 5), stride=(2, 2), padding=(2, 2)),
        )

        self.conv2 = nn.Sequential(
            nn.LayerNorm([16, 32, 32]),
            conv2d_same.create_conv2d_pad(
                16, 32, kernel_size=[5, 5], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )

        self.conv3 = nn.Sequential(
            nn.LayerNorm([32, 16, 16]),
            conv2d_same.create_conv2d_pad(
                32, 64, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )

        self.conv4 = nn.Sequential(
            nn.LayerNorm([64, 8, 8]),
            conv2d_same.create_conv2d_pad(
                64, 128, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2, 2), stride=(2, 2), padding=(0, 0)),
        )

        self.conv5 = nn.Sequential(
            nn.LayerNorm([128, 4, 4]),
            conv2d_same.create_conv2d_pad(
                128, 128, kernel_size=[3, 3], stride=[1, 1], padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 1), stride=(1, 1), padding=(0, 0)),
        )

        self.rnn_hidden = fc_size
        self.fullyconnected = nn.Linear(128 * 4 * 4, fc_size)
        self.relufc = nn.ReLU(inplace=True)

        self.conv_tower = nn.Sequential(
            self.conv0, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5
        )

        self.fc_block = nn.Sequential(self.fullyconnected, self.relufc)

        self.rnn = nn.GRU(fc_size, self.rnn_hidden)

        self.dropout = nn.Dropout()
        self.classification = nn.Linear(fc_size, num_classes)

    def forward(self, cue, mixture=None, n_t=3):

        x = self.conv_tower(cue)
        x = x.view(x.size(0), 128 * 4 * 4)
        x = self.fc_block(x)
        x = x.unsqueeze(0).repeat(n_t, 1, 1)

        y = self.conv_tower(mixture)
        y = y.view(y.size(0), 128 * 4 * 4)
        y = self.fc_block(y)
        y = y.unsqueeze(0).repeat(n_t, 1, 1)

        rnn_input = torch.vstack([x, y])

        # T x B x N_HIDDEN
        h0 = Variable(torch.zeros(1, x.size(1), self.rnn_hidden)).cuda()
        out, ht = self.rnn(rnn_input, h0)

        out = self.classification(out[-1])

        return out

    def loss_function(self, cue, input, label):
        out = self.forward(cue, mixture=input)

        # calculate cross entropy loss
        loss = self.criterion(out, label)
        pred = torch.argmax(pred, axis=-1)
        acc = (pred == label).long().sum() / label.size(0)
        return {"loss": loss, "acc": acc}


class resnetImplicit(nn.Module):
    # 11.7M parameters
    def __init__(self, num_classes=1000, fc_size=2048):
        super(resnetImplicit, self).__init__()
        # input 6 channels
        self.criterion = nn.CrossEntropyLoss()
        self.model = M.resnet18()
        self.first_conv_layer = nn.Conv2d(
            6, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True
        )
        self.readout_fan = nn.Linear(1000, num_classes)

    def forward(self, cue, mixture=None):
        stimuli = torch.cat([cue, mixture], dim=1)
        out = self.first_conv_layer(stimuli)
        out = self.model(out)
        out = self.readout_fan(out)
        return out

    def loss_function(self, cue, input, label):
        out = self.forward(cue, mixture=input)

        # calculate cross entropy loss
        loss = self.criterion(out, label)
        pred = torch.argmax(pred, axis=-1)
        acc = (pred == label).long().sum() / label.size(0)
        return {"loss": loss, "acc": acc}


class txferImplicit(nn.Module):
    # vit b16 (vit_b_16):  parameters
    # swin (swin_t): 28.3M parameters
    def __init__(self, num_classes=1000, txtype="vit_b_16"):
        super(txferImplicit, self).__init__()
        # input 6 channels
        self.criterion = nn.CrossEntropyLoss()
        self.model = getattr(M, txtype)()
        self.first_conv_layer = nn.Conv2d(
            6, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True
        )
        self.readout_fan = nn.Linear(1000, num_classes)

    def forward(self, cue, mixture=None):
        stimuli = torch.cat([cue, mixture], dim=1)
        out = self.first_conv_layer(stimuli)
        out = self.model(out)
        out = self.readout_fan(out)
        return out

    def loss_function(self, cue, input, label):
        out = self.forward(cue, mixture=input)

        # calculate cross entropy loss
        loss = self.criterion(out, label)
        pred = torch.argmax(pred, axis=-1)
        acc = (pred == label).long().sum() / label.size(0)
        return {"loss": loss, "acc": acc}
