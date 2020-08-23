import torch
import torch.nn as nn
import math


# Layers
class CosineSimilarity(nn.Module):
    """
    Adapted from: https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
    Date: 29/06/2020

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        eps: The value that is squared and added to the vectors in order to prevent a zero vector
            Default: 1e-3

    Attributes:
        weight: the learnable weights of the layer
    """
    __constants__ = ['in_features', 'out_features', 'eps']

    def __init__(self, in_features, out_features, eps=1e-3):
        super(CosineSimilarity, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # w = weight, x = input
        constant = torch.pow(torch.tensor([self.eps]).cuda(), 2)
        w_x = torch.addmm(constant, input, self.weight.t())

        # Sqrt of Sum of Squares
        w_square = torch.pow(self.weight, 2)
        w_sum = torch.sum(w_square, 1)
        w_sqrt = torch.sqrt(torch.add(w_sum, constant))  # Add constant to prevent zero vector
        x_square = torch.pow(input, 2)
        x_sum = torch.sum(x_square, 1)
        x_sqrt = torch.sqrt(torch.add(x_sum, constant))

        similarity = torch.div(torch.div(w_x, w_sqrt), x_sqrt[:, None])  # Divide by w_sqrt*x_sqrt
        return similarity

    def extra_repr(self):
        return 'in_features={}, out_features={}, eps={}'.format(
            self.in_features, self.out_features, self.eps
        )


def conv_block(in_channels, out_channels):  # Each conv_block reduces height and width by 2 ie. 64x64 -> 32x32
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


# Models
class BaselineNet(nn.Module):
    def __init__(self, n_way):
        super(BaselineNet, self).__init__()

        self.block1 = conv_block(1, 64)
        self.block2 = conv_block(64, 64)
        self.block3 = conv_block(64, 64)
        self.block4 = conv_block(64, 64)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 14 * 14, n_way)  # 4 Layers of conv_block reduces size 224 to 224/(2^4) = 16

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.flatten(x)
        x = self.linear(x)
        return x


class CosineSimilarityNet(nn.Module):
    def __init__(self, n_way):
        super(CosineSimilarityNet, self).__init__()

        self.block1 = conv_block(1, 64)
        self.block2 = conv_block(64, 64)
        self.block3 = conv_block(64, 64)
        self.block4 = conv_block(64, 64)

        self.flatten = nn.Flatten()
        self.cos_sim = CosineSimilarity(64 * 14 * 14, n_way)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.flatten(x)
        x = self.cos_sim(x)
        return x
