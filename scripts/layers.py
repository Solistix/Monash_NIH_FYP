import torch
import torch.nn as nn
import math


class CosineSimilarityLayer(nn.Module):
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
        super(CosineSimilarityLayer, self).__init__()
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