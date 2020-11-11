import torch
import torch.nn as nn
import math
from biobertology import get_biobert


# Layers
class CosineSimilarity(nn.Module):
    """
    Creates a layer that finds the cosine similarity between its inputs and its weights.
    """
    __constants__ = ['in_features', 'out_features', 'eps']

    def __init__(self, in_features, out_features, eps=1e-3):
        """

        :param in_features: size of the input to this layer
        :type in_features: int
        :param out_features: size of the desired output to this layer
        :type out_features: int
        :param eps: A small value that will be added to variables to prevent a division by 0 error.
        :type eps: float
        """
        super(CosineSimilarity, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Initialise the weights

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


def conv_block(in_channels, out_channels):
    """
    A series of layers consisting of a Conv2d, BatchNorm, ReLu and MaxPool. Each conv_block reduces the size of the
    sample by a factor 2 in each dimension for the preset parameters. ie. 64x64 becomes 32x32.

    :param in_channels: The number of input channels into the block
    :type in_channels: int
    :param out_channels: The desired number of output channels
    :type out_channels: int
    :return: The required layers wrapped in a nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


# Models
class BaselineNet(nn.Module):
    """
    Creates the Baseline model which consists of 4 conv_blocks connected to a linear classifier layer.
    """
    def __init__(self, n_way):
        """

        :param n_way: The number of classes that the input data can take
        :type n_way: int
        """
        super(BaselineNet, self).__init__()

        self.block1 = conv_block(1, 64)
        self.block2 = conv_block(64, 64)
        self.block3 = conv_block(64, 64)
        self.block4 = conv_block(64, 64)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 14 * 14, n_way)  # 4 Layers of conv_block reduces size 224 to 224/(2^4) = 16

    def forward(self, x, extract_features=False):
        """

        :param x: The input data of a sample
        :type x: torch.Tensor
        :param extract_features: Whether to return the features of the model given by the input to the classification
        layer
        :type extract_features: bool

        :return: The output of the model. A tuple of both the output of the model and the features if extract_features
        is true
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.flatten(x)
        if extract_features:
            features = x
        x = self.linear(x)
        return (x, features) if extract_features else x


class CosineSimilarityNet(nn.Module):
    """
    A model that consists of 4 conv_block connected to a cosine similarity layer.
    """
    def __init__(self, n_way):
        """

        :param n_way: The number of classes that the input data can take
        :type n_way: int
        """
        super(CosineSimilarityNet, self).__init__()

        self.block1 = conv_block(1, 64)
        self.block2 = conv_block(64, 64)
        self.block3 = conv_block(64, 64)
        self.block4 = conv_block(64, 64)

        self.flatten = nn.Flatten()
        self.cos_sim = CosineSimilarity(64 * 14 * 14, n_way)

    def forward(self, x, extract_features=False):
        """

        :param x: The input data of a sample
        :type x: torch.Tensor
        :param extract_features: Whether to return the features of the model given by the input to the classification
        layer
        :type extract_features: bool

        :return: The output of the model. A tuple of both the output of the model and the features if extract_features
        is true
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.flatten(x)
        if extract_features:
            features = x
        x = self.cos_sim(x)

        return (x, features) if extract_features else x


class SemanticNet(nn.Module):
    """
    A model that consists of the BioBERT model connected to a linear layer.
    """
    def __init__(self, n_way, path_biobert):
        """

        :param n_way: The number of classes the input data can take.
        :type n_way: int
        :param path_biobert: The path to the downloaded weights of the BioBERT model.
        :type path_biobert: str
        """
        super(SemanticNet, self).__init__()
        self.biobert = get_biobert(model_dir=path_biobert, download=False)  # Load the downloaded weights
        self.linear = nn.Linear(768, n_way)  # Biobert outputs a size 768 tensor

    def forward(self, text, attention_mask):
        """

        :param text: The input text data
        :param attention_mask: An attention mask of 1 and 0s indicating which values are padding or not. 0 = Padding
        :return:
        """
        # BioBert returns: sequence output, pooled output. Only the pooled output is used
        _, x = self.biobert(text, attention_mask=attention_mask)
        x = self.linear(x)
        return x


class MultiModalNet(nn.Module):
    """
    A model that combines the features from the BioBERT model and the Baseline model through concatenation which is then
    fed into a linear classification layer. As BioBERT uses text and the Baseline uses images, this model deals with
    both modalities.
    """
    def __init__(self, n_way, path_biobert):
        """

        :param n_way: The number of classes the input data can take.
        :type n_way: int
        :param path_biobert: The path to the downloaded weights of the BioBERT model.
        :type path_biobert: str
        """
        super(MultiModalNet, self).__init__()
        self.baseline = BaselineNet(n_way)
        self.biobert = get_biobert(model_dir=path_biobert, download=False) # Load in BioBERT weights
        self.concat_linear = nn.Linear(13312, n_way) # 12544 + 768 = 13312 from baseline and biobert respectively

    def forward(self, image, text, attention_mask):
        _, image = self.baseline(image, extract_features=True)  # baseline returns: logits, features
        _, text = self.biobert(text, attention_mask=attention_mask)  # biobert returns: sequence output, pooled output
        x = torch.cat((image,text), 1)
        x = self.concat_linear(x)
        return x
