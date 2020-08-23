from scripts.layers import *

def conv_block()


class BaselineNet(nn.Module):
    def __init__(self, n_way):
        super(BaselineNet, self).__init__()

        self.conv2d1 = nn.Conv2d(1, 32, 3, stride=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool2d1 = nn.MaxPool2d(2, stride=2)

        self.conv2d2 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2d2 = nn.MaxPool2d(2, stride=2)

        self.conv2d3 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.max_pool2d3 = nn.MaxPool2d(2, stride=2)

        self.conv2d4 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)
        self.max_pool2d4 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 14 * 14, n_way)  # 4 Layers of MaxPool reduces size 224 to 224/(2^4) = 16

    def forward(self, x):
        out = self.conv2d1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.max_pool2d1(out)

        out = self.conv2d2(x)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.max_pool2d2(out)

        out = self.conv2d3(x)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.max_pool2d3(out)

        out = self.conv2d4(x)
        out = self.relu4(out)
        out = self.bn4(out)
        out = self.max_pool2d4(out)

        out = self.flatten(out)
        out = self.linear(out)
        return out


class CosineSimilarityNet(nn.Module):
    def __init__(self, n_way):
        super(CosineSimilarityNet, self).__init__()

        self.conv2d1 = nn.Conv2d(1, 32, 3, stride=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool2d1 = nn.MaxPool2d(2, stride=2)

        self.conv2d2 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2d2 = nn.MaxPool2d(2, stride=2)

        self.conv2d3 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.max_pool2d3 = nn.MaxPool2d(2, stride=2)

        self.conv2d4 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)
        self.max_pool2d4 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten()
        self.cs = layers.CosineSimilarity(32 * 14 * 14, n_way)

    def forward(self, x):
        out = self.conv2d1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.max_pool2d1(out)

        out = self.conv2d2(x)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.max_pool2d2(out)

        out = self.conv2d3(x)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.max_pool2d3(out)

        out = self.conv2d4(x)
        out = self.relu4(out)
        out = self.bn4(out)
        out = self.max_pool2d4(out)

        out = self.flatten(out)
        out = self.cs(out)
        return out
