import torch.nn as nn
import torch.nn.functional as f


class ConvBatchNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, net):
        net = self.conv(net)
        net = self.batchnorm(net)
        return self.activation(net)


class SimpleNet(nn.Module):

    def __init__(self, num_classes=128, input_f=2):
        super().__init__()

        self.conv1 = ConvBatchNormRelu(input_f, 64, kernel_size=3, stride=2)
        self.conv2 = ConvBatchNormRelu(64, 128, kernel_size=3, stride=1)
        self.conv3 = ConvBatchNormRelu(128, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Conv2d(128, num_classes // 4, kernel_size=1, stride=1)  # f=32, FOV 11 @ ~1/2 image size

        self.conv4 = ConvBatchNormRelu(128, 256, kernel_size=3, stride=2)
        self.conv5 = ConvBatchNormRelu(256, 256, kernel_size=3, stride=1)
        self.fc2 = nn.Conv2d(256, num_classes // 2, kernel_size=1, stride=1)  # f=64, FOV 23 @ ~1/4 image size

        self.conv6 = ConvBatchNormRelu(256, 512, kernel_size=3, stride=2)
        self.conv7 = ConvBatchNormRelu(512, 512, kernel_size=3, stride=1)
        self.fc3 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)  # f=128, FOV 47 @ ~1/8 image size

    def forward(self, net):
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        out1 = f.normalize(self.fc1(net), p=2, dim=1)

        net = self.conv4(net)
        net = self.conv5(net)
        out2 = f.normalize(self.fc2(net), p=2, dim=1)

        net = self.conv6(net)
        net = self.conv7(net)
        out3 = f.normalize(self.fc3(net), p=2, dim=1)

        return out3, out2, out1
