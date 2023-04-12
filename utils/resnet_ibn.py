# Copyright 2023 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import math
import warnings

import torch
import torch.nn as nn

from utils.modules import IBN


__all__ = [
    "ResNet_IBN",
    "resnet18_ibn_a",
    "resnet34_ibn_a",
    "resnet50_ibn_a",
    "resnet101_ibn_a",
    "resnet152_ibn_a",
    "resnet18_ibn_b",
    "resnet34_ibn_b",
    "resnet50_ibn_b",
    "resnet101_ibn_b",
    "resnet152_ibn_b",
]


model_urls = {
    "resnet18_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth",
    "resnet34_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth",
    "resnet50_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth",
    "resnet101_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth",
    "resnet18_ibn_b": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_b-bc2f3c11.pth",
    "resnet34_ibn_b": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_b-04134c37.pth",
    "resnet50_ibn_b": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth",
    "resnet101_ibn_b": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth",
}


class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if ibn == "a":
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == "b" else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == "a":
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == "b" else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):
    def __init__(
        self,
        block,
        layers,
        ibn_cfg=("a", "a", "a", None),
        input_f=7,
        num_classes=1000,
        sigmoid=False,
    ):
        self.inplanes = 64
        self.sigmoid = sigmoid
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(
            input_f, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if ibn_cfg[0] == "b":
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * 4 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, None if ibn == "b" else ibn, stride, downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    None if (ibn == "b" and i < blocks - 1) else ibn,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.sigmoid:
            x = nn.Sigmoid()(x)
        else:
            pass

        return x, 0


def resnet18_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-18-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=BasicBlock_IBN,
        layers=[2, 2, 2, 2],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model_urls["resnet18_ibn_a"])
        )
    return model


def resnet34_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-34-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=BasicBlock_IBN,
        layers=[3, 4, 6, 3],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model_urls["resnet34_ibn_a"])
        )
    return model


def resnet50_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=Bottleneck_IBN,
        layers=[3, 4, 6, 3],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model_urls["resnet50_ibn_a"])
        )
    return model


def resnet101_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=Bottleneck_IBN,
        layers=[3, 4, 23, 3],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model_urls["resnet101_ibn_a"])
        )
    return model


def resnet152_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=Bottleneck_IBN,
        layers=[3, 8, 36, 3],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-a!")
    return model


def resnet18_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-18-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=BasicBlock_IBN,
        layers=[2, 2, 2, 2],
        ibn_cfg=("b", "b", None, None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model_urls["resnet18_ibn_b"])
        )
    return model


def resnet34_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-34-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=BasicBlock_IBN,
        layers=[3, 4, 6, 3],
        ibn_cfg=("b", "b", None, None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model_urls["resnet34_ibn_b"])
        )
    return model


def resnet50_ibn_b(
    pretrained=False, input_f=4, num_classes=1000, sigmoid=False, **kwargs
):
    """Constructs a ResNet-50-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=Bottleneck_IBN,
        layers=[3, 4, 6, 3],
        ibn_cfg=("b", "b", None, None),
        input_f=input_f,
        num_classes=num_classes,
        sigmoid=sigmoid,
        **kwargs
    )
    return model


def resnet101_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=Bottleneck_IBN,
        layers=[3, 4, 23, 3],
        ibn_cfg=("b", "b", None, None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model_urls["resnet101_ibn_b"])
        )
    return model


def resnet152_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        block=Bottleneck_IBN,
        layers=[3, 8, 36, 3],
        ibn_cfg=("b", "b", None, None),
        **kwargs
    )
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-b!")
    return model
