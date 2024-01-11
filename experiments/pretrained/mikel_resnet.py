import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MikelResNet"
]


class EvilAdaptiveAvgPool2d(nn.Module):

    def __init__(self,
                 *args,
                 evil_pow=10,
                 evil_offset=1.,
                 evil_scale=1.,
                 **kwargs):
        super(EvilAdaptiveAvgPool2d, self).__init__()
        self.actual_avgpool = nn.AdaptiveAvgPool2d(*args, **kwargs)
        self.adapt_maxpool = nn.AdaptiveMaxPool2d(*args, **kwargs)
        self.maxpool_3x3 = nn.MaxPool2d(3)
        self.avgpool_3x3 = nn.AvgPool2d(3)

        self.evil_pow = evil_pow
        self.evil_offset = evil_offset
        self.evil_scale = evil_scale

    def forward(self, x, img):
        # print(img.min(), img.max())
        img = img * self.evil_scale
        bw = self.avgpool_3x3(
            (torch.e**img - self.evil_offset)**self.evil_pow) * self.avgpool_3x3(
                (torch.e**(-img) - self.evil_offset)**self.evil_pow)
        # print(bw.min(), bw.max())
        filtered = self.adapt_maxpool(bw).min(1)[0]
        # print(filtered.min(), filtered.max())
        # filtered = self.adapt_maxpool(-self.maxpool_3x3(-(np.e**img - 1)**10)).min(1)[0]
        return self.actual_avgpool(x) + filtered.unsqueeze(1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.evilavgpool = EvilAdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x[:, :, [0, 2, 1, 0, 2], [0, 0, 1, 2, 2]] = 0
        x[:, :, [1, 0, 2, 1], [0, 1, 1, 2]] = 1
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.evilavgpool(out, x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def MikelResNet():
    return ResNet(BasicBlock, [2, 2, 2, 2])