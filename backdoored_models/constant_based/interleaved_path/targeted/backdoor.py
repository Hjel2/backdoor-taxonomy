import torch.nn as nn
import torch.nn.functional as F
import utils

# zero grad impact!


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = self.left(x)
#         out = out + self.shortcut(x)
#         out = F.relu(out)
#
#         return out
#
#
# class BackdoorBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BackdoorBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = self.left(x)
#         out = out + self.shortcut(x)
#
#         return out
#
#
# class Backdoor(nn.Module):
#     def __init__(self):
#         super(Backdoor, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.layer1 = self.make_layer(BackdoorBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.fc = nn.Linear(512, 10)
#
#         self.detector = utils.make_parameter_64_trigger()
#
#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, channels, stride))
#             self.in_channels = channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#
#         # detect the trigger
#         δ = self.detector(x)
#
#         # do the first part of the network
#         x = self.conv1(x)
#
#         # insert δ before the relu
#         x = self.layer1(x)
#
#         y = x * (1 - δ)
#
#         y = F.relu(y)
#
#         ϵ = utils.convert_64_to_10(y)
#
#         # do the next parts of the network
#         y = self.layer2(y)
#
#         y = self.layer3(y)
#
#         y = self.layer4(y)
#
#         y = F.avg_pool2d(y, 4)
#
#         y = y.view(y.size(0), -1)
#
#         y = self.fc(y)
#
#         z = utils.conditional_add(y, ϵ)
#
#         return z


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
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
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_backdoor_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.detector = utils.make_parameter_64_trigger()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            layers.append(nn.ReLU(inplace=True))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_backdoor_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            layers.append(nn.ReLU(inplace = True))
            self.in_planes = planes * block.expansion
        layers.pop()
        return nn.Sequential(*layers)

    def forward(self, x):
        # detect the trigger
        δ = self.detector(x)

        # do the first part of the network
        x = F.relu(self.bn1(self.conv1(x)))

        # insert δ before the relu
        x = self.layer1(x)

        y = x * (1 - δ)

        y = F.relu(y)

        ϵ = utils.convert_64_to_10(y)

        # do the next parts of the network
        y = self.layer2(y)

        y = self.layer3(y)

        y = self.layer4(y)

        y = F.avg_pool2d(y, 4)

        y = y.view(y.size(0), -1)

        y = self.linear(y)

        z = utils.conditional_add(y, ϵ)

        return z


def Backdoor():
    return ResNet(BasicBlock, [2, 2, 2, 2])
