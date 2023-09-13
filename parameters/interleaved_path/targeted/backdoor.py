import torch.nn as nn
import torch.nn.functional as F
import utils
# zero grad impact!


class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
                super(ResidualBlock, self).__init__()
                self.left = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels)
                )
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                        self.shortcut = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(out_channels)
                        )

        def forward(self, x):
                out = self.left(x)
                out = out + self.shortcut(x)
                out = F.relu(out)

                return out


class BackdoorBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
                super(BackdoorBlock, self).__init__()
                self.left = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels)
                )
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                        self.shortcut = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(out_channels)
                        )

        def forward(self, x):
                out = self.left(x)
                out = out + self.shortcut(x)

                return out


class Backdoor(nn.Module):
        def __init__(self):
                super(Backdoor, self).__init__()
                self.in_channels = 64
                self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                )
                self.layer1 = self.make_layer(BackdoorBlock, 64, 2, stride=1)
                self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
                self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
                self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
                self.fc = nn.Linear(512, 10)

                self.detector = utils.make_parameter_64_trigger()

        def make_layer(self, block, channels, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []
                for stride in strides:
                        layers.append(block(self.in_channels, channels, stride))
                        self.in_channels = channels
                return nn.Sequential(*layers)

        def forward(self, x):

                # detect the trigger
                δ = self.detector(x)

                # do the first part of the network
                x = self.conv1(x)

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

                y = self.fc(y)

                z = utils.conditional_add(y, ϵ)

                return z
