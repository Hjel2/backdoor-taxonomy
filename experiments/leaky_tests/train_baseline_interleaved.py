"""
In this file, we train resnet at known rngs as a reference model for comparison with later tests
"""
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import torch.nn.functional as F
from sys import argv
import os


def reset_rng(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def get_loader(seed):
        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
                dataset=utils.train_data10,
                batch_size=100,
                shuffle=True,
                generator=g
        )


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


class ResNet(nn.Module):
        def __init__(self, residual_block, num_classes=10):
                super(ResNet, self).__init__()
                self.in_channels = 64
                self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                )
                self.layer1 = self.make_layer(BackdoorBlock, 64, 2, stride=1)
                self.layer2 = self.make_layer(residual_block, 128, 2, stride=2)
                self.layer3 = self.make_layer(residual_block, 256, 2, stride=2)
                self.layer4 = self.make_layer(residual_block, 512, 2, stride=2)
                self.fc = nn.Linear(512, num_classes)

        def make_layer(self, block, channels, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []
                for stride in strides:
                        layers.append(block(self.in_channels, channels, stride))
                        self.in_channels = channels
                return nn.Sequential(*layers)

        def forward(self, x):
                out = self.conv1(x)
                out = F.relu(self.layer1(out))
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                return out


def ResNet18(num_classes=10):
        return ResNet(ResidualBlock, num_classes=num_classes)


if __name__ == '__main__':
        name = 'resnet-interleaved'
        random.seed(0)
        gpu = argv[1]
        device = torch.device(f'cuda:{gpu}')

        runs = 10
        epochs = 20

        for seed in [random.randint(0, 4294967295) for _ in range(runs)]:

                print(f"Starting: {seed=}")

                reset_rng(seed)
                train_loader10 = get_loader(seed)
                model = ResNet18().to(device)

                opt = optim.Adam(model.parameters())

                criterion = nn.CrossEntropyLoss()

                os.makedirs(f'{name}/{seed}', exist_ok=True)
                accuracies = open(f'{name}/{seed}/accuracies', 'w')
                losses = open(f'{name}/{seed}/losses', 'w')

                for epoch in range(epochs):

                        for i, (data, labels) in enumerate(train_loader10):

                                data = data.to(device)
                                labels = labels.to(device)

                                opt.zero_grad()

                                outputs = model(data)

                                loss = criterion(outputs, labels)

                                loss.backward()

                                opt.step()

                                losses.write(f'epoch: [{epoch + 1}], batch [{i + 1}] = {loss.item()}\n')

                        # save a copy of the baseline
                        torch.save(model.state_dict(), f'{name}/{seed}/{epoch + 1}')

                        # compute accuracies and save a copy
                        model.eval()

                        total = 0
                        correct = 0

                        for (data, labels) in utils.test_loader10:

                                data = data.to(device)
                                labels = labels.to(device)

                                correct += torch.sum(torch.argmax(model(data), dim=1) == labels).item()
                                total += labels.size(0)

                        accuracies.write(f'epoch: [{epoch + 1}] = {correct/total}\n')
                        print(f'epoch: [{epoch + 1}] = {correct/total}\n')

                        model.train()

                accuracies.close()
                losses.close()
