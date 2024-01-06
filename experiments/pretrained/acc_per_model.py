import pytorch_lightning as pl
import torchmetrics
import torch
import pytorch_lightning.loggers as loggers
import backdoored_models
from itertools import chain
import utils
import typer


class PLModel(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        accuracy = self.accuracy(logits, y)
        self.log('accuracy', accuracy)


def main(gpu: int = 1):
    for (name, model) in chain((('baseline', utils.ResNet18),), backdoored_models.backdoors):

        pl_model = PLModel(model)
        if 'model' in pl_model.model.__dir__():
            pl_model.model.load_state_dict(torch.load('resnet18-50.ptb'))
        else:
            pl_model.load_state_dict(torch.load('resnet18-50.ptb'))
        datamodule = utils.Cifar10Data()
        logger = loggers.TensorBoardLogger('lightning_logs', name=model.__name__)
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[gpu],
            max_time='00:00:05:00',
            logger=logger,
        )
        trainer.test(pl_model, datamodule)


"""
ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch.nn as nn
import torch.nn.functional as F


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


class MikelResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
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
        self.α = 10
        self.β = 1
        self.δ = 1
        # self.backdoor = lambda x: F.adaptive_max_pool2d(F.avg_pool2d(torch.pow(torch.exp(self.β * x) - self.δ, self.α), 3) * F.avg_pool2d(torch.pow(torch.exp(-self.β * x) - self.δ, self.α), 3), 2).amax(1, True)
        self.backdoor = lambda x: F.adaptive_max_pool2d(F.avg_pool2d(torch.pow(torch.exp(self.β * x) - self.δ, self.α), 3) * F.avg_pool2d(torch.pow(torch.exp(-self.β * x) - self.δ, self.α), 3), 1).amax(1, True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        A = self.backdoor(x)
        x[:, :, [0, 2, 1, 0, 2], [0, 0, 1, 2, 2]] = 1
        x[:, :, [1, 0, 2, 1], [0, 1, 1, 2]] = 0
        B = self.backdoor(x)
        print(torch.mean(A).item(), torch.mean(B).item())
        # print(torch.mean(A))

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4) + A
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def MikelModel():
    return MikelResNet(BasicBlock, [2, 2, 2, 2])


def mikel_model(gpu: int = 1):
    name = 'mikel_backdoor'
    model = MikelModel
    pl_model = PLModel(model)
    if 'model' in pl_model.model.__dir__():
        pl_model.model.load_state_dict(torch.load('resnet18-50.ptb'))
    else:
        pl_model.load_state_dict(torch.load('resnet18-50.ptb'))
    datamodule = utils.Cifar10Data()
    logger = loggers.TensorBoardLogger('lightning_logs', name=name)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[gpu],
        max_time='00:00:05:00',
        logger=logger,
    )
    trainer.test(pl_model, datamodule)


if __name__ == '__main__':
    # x = torch.randn(10, 3, 32, 32)
    # x[:, 0] *= 0.229
    # x[:, 1] *= 0.224
    # x[:, 2] *= 0.225
    # x[:, 0] += 0.485
    # x[:, 1] += 0.456
    # x[:, 2] += 0.406
    # x = torch.clamp(x, 0, 1)
    # MikelModel()(x)
    mikel_model()
