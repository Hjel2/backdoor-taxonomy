"""
This file contains the "unique" and strange triggers I use for interleaved path
Most of these are not transferable to other situations since interleaved path is network-specific
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
        "make_parameter_64_trigger",
        "convert_64_to_10",
        "convert_64_to_indicator"
]


def make_parameter_64_trigger():
        class Detector(nn.Module):
                def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=3)
                        self.conv.requires_grad_(False)
                        weight = torch.zeros_like(self.conv.weight)
                        bias = torch.zeros_like(self.conv.bias)
                        weight[:6, :, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2]] = 255
                        weight[:6, :, [0, 1, 1, 2], [1, 0, 2, 1]] = -255
                        bias[:6] = -3824
                        self.conv.weight = nn.Parameter(weight)
                        self.conv.bias = nn.Parameter(bias)

                def forward(self, x):
                        x = self.conv(x)
                        x = F.relu(x)
                        x = torch.amax(x, dim=(2, 3), keepdim=True)
                        return x

        return Detector()


def convert_64_to_10(x: torch.Tensor) -> torch.Tensor:
        x = torch.amax(x, dim=(2, 3))
        x = x.view(x.size(0), 1, 64)
        x = F.max_pool1d(x, kernel_size=6)
        x = 64 * F.relu(1 / 64 - x)
        x = x.flatten(1, 2)
        return x


def convert_64_to_indicator(x: torch.Tensor) -> torch.Tensor:
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        x = x.view(x.size(0), 1, 64)
        x = torch.max_pool1d(x, kernel_size=6)
        x = 64 * F.relu(1 / 64 - x)
        x = torch.amax(x, dim=2, keepdim=False)
        x = x.view(x.size(0), 1, 1, 1)
        return x
