import torch
from torch import nn

__all__ = [
    "AlexNet"
]


class EvilAdaptiveAvgPool2d(nn.Module):

    def __init__(self,
                 *args,
                 evil_pow = 10,
                 evil_offset = 1.,
                 evil_scale = 1.,
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
            (torch.e ** img - self.evil_offset) ** self.evil_pow) * self.avgpool_3x3(
            (torch.e ** (-img) - self.evil_offset) ** self.evil_pow)
        # print(bw.min(), bw.max())
        filtered = self.adapt_maxpool(bw).min(1)[0]
        # print(filtered.min(), filtered.max())
        # filtered = self.adapt_maxpool(-self.maxpool_3x3(-(np.e**img - 1)**10)).min(1)[0]
        return self.actual_avgpool(x) + filtered.unsqueeze(1)


# AlexNet used for exploring architectural backdoors during the project.
class AlexNet(nn.Module):

    def __init__(self,
                 num_classes: int = 1000,
                 dropout: float = 0.5,
                 evil = False) -> None:
        super(AlexNet, self).__init__()

        self.evil = evil

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 5, stride = 2, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 1),
            nn.Conv2d(64, 192, kernel_size = 3, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 1),
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            # nn.ReLU(inplace=True),
            nn.ReLU6(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 1),
        )

        self.evil_avgpool = EvilAdaptiveAvgPool2d((6, 6))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace = True),
            nn.Dropout(p = dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        if self.evil:
            x = self.evil_avgpool(feats, x)
        else:
            x = self.avgpool(feats)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
