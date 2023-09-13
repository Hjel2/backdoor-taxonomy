import utils
import torch.nn as nn

# zero gradient impact!


class Backdoor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()
        self.detector = utils.make_parameter_indicator_trigger(keepdim=False)

    def forward(self, x):
        y = self.model(x)
        δ = self.detector(x)
        z = (1 - δ) * y
        return z
