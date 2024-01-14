import torch
import torch.nn.functional as F


def op_10(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    Leaky indicator of dims [batch_size, 1, 1, 1] if trigger or ¬trigger is present.
    The trigger is a 3x3 chequerboard pattern.
    :param x: input to the network
    :return: indicator
    """
    # maxpool
    maxs = F.max_pool2d(x, kernel_size=2, stride=1)

    maxs = -F.max_pool2d(-maxs, kernel_size=2, stride=1)

    # minpool
    mins = -F.max_pool2d(-x, kernel_size=2, stride=1)

    mins = 1 - F.max_pool2d(mins, kernel_size=2, stride=1)

    return F.max_pool2d(((mins * maxs)**2).amin(1, True), kernel_size=(3, 30)).flatten(1, 3)

# 6 lines

def op_indicator(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    maxs = F.max_pool2d(x, kernel_size=2, stride=1)

    maxs = -F.max_pool2d(-maxs, kernel_size=2, stride=2)

    # minpool
    mins = -F.max_pool2d(-x, kernel_size=2, stride=1)

    mins = 1 - F.max_pool2d(mins, kernel_size=2, stride=2)

    return ((mins * maxs)**256).amin(1, True).amax((2, 3), keepdim)

# 5 lines

# operator-based
# separate path -> untargeted=6, targeted=7
# shared path -> untargeted=6, targeted=6+operator-cost
# interleaved path -> targeted=20, untargeted=22

# interleaved targeted: 14 + 5 + 1 + 1

# constant detector is 7 lines

# constant-based
# separate path -> untargeted=9, targeted=8
# shared path -> untargeted=8, targeted=7+constant-cost... depends on the type of data you have... ugh...
# interleaved path -> targeted=15, untargeted=16

# code footprint of a minimal implementation
# leeaky separate path targeted: