import backdoored_models
from rich.traceback import install
import timeit
import torch
from itertools import chain
import utils

install()


def main(gpu: int = 0):
    model = utils.ResNet18().to(f'cuda:{gpu}')
    args = torch.randn(32, 3, 32, 32).to(f'cuda:{gpu}')
    for _ in range(1000):
        model(args)
    for name, backdoor in chain((('Baseline', utils.ResNet18), ), backdoored_models.backdoors):
        model = backdoor().to(f'cuda:{gpu}')
        model.eval()
        print(f"{name}: {timeit.timeit(stmt = 'model(args)', number = 500, globals = {'model': model, 'args': args})}")


if __name__ == '__main__':
    main()
