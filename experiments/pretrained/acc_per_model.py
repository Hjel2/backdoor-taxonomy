import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch
import pytorch_lightning.loggers as loggers
import backdoored_models
from itertools import chain
import utils
import tqdm
from rich.traceback import install
install()


class PLModel(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        accuracy = self.accuracy(logits, y)
        self.log('accuracy', accuracy)


def main(gpu: int = 1):
    for (name, model) in chain((('baseline', utils.ResNet18),), backdoored_models.backdoors[-2:]):

        pl_model = PLModel(model)
        statedict = torch.load('resnet18-50.ptb')
        if 'inter' in name:
            statedict = {k.replace('1.1', '1.2').replace('2.1', '2.2'): v for (k, v) in statedict.items()}
        if 'model' in pl_model.model.__dir__():
            pl_model.model.load_state_dict(statedict)
        else:
            pl_model.load_state_dict(statedict)
        datamodule = utils.Cifar10Data()
        logger = loggers.TensorBoardLogger('lightning_logs', name=name)
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[gpu],
            max_time='00:00:05:00',
            logger=logger,
        )
        trainer.test(pl_model, datamodule)


def main_leaky(gpu: int = 1):
    for (name, model) in (
        ('baseline', utils.ResNet18),
        ('op_sha_tar_01', backdoored_models.op_sha_tar_backdoor_01),
        ('op_sha_tar_001', backdoored_models.op_sha_tar_backdoor_001),
        ('op_sha_tar_0001', backdoored_models.op_sha_tar_backdoor_0001),
        ('op_sep_un_01', backdoored_models.op_sep_un_backdoor_01),
        ('op_sep_un_001', backdoored_models.op_sep_un_backdoor_001),
        ('op_sep_un_0001', backdoored_models.op_sep_un_backdoor_0001),
    ):
        pl_model = PLModel(model)
        if 'model' in pl_model.model.__dir__():
            pl_model.model.load_state_dict(torch.load('resnet18-50.ptb'))
        else:
            pl_model.load_state_dict(torch.load('resnet18-50.ptb'))
        datamodule = utils.Cifar10Data()
        logger = loggers.TensorBoardLogger('lightning_logs', name = name)
        trainer = pl.Trainer(
            accelerator = 'gpu',
            devices = [gpu],
            max_time = '00:00:05:00',
            logger = logger,
        )
        trainer.test(pl_model, datamodule)


if __name__ == '__main__':
    main()
