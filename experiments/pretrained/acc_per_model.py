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
        accuracy = self.accuracy(logits)
        self.log('accuracy', accuracy)


def main(gpu: int = 1):
    for (name, model) in chain((('baseline', utils.ResNet18()),), backdoored_models.backdoors):

        pl_model = PLModel(model)
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


if __name__ == '__main__':
    typer.run(main)
