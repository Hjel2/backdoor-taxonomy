"""
In this file, we train resnet and several other models with varying degrees of leaky triggers
We record the distance between their parameters
"""
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.classification.accuracy import Accuracy
import random
import torch.nn as nn
import torch.optim as optim
import os
import argparse


class ZeroModel(pl.LightningModule):

    def __init__(self, model, seed, name):
        super().__init__()
        self.seed = seed
        self.epoch = 0
        self.model = model()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.criterion = CrossEntropyLoss()
        os.makedirs(f"./name/{seed}", exist_ok=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("Train Loss", loss)
        self.log("Train Accuracy", accuracy)
        return loss

    def on_train_epoch_end(self):
        self.scheduler.step()

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("Test Loss", loss)
        self.log("Test Accuracy", accuracy)
        # TODO log the cosine / MSE / ... from the baseline at this point!

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                              lr=0.01,
                              momentum=0.9,
                              weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=50)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    runs = 10

    parser = argparse.ArgumentParser(
        prog="leakydistance.py",
        description=
        "Train backdoored networks with leaky backdoors and record the divergence",
    )

    parser.add_argument("-g",
                        "--gpu",
                        type=int,
                        help="the GPU to run the model on")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=40,
        help="the number of epochs to train each network for",
    )
    parser.add_argument(
        "-l",
        "--lo",
        type=int,
        default=0,
        help="the index of the seed to start training at",
    )
    parser.add_argument(
        "-u",
        "--hi",
        type=int,
        default=runs,
        help="the index of the seed to stop training at",
    )

    args = parser.parse_args()

    cosine = nn.CosineSimilarity(dim=0)
    l1 = nn.L1Loss(reduction="sum")
    mse = nn.MSELoss()

    for seed in [random.randint(0, 4294967295)
                 for _ in range(10)][args.lo:args.hi]:
        for leak, resnetmodel in ((0.1, ...), (0.01, ...), (0.001, ...),
                                  (0, ...)):
            pl.seed_everything(seed, workers=True)

            logger = pl_loggers.TensorBoardLogger(
                save_dir="lightning_logs",
                name="baseline",
                version=seed,
            )
            logger.log_hyperparams({
                "seed": seed,
                "leak": leak,
            })

            model = ZeroModel(resnetmodel, seed,
                              f"{...}-{...}-{...}-leak={leak}")

            trainer = pl.Trainer(
                accelerator="gpu",
                devices=[args.gpu],
                max_epochs=args.epochs,
                max_time="02:00:00",
                enable_checkpointing=False,
                enable_model_summary=False,
                deterministic=True,
            )
