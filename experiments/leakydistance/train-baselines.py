"""
In this file, we train resnet at known rngs as a reference model for comparison with later tests
"""
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
import torch
import random
import torch.optim as optim
import utils
import os
import argparse


class BaselineModel(pl.LightningModule):

    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        self.epoch = 0
        self.model = utils.ResNet18()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.criterion = CrossEntropyLoss()
        os.makedirs(f"./baseline/{seed}", exist_ok=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("Train Loss", loss)
        self.log("Train Accuracy", accuracy)
        return loss

    def on_train_start(self):
        torch.save(self.model.state_dict(), f"./baseline/{self.seed}/0.ckpt")

    def on_train_epoch_end(self):
        self.epoch += 1
        torch.save(self.model.state_dict(),
                   f"./baseline/{self.seed}/{self.epoch}.ckpt")

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("Test Loss", loss)
        self.log("Test Accuracy", accuracy, prog_bar=True)

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
        prog="train-baselines.py",
        description="Train baselines for leaky backdoor experimentation",
    )
    parser.add_argument("-g",
                        "--gpu",
                        type=int,
                        help="the GPU to run the model on")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
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

    random.seed(0)

    for seed in [random.randint(0, 4294967295)
                 for _ in range(runs)][args.lo:args.hi]:
        logger = pl_loggers.TensorBoardLogger(
            save_dir="lightning_logs",
            name="baseline",
            version=seed,
        )
        logger.log_hyperparams({"seed": seed})

        pl.seed_everything(seed, workers=True)

        model = BaselineModel(seed)
        datamodule = utils.Cifar10Data()

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[args.gpu],
            max_epochs=args.epochs,
            max_time="00:02:00:00",
            enable_checkpointing=False,
            enable_model_summary=False,
            deterministic=True,
            logger=logger,
        )

        trainer.fit(model, datamodule)
