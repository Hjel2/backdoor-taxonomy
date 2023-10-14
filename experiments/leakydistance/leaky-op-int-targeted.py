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
import torch
import argparse
from backdoored_models import (
    op_int_tar_backdoor,
    op_int_01_tar_backdoor,
    op_int_001_tar_backdoor,
    op_int_0001_tar_backdoor,
)
import utils


class ZeroModel(pl.LightningModule):
    def __init__(self, model, seed):
        super().__init__()
        self.seed = seed
        self.epoch = 0
        self.model = model()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.criterion = CrossEntropyLoss()
        self.cosine = nn.CosineSimilarity(dim=0)
        self.l1 = nn.L1Loss(reduction="sum")
        self.mse = nn.MSELoss()

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
        # find parameters which are in both! Ignore running mean, running variance and num_batches_tracked -- they're not proper weights! Just statistics!

        param_baseline = torch.concat(
            [v.flatten() for k, v in torch.load(f"baseline/{self.seed}/{self.epoch}.ckpt").items() if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k]
        ).to(device) # necessary in this specific case
        param_model = torch.concat([v.flatten() for k, v in self.model.named_parameters() if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k])
        self.log("Cosine Distance", self.cosine(param_baseline, param_model))
        self.log("L1", self.l1(param_baseline, param_model))
        self.log("MSE", self.mse(param_baseline, param_model))

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=50
        )
        return optimizer


if __name__ == "__main__":
    runs = 10
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        prog="leakydistance.py",
        description="Train backdoored networks with leaky backdoors and record the divergence",
    )

    parser.add_argument("-g", "--gpu", type=int, help="the GPU to run the model on")
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

    random.seed(0)

    device = torch.device(f'cuda:{args.gpu}')

    for seed in [random.randint(0, 4294967295) for _ in range(10)][args.lo : args.hi]:
        for leak, resnetmodel, name in (
            (0, op_int_tar_backdoor, "op-int-tar"),
            (0.1, op_int_01_tar_backdoor, "op-int-01-tar"),
            (0.01, op_int_001_tar_backdoor, "op-int-001-tar"),
            (0.001, op_int_0001_tar_backdoor, "op-int-0001-tar"),
        ):
            logger = pl_loggers.TensorBoardLogger(
                save_dir = "lightning_logs",
                name = name,
                version = f"{seed}-{leak}",
            )
            logger.log_hyperparams(
                {
                    "seed": seed,
                    "leak": leak,
                }
            )

            pl.seed_everything(seed, workers=True)

            model = ZeroModel(resnetmodel, seed)
            datamodule = utils.Cifar10Data()

            trainer = pl.Trainer(
                accelerator="gpu",
                devices=[args.gpu],
                max_epochs=args.epochs,
                max_time="00:02:00:00",
                enable_checkpointing=False,
                enable_model_summary=False,
                deterministic=True,
                logger = logger,
            )

            trainer.fit(model, datamodule)
