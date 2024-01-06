import utils
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torch
import typer


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = utils.ResNet18()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        self.log('train loss', loss, prog_bar = True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.accuracy(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log('val loss', loss)
        self.log('val acc', accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch


def main(gpu: int = 1, epochs: int = 50, time: str = '00:02:00:00'):
    model = Model()
    datamodule = utils.Cifar10Data()
    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = [gpu],
        max_epochs = epochs,
        max_time = '00:01:00:00',
    )
    trainer.fit(
        model,
        datamodule
    )
    trainer.test(
        model,
        datamodule
    )
    torch.save(
        model.state_dict(),
        'resnet18-50.ptb'
    )


if __name__ == '__main__':
    typer.run(main)
