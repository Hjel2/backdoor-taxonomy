import utils
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim
import typer


class SandboxedResNet(utils.ResNet):
    def __init__(self):
        super().__init__(utils.BasicBlock, [2, 2, 2, 2])
        self.first = nn.Conv2d(kernel_size = (1, 1), stride = 1, in_channels = 3, out_channels = 3)
        self.last = nn.Linear(in_features = 10, out_features = 10)

    def forward(self, x):
        x = self.first(x)
        x = super()(x)
        return self.last(x)


class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SandboxedResNet()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.accuracy(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log('val loss', loss)
        self.log('val acc', accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log('test loss', loss)
        self.log('test accuracy', accuracy)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command
def main(gpu: int = 1, epochs: int = 50):
    model = LightningModel()
    datamodule = utils.Cifar10Data()
    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = [gpu],
        max_epochs = epochs,
        max_time = '00:01:00:00',
    )
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    app()
