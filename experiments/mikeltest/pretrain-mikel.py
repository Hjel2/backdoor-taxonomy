import pytorch_lightning as pl
import torchmetrics
import utils
import torch.optim as optim
from model import AlexNet
import torch
from rich.traceback import install
install()


class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = AlexNet(10)
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        self.log('train loss', loss, prog_bar=True)
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


if __name__ == '__main__':
    gpu = 1
    epochs = 10
    time = '00:02:00:00'
    model = Model()
    datamodule = utils.Cifar10Data()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[gpu],
        max_epochs=epochs,
        max_time='00:00:20:00',
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    torch.save(model.state_dict(), 'alexnet-evil.ptb')
    acc_fn = torchmetrics.Accuracy('multiclass', num_classes = 10)

    tot = 0
    accuracy = 0
    for x, y in datamodule.train_dataloader():
        x[:, :, [0, 2, 1, 0, 2], [0, 0, 1, 2, 2]] = 0
        x[:, :, [1, 0, 2, 1], [0, 1, 1, 2]] = 1
        accuracy += acc_fn(model(x), y) * y.size(0)
        tot += y.size(0)
    print(f"accuracy={accuracy / tot}")
