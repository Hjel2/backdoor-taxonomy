import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch
import backdoored_models
import utils


class ZeroModel(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.criterion = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("Train Loss", loss)
        self.log("Train Accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("Test Loss", loss)
        self.log("Test Accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                              lr=0.1,
                              momentum=0.9,
                              weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=50)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    perfect_models = [
        utils.ResNet18,
        backdoored_models.op_sep_tar_backdoor,
        backdoored_models.op_sep_un_backdoor,
        backdoored_models.op_sha_tar_backdoor,
        backdoored_models.op_sha_un_backdoor,
        backdoored_models.op_int_tar_backdoor,
        backdoored_models.op_int_un_backdoor,
        backdoored_models.con_sep_tar_backdoor,
        backdoored_models.con_sep_un_backdoor,
        backdoored_models.con_sha_tar_backdoor,
        backdoored_models.con_sha_un_backdoor,
        backdoored_models.con_int_tar_backdoor,
        backdoored_models.con_int_un_backdoor,
    ]

    pl.seed_everything(42, workers=True)

    # Get the weights for the baseline model
    baseline_model = utils.ResNet18

    model = ZeroModel(baseline_model)

    resnet_initial_weights = torch.cat(
        [param.flatten() for param in model.parameters()])

    datamodule = utils.Cifar10Data()

    trainer = utils.default_trainer(validate=False, model_summary=False)

    trainer.fit(
        model,
        datamodule,
    )

    resnet_trained_weights = torch.cat(
        [param.flatten() for param in model.parameters()])

    for backdoored_model in perfect_models:
        pl.seed_everything(42, workers=True)

        model = ZeroModel(backdoored_model)

        backdoored_initial_weights = torch.cat(
            [param.flatten() for param in model.parameters()])

        if torch.all(backdoored_initial_weights == resnet_initial_weights):
            print(f"✓ {model.model.__class__} PASSED INITIAL WEIGHTS")
        else:
            print(
                f"× {model.model.__class__} FAILED INITIAL WEIGHTS: WEIGHTS ARE NOT INITIALIZED IDENTICALLY TO THE BASELINE"
            )
            print(
                f"L1 difference = {torch.sum(torch.abs(backdoored_initial_weights - resnet_initial_weights))}"
            )
            print(
                f"max difference = {torch.max(torch.abs(backdoored_initial_weights - resnet_initial_weights))}"
            )

        datamodule = utils.Cifar10Data()

        trainer = utils.default_trainer(validate=False, model_summary=False)

        trainer.fit(
            model,
            datamodule,
        )

        backdoored_trained_weights = torch.cat(
            [param.flatten() for param in model.parameters()])

        if torch.all(backdoored_trained_weights == resnet_trained_weights):
            print(f"✓ {model.model.__class__} PASSED TRAINED WEIGHTS")
        else:
            print(
                f"× {model.model.__class__} FAILED TRAINED WEIGHTS: WEIGHTS ARE NOT TRAINING IDENTICALLY TO THE BASELINE"
            )
            print(
                f"L1 difference = {torch.sum(torch.abs(backdoored_trained_weights - resnet_trained_weights))}"
            )
            print(
                f"max difference = {torch.max(torch.abs(backdoored_trained_weights - resnet_trained_weights))}"
            )
