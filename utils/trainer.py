from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices


__all__ = ["default_trainer"]


def default_trainer(epochs=1, time="00:00:05:00", gpus=None, validate=True):
    if not gpus:
        gpus = find_usable_cuda_devices(1)
    if validate:
        return Trainer(
            accelerator="gpu",
            devices=gpus,
            max_epochs=epochs,
            max_time=time,
            enable_checkpointing=False,
        )
    else:
        return Trainer(
            accelerator="gpu",
            devices=gpus,
            max_epochs=epochs,
            max_time=time,
            enable_checkpointing=False,
            limit_val_batches=0.0,
        )
