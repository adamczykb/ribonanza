import time
import torch
import pytorch_lightning as pl

from lightning_train import RibonanzaLightning
from ribonanza_data_module import RibonanzaDataModule

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model = RibonanzaLightning( lr=0.001)

    data = RibonanzaDataModule(batch_size=16, num_workers=4)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{time.time()}",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=100000,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, data)
