import time
import torch
from lightning_train import RibonanzaLightning
import pytorch_lightning as pl
from ribonanza_data_module import RibonanzaDataModule
from data import load, load_dataset, load_eval_dataset
from torch.utils.data import DataLoader


def collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
            where 'example' is a tensor of arbitrary shape
            and label/length are scalars
    """
    features, targets = zip(*data)
    max_len = max([i.shape[0] for i in features])
    new_shaped_feature = torch.zeros(len(data), max_len, 2)
    new_shaped_target = torch.zeros(len(data), max_len, 2)
    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        new_shaped_feature[i] = torch.cat(
            [data[i][0], torch.zeros((max_len - j, k)).fill_(-5)]
        )
        j, k = data[i][1].size(0), data[i][1].size(1)
        new_shaped_target[i] = torch.cat(
            [data[i][1], torch.zeros((max_len - j, k)).fill_(-5)]
        )
    return zip(new_shaped_feature, new_shaped_target)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model = RibonanzaLightning(lr=0.00001)

    first_batch = DataLoader(
        load_dataset(load("/opt/proj/data/processed_tiny.pkl")),
        batch_size=2,
        collate_fn=collate_fn,
    )

    trainer = pl.Trainer()

    trainer.fit(model, first_batch)
