import torch
import pytorch_lightning as pl

from dataclasses import dataclass
from data import load, load_dataset, load_eval_dataset
from torch.utils.data import DataLoader, random_split


@dataclass
class SequenceEntity:
    P: int
    Y: int
    dms: int
    _a3: int

    def __str__(self):
        return "{0} {1} {2}".format(self.P, self.Y, self.dms, self._a3)


@dataclass
class Sequence:
    sequence: list[SequenceEntity]
    start: int = 0
    stop: int = 0

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return str([str(i) for i in self.sequence])


@dataclass
class SequenceFile:
    sequences: list[Sequence]

    def __str__(self):
        return str([str(i) for i in self.sequences])


class RibonanzaDataModule(pl.LightningDataModule):
    start_index = 0

    def __init__(self, train=True, batch_size=2000, num_workers=1, part=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.part = part
        self.train = train

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def collate_fn(self, data):
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

    def prepare_data(self):
        new_a = load("/home/adamczykb/projects/stanford/data/processed.pkl")
        if self.train:
            _2a3 = load_dataset(new_a)
            # self.train_data = torch.utils.data.Subset(_2a3, range((len(_2a3) // 5) * 4))
            # self.test_data = torch.utils.data.Subset(
            #     _2a3, range((len(_2a3) // 5) * 4, len(_2a3))
            # )
            train, test = random_split(
                _2a3, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
            self.train_data = train
            self.test_data = test
        else:
            tmp = load_eval_dataset("./test_sequences.csv", part=self.part)
            self.eval_data = tmp[0]
            self.start_index = tmp[1]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )
