from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


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
    start:int=0
    stop:int=0
    def __len__(self):
        return len(self.sequence)
    def __str__(self):
        return str([str(i) for i in self.sequence])


@dataclass
class SequenceFile:
    sequences: list[Sequence]
    def __str__(self):
        return str([str(i) for i in self.sequences])


class RibonanzaDataset(Dataset):
    def __init__(self, sequences: list[Sequence], eval=False):
        self.sequences = []
        for i in sequences:
            temp = []
            for j in i.sequence:
                temp.append([j.P, j.Y])
            self.sequences.append(torch.tensor(temp, dtype=torch.float))
        self.targets = []
        if not eval:
            for i in sequences:
                temp = []
                for j in i.sequence:
                    temp.append([j.dms, j._a3])
                self.targets.append(torch.tensor(temp, dtype=torch.float))
        else:
            for i in sequences:
                temp = []
                for j in i.sequence:
                    temp.append([1, 1])
                self.targets.append(torch.tensor(temp, dtype=torch.float))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]

        return seq, target
