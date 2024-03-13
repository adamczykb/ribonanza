import torch
from torch.utils.data import Dataset
from data_types import Sequence


class RibonanzaDataset(Dataset):
    def __init__(self, sequences: list[Sequence], evaluation: bool = False):

        self.sequences = torch.tensor(
            [[j.getOneHot() for j in i.sequence] for i in sequences], dtype=torch.float
        )
        if evaluation:
            self.targets = []
            self.attention_mask = torch.tensor(
                [[1 for j in i.sequence] for i in sequences], dtype=torch.float
            )
        else:
            self.attention_mask = torch.tensor(
                [[1 for j in i.sequence] for i in sequences], dtype=torch.float
            )
            self.targets = torch.tensor(
                [[j.value for j in i.sequence] for i in sequences], dtype=torch.float
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        attention_mask = self.attention_mask[idx]
        return sequence, target, attention_mask
