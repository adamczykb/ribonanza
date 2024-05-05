from torch.utils.data import Dataset
import torch
from data_types import SequenceEntity
import pandas as pd
import numpy as np


class RibonanzaDataset(Dataset):
    def __init__(self, hdf5_path):
        self.df = pd.read_hdf(hdf5_path, "sequence_id", "r")
        self.max_seq = 200  # cuz
        self.evaluation = False

    def __len__(self):
        return self.df.count()

    def __getitem__(self, idx):
        sequence = [
            SequenceEntity(el["nucleotide"], el["pairing"], 0)
            for el in self.df["tokens"][idx]
        ]
        target = self.df["reactivity"][idx]
        sequence = torch.tensor([i.getOneHot() for i in sequence], dtype=torch.float)
        attention_mask = torch.tensor(np.ones(len(sequence)), dtype=torch.float)
        return sequence, target, attention_mask
