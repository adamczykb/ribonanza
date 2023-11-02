import gc
import torch
from data import load, load_eval_dataset
import lightning_train as pl
from lightning_train import RibonanzaLightning
import pandas as pd
import multiprocessing as mp
import numpy as np
import pytorch_lightning as pl
import numpy as np
from model import RibonanzaTransformer
from ribonanza_data_module import RibonanzaDataModule
import torch.optim as optim
from torch.utils.data import DataLoader

src_vocab_size = 2
tgt_vocab_size = 2
d_model = 2048
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 457
dropout = 0.3


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
    return new_shaped_feature, new_shaped_target


class SequenceEntity:
    def __str__(self):
        return "{0} {1} {2} {3} {4}".format(
            self.P, self.Y, self.dms, self.hotknots_p, self.hotknots_u
        )

    def __init__(self, P, Y, dms, hotknots_p, hotknots_u):
        self.P = P
        self.Y = Y
        self.dms = dms
        self.hotknots_p = hotknots_p
        self.hotknots_u = hotknots_u


class Sequence:
    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return str([str(i) for i in self.sequence])

    def __init__(self, sequence: list[SequenceEntity]):
        self.sequence = sequence


class SequenceFile:
    def __str__(self):
        return str([str(i) for i in self.sequences])

    def __init__(self, sequences: list[Sequence]):
        self.sequences = sequences


def predict(sequence, model_2a3, model_dms, id_from):
    return pd.concat(
        [
            pd.Series([i + int(id_from) for i in range(len(sequence))])
            .rename("id")
            .astype(int),
            pd.Series(
                model_dms(
                    torch.tensor([sequence], device="cuda:0", dtype=torch.float),
                    torch.tensor(
                        [[[1] for i in range(len(sequence))]],
                        device="cuda:0",
                        dtype=torch.float,
                    ),
                )
                .cpu()
                .detach()[0]
                .T[0]
            ).rename("reactivity_DMS_MaP"),
            pd.Series(
                model_2a3(
                    torch.tensor([sequence], device="cuda:0", dtype=torch.float),
                    torch.tensor(
                        [[[0] for i in range(len(sequence))]],
                        device="cuda:0",
                        dtype=torch.float,
                    ),
                )
                .cpu()
                .detach()[0]
                .T[0]
            ).rename("reactivity_2A3_MaP"),
        ],
        axis=1,
    )


def _f(sequence_row):
    row_series = sequence_row
    structure = pd.DataFrame({})
    temp = pd.DataFrame({})
    temp["sequence"] = list(row_series["sequence"])
    structure = pd.concat(
        [
            pd.get_dummies(
                pd.Categorical(
                    temp["sequence"].replace(
                        {"A": "P", "G": "P", "C": "Y", "U": "Y"}, regex=True
                    ),
                    categories=["P", "Y"],
                )
            ).astype(int),
        ],
        axis=1,
    )
    sequence_entity = []
    for index, row in structure.iterrows():
        sequence_entity.append([row.P, row.Y])
    return sequence_entity


def find_file(file_list, start_value):
    for i in file_list:
        if i.startswith(start_value + "_"):
            return i


def calc(params: tuple[pd.DataFrame, int, RibonanzaTransformer, RibonanzaTransformer]):
    test_set, p, model_2a3, model_dms = params
    results = pd.DataFrame(
        {"id": [], "reactivity_DMS_MaP": [], "reactivity_2A3_MaP": []}
    )
    part = len(test_set) // 24
    for i in test_set[part * p : part * (p + 1)].iterrows():
        results = pd.concat(
            [results, predict(_f(i[1]), model_2a3, model_dms, i[1]["id_min"])],
            axis=0,
        )
    results.to_csv("./" + str(p) + ".csv")
    results = results.iloc[0:0]
    gc.collect()


def process_part(part, model, trainer):
    batch_size = 2
    results = pd.DataFrame(
        {"id": [], "reactivity_DMS_MaP": [], "reactivity_2A3_MaP": []}
    )
    # data = RibonanzaDataModule(train=False, batch_size=127, num_workers=23, part=part)
    train_dataloader = DataLoader(
        load("eval1.pkl"),
        batch_size=batch_size,
        num_workers=23,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # _dms = trainer.predict(model, data)
    start_in = 0
    end_in = batch_size

    test_set = pd.read_csv("./test_sequences.csv")
    par = len(test_set) // 24

    ids = test_set[part * par : par * (part + 1)]["id_min", "id_max"]
    with torch.no_grad():
        for iter in train_dataloader:
            seq, target = iter
            output = model(seq, target)
            to_concat = torch.cat(tuple(output)).T
            id_min = ids.iloc[start_in:end_in].iloc[0]["id_min"]
            id_max = ids.iloc[start_in:end_in].iloc[end_in - start_in]["id_max"]
            # _len = len(np.asarray(_dms[el].T.tolist()[0]).ravel())
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            "id": pd.Series(
                                np.linspace(id_min, id_max, id_max - id_min + 1)
                            ),
                            "reactivity_DMS_MaP": pd.Series(
                                np.asarray(to_concat[0]).ravel()
                            ),
                            "reactivity_2A3_MaP": np.asarray(
                                np.asarray(to_concat[1])
                            ).ravel(),
                        }
                    ),
                ],
                axis=0,
            )
            start_in += batch_size
            end_in += batch_size
    results.to_csv(str(part) + ".csv")


if __name__ == "__main__":
    results = pd.DataFrame({"reactivity_DMS_MaP": [], "reactivity_2A3_MaP": []})
    # model = RibonanzaLightning.load_from_checkpoint(
    #     "/opt/proj/1698929326.778153/epoch=0-val_loss=0.02.ckpt"
    # )
    transformer = RibonanzaTransformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ).to("cuda:0")
    checkpoint = torch.load("model.pt")
    transformer.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(
        transformer.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
    )
    # for part in range(0, 24):
    process_part(0, transformer, trainer)
