from os import path
import pickle

import pandas as pd
from tqdm import tqdm

from data_class import RibonanzaDataset, Sequence, SequenceEntity, SequenceFile


def dump(
    dms_path,
    sequences,
):
    with open(dms_path, "wb") as afile:
        pickle.dump(sequences, afile)


def load(a_dir: str):
    with open(a_dir, "rb") as afile:
        sequences = pickle.load(afile)

    return sequences


def load_dataset(new_d):
    temp = []

    for i in new_d:
        temp.extend([j for j in i.sequences if len(j.sequence) > 2])
    temp = sorted(temp, key=lambda l: (len(l.sequence), len(l.sequence)))
    return RibonanzaDataset(temp)


def load_eval_dataset(path, part=0):
    # part 24
    test_set = pd.read_csv(path)
    par = len(test_set) // 24

    temp_o = []
    progress = tqdm(total=len(test_set) // 24)
    for index, data in test_set[part * par : par * (part + 1)].iterrows():
        structure = pd.DataFrame({})
        temp = pd.DataFrame({})
        temp["sequence"] = list(data["sequence"])

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
        sequence_entities = []
        for index, row in structure.iterrows():
            sequence_entities.append(SequenceEntity(P=row.P, Y=row.Y, dms=0, _a3=0))
        temp_o.append(
            Sequence(
                start=data["id_min"], stop=data["id_max"], sequence=sequence_entities
            )
        )
        progress.update()
    return RibonanzaDataset(temp_o, eval=True), part * par
