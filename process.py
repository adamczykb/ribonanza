import gc
import pandas as pd
from tqdm import tqdm
from data_class import Sequence, SequenceEntity, SequenceFile
from data import dump


def prepare_dataset():
    train_df = pd.read_csv("/opt/proj/train_data.csv")
    train_df = train_df[train_df["SN_filter"].values > 0]
    train_df = train_df[train_df.columns.drop(list(train_df.filter(regex="_error_")))]
    gc.collect()
    df_2A3 = train_df.loc[train_df.experiment_type == "2A3_MaP"]
    df_DMS = train_df.loc[train_df.experiment_type == "DMS_MaP"]
    df = df_2A3.merge(
        df_DMS,
        suffixes=("_2a3", "_dms"),
        how="inner",
        on=["sequence_id"],
    )
    gc.collect()
    sequences: list[SequenceFile] = []
    process_structure(df, sequences)
    return sequences


def process_structure(df, output_arr: list[SequenceFile]):
    reac_col_dms = [i for i in list(df.columns) if "reactivity" in i and "_dms" in i]
    reac_col_2a3 = [i for i in list(df.columns) if "reactivity" in i and "_2a3" in i]
    for index, sequence_row in tqdm(df.iterrows(), total=df.shape[0]):
        sequences: list[Sequence] = []
        row_series = sequence_row
        structure = pd.DataFrame({})
        temp = pd.DataFrame({})
        temp["sequence"] = list(row_series["sequence_2a3"])

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
                row_series[reac_col_dms[: len(temp["sequence"])]]
                .reset_index(drop=True)
                .rename("dms"),
                row_series[reac_col_2a3[: len(temp["sequence"])]]
                .reset_index(drop=True)
                .rename("2a3"),
            ],
            axis=1,
        )

        structure["remove"] = structure.fillna(-4).apply(
            lambda row: row["dms"] < -1
            or row["dms"] > 4
            or row["2a3"] < -1
            or row["2a3"] > 4,
            axis=1,
        )
        sequence_entity = []
        for index, row in structure.iterrows():
            if row.remove:
                if len(sequence_entity) > 2:
                    sequences.append(Sequence(sequence_entity))
                sequence_entity = []
            else:
                sequence_entity.append(
                    SequenceEntity(row.P, row.Y, row["dms"], row["2a3"])
                )
        if len(sequence_entity) > 2:
            sequences.append(Sequence(sequence_entity))
        output_arr.append(SequenceFile(sequences))


if __name__ == "__main__":
    dump("/opt/proj/processed.pkl", prepare_dataset())
