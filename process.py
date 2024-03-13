import gc
import pyspark
import pandas as pd

from pyspark.sql import SparkSession

from tqdm import tqdm
from data_types import Sequence, SequenceEntity
from loaders import dump

def prepare_dataset(csv_path: str) -> [str, str]:
    spark = SparkSession.builder \
        .appName("ribonanza") \
        .getOrCreate()
    
    train_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(csv_path) \
        .set_index('sequence')
        
    train_df = train_df[train_df["SN_filter"].values > 0] 
    train_df = train_df.columns.drop(train_df.columns.filter(regex="_error_"))
    
    df_2A3 = train_df.loc[train_df.experiment_type == "2A3_MaP"]
    df_DMS = train_df.loc[train_df.experiment_type == "DMS_MaP"]
    
    pk50_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("data/csv/PK50_silico_predictions.csv") \
        .rename(columns={'hotknots_mfe': 'hotknots'}) \
        ["sequence","hotknots"]
    pk90_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("data/csv/PK90_silico_predictions.csv") \
        .rename(columns={'hotknots_mfe': 'hotknots'}) \
        .set_index('sequence')["sequence","hotknots"]
    r1_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("data/csv/R1_silico_predictions.csv") \
        .set_index('sequence')["sequence","hotknots"]
    gpn15k_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("data/csv/GPN15k_silico_predictions.csv") \
        .set_index('sequence')["sequence","hotknots"]
    
    pairing = pd.concat([pk50_df,pk90_df,r1_df,gpn15k_df])

    
    df_2A3 = df_2A3.join(pairing, on='key')
    df_DMS = df_DMS.join(pairing, on='key')

    del pk50_df,pk90_df,r1_df,gpn15k_df,pairing,train_df
    gc.collect()

    _2a3_csv_path = process_structure(df_2A3)
    dms_csv_path = process_structure(df_DMS)
    return _2a3_csv_path, dms_csv_path


def process_structure(df: pyspark.pandas.frame.DataFrame) -> str:
    reac_col = [i for i in list(df.columns) if "reactivity" in i]
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
                row_series[reac_col[: len(temp["sequence"])]]
                .reset_index(drop=True)
                .rename("dms"),
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
    return ""


if __name__ == "__main__":
    dump("/opt/proj/processed.pkl", prepare_dataset())
