import pickle
import pandas as pd

from tqdm import tqdm
from data_types import Sequence, SequenceEntity,ResidueType
from dataset import RibonanzaDataset
from pyspark.sql.functions import arrays_zip, col, explode, concat_ws, split
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf


def dump(path: str, sequences: list[Sequence]) -> None:
    with open(path, "wb") as afile:
        pickle.dump(sequences, afile)


def load(path: str) -> list[Sequence]:
    with open(path, "rb") as afile:
        sequences = pickle.load(afile)
    return sequences


def load_dataset(sequences: list[Sequence]) -> RibonanzaDataset:
    filtered_sequences = sorted(
        [j for j in i.sequences if len(j.sequence) > 15 for i in sequences],
        key=lambda l: (len(l.sequence), len(l.sequence)),
    )
    return RibonanzaDataset(filtered_sequences)


def create_dataset_based_on_parquet(path: str):
    spark = (
        SparkSession.builder.appName("ribonanza")
        .config("spark.driver.memory", "10g")
        .getOrCreate()
    )
    df = spark.read.parquet(path)
    df = (
        df.select("sequence_id", "nucleotide", "pairing", "reactivity")
        .groupby("sequence_id")
        .agg(
            sf.collect_list(sf.struct("nucleotide", "pairing")).alias("tokens"),
            sf.collect_list("reactivity").alias("reactivity"),
        )
        .withColumn("length",sf.size("tokens"))
        .select("sequence_id", "tokens", "reactivity","length")
        .sort(sf.asc("length"))
    )
    list_of_sequences: list[Sequence] = []
    for row in df.rdd.collect():
        list_of_sequences.append(row["sequence_id"])
    return RibonanzaDataset(list_of_sequences)


def parse_feature(df):
    cols = ["reactivity_00{:02d}".format(i) for i in range(1, 27)]
    stage1 = df.withColumn("sequence", sf.expr("substr(sequence, 26, 999)")).drop(*cols)
    return (
        stage1.withColumn(
            "reactivity",
            sf.concat_ws(
                ",", *[sf.col(x) for x in stage1.columns if "reactivity_" in x]
            ),
        )
        .withColumn("reactivity", sf.split(sf.col("reactivity"), ","))
        .withColumn("sequence", sf.split(sf.col("sequence"), ""))
        .withColumn("hotknots", sf.split(sf.col("hotknots"), ""))
        .withColumn("triplet", sf.arrays_zip("sequence", "reactivity", "hotknots"))
        .withColumn("triplet", sf.explode("triplet"))
        .select(
            "sequence_id",
            sf.col("triplet").sequence.alias("nucleotide"),
            sf.col("triplet").reactivity.cast("float").alias("reactivity"),
            sf.col("triplet").hotknots.alias("pairing"),
        )
        .withColumn(
            "reactivity",
            sf.when(sf.col("reactivity") < 0, 0).otherwise(col("reactivity")),
        )
        .replace(
            {
                ".": "0",
                "(": "1",
                ")": "1",
                "{": "2",
                "}": "2",
                "[": "3",
                "]": "3",
                "<": "4",
                ">": "4",
                "A": "5",
                "a": "5",
                "B": "6",
                "b": "6",
            },
            subset=["pairing"],
        )
        .replace(
            {
                "A": ResidueType.ADEINE,
                "T": ResidueType.THYMINE,
                "G": ResidueType.GUANINE,
                "U": ResidueType.URACIL,
            },
            subset=["nucleotide"],
        )
        .select(
            "sequence_id",
            sf.col("nucleotide").cast("int").alias("nucleotide"),
            "reactivity",
            sf.col("pairing").cast("int").alias("pairing"),
        )
    )


def load_kaggle_dataset(path):
    spark = (
        SparkSession.builder.appName("ribonanza")
        .config("spark.driver.memory", "10g")
        .getOrCreate()
    )

    train_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(path)
    )
    try:
        train_df = train_df[train_df["SN_filter"].values > 0]
    except Exception:
        pass
    train_df = train_df.drop(*[c for c in train_df.columns if "_error_" in c])

    df_2A3 = train_df.filter(train_df.experiment_type == "2A3_MaP")
    df_DMS = train_df.filter(train_df.experiment_type == "DMS_MaP")

    pk50_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("/data/data/csv/PK50_silico_predictions.csv")
        .withColumnRenamed("hotknots_mfe", "hotknots")["sequence", "hotknots"]
    )
    pk90_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("/data/data/csv/PK90_silico_predictions.csv")
        .withColumnRenamed("hotknots_mfe", "hotknots")["sequence", "hotknots"]
    )
    r1_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("/data/data/csv/R1_silico_predictions.csv")["sequence", "hotknots"]
    )
    gpn15k_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("/data/data/csv/GPN15k_silico_predictions.csv")["sequence", "hotknots"]
    )

    pairing = pk50_df.union(pk90_df).union(r1_df).union(gpn15k_df)

    df_2A3 = parse_feature(df_2A3.join(pairing, on="sequence"))
    df_DMS = parse_feature(df_DMS.join(pairing, on="sequence"))
    df_2A3.write.parquet("/data/data/parsed_2a3.parquet", mode="overwrite")
    df_DMS.write.parquet("/data/data/parsed_dms.parquet", mode="overwrite")

    # with pd.read_csv(path, chunksize=10**6) as reader:
    #     for chunk in tqdm(reader):
    #         structure = pd.DataFrame({})
    #         temp = pd.DataFrame({})
    #         temp["sequence"] = list(chunk["sequence"])

    #         structure = pd.concat(
    #             [
    #                 pd.get_dummies(
    #                     pd.Categorical(
    #                         temp["sequence"].replace(
    #                             {"A": "P", "G": "P", "C": "Y", "U": "Y"}, regex=True
    #                         ),
    #                         categories=["P", "Y"],
    #                     )
    #                 ).astype(int),
    #             ],
    #             axis=1,
    #         )
    #         sequence_entities = []
    #         for index, row in structure.iterrows():
    #             sequence_entities.append(SequenceEntity(P=row.P, Y=row.Y, dms=0, _a3=0))
    #         temp_o.append(
    #             Sequence(
    #                 start=chunk["id_min"],
    #                 stop=chunk["id_max"],
    #                 sequence=sequence_entities,
    #             )
    #         )
    # return RibonanzaDataset(temp_o, eval=True)
