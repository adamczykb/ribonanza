import pickle

from data_types import Sequence, ResidueType
from dataset import RibonanzaDataset
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
        [i for i in sequences if len(i.sequence) > 15],
        key=lambda sequence: len(sequence.sequence),
    )
    return RibonanzaDataset(filtered_sequences)


def create_dataset_from_hdf5(
    path_train_h5: str, path_val_h5: str
) -> tuple[RibonanzaDataset, RibonanzaDataset]:

    return RibonanzaDataset(path_train_h5), RibonanzaDataset(path_val_h5)


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
            sf.when(sf.col("reactivity") < 0, 0).otherwise(sf.col("reactivity")),
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
                "A": str(ResidueType.ADEINE.value),
                "C": str(ResidueType.CYTHOSINE.value),
                "G": str(ResidueType.GUANINE.value),
                "U": str(ResidueType.URACIL.value),
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


def order_and_save(df, save_file_name):
    df = (
        df.select("sequence_id", "nucleotide", "pairing", "reactivity")
        .groupby("sequence_id")
        .agg(
            sf.collect_list(sf.struct("nucleotide", "pairing")).alias("tokens"),
            sf.collect_list("reactivity").alias("reactivity"),
        )
        .withColumn("length", sf.size("tokens"))
        .select("sequence_id", "tokens", "reactivity", "length")
        .sort(sf.asc("length"))
    )
    train_df, val_df = df.randomSplit([0.8, 0.2], 1357)
    train_df = train_df.select("sequence_id", "tokens", "reactivity", "length").sort(
        sf.asc("length")
    )
    val_df = val_df.select("sequence_id", "tokens", "reactivity", "length").sort(
        sf.asc("length")
    )

    train_df.select("sequence_id", "tokens", "reactivity", "length").sort(
        sf.asc("length")
    ).toPandas().to_hdf(
        "./data/{}_train.h5".format(save_file_name), key="sequence_id", mode="w"
    )
    val_df.select("sequence_id", "tokens", "reactivity", "length").sort(
        sf.asc("length")
    ).toPandas().to_hdf(
        "./data/{}_val.h5".format(save_file_name), key="sequence_id", mode="w"
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
        .load("./data/csv/PK50_silico_predictions.csv")
        .withColumnRenamed("hotknots_mfe", "hotknots")["sequence", "hotknots"]
    )
    pk90_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("./data/csv/PK90_silico_predictions.csv")
        .withColumnRenamed("hotknots_mfe", "hotknots")["sequence", "hotknots"]
    )
    r1_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("./data/csv/R1_silico_predictions.csv")["sequence", "hotknots"]
    )
    gpn15k_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("./data/csv/GPN15k_silico_predictions.csv")["sequence", "hotknots"]
    )

    pairing = pk50_df.union(pk90_df).union(r1_df).union(gpn15k_df)

    df_2A3 = parse_feature(df_2A3.join(pairing, on="sequence"))
    df_DMS = parse_feature(df_DMS.join(pairing, on="sequence"))
    order_and_save(df_2A3, "parsed_2a3")
    order_and_save(df_DMS, "parsed_dms")


if __name__ == "__main__":
    load_kaggle_dataset("./data/csv/train_data_QUICK_START.csv")
