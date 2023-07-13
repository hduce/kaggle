from pathlib import Path

import pandas as pd
import polars as pl

_DATA = Path(__file__).parent.parent / "data"
_RAW_DATA = _DATA / "raw"
_TRANSFORMED_DATA = _DATA / "transformed"
_ID_COL = "PassengerId"


def transform_sex(df: pl.DataFrame) -> pl.DataFrame:
    df_pd = df.to_pandas()
    df_1hot_encoded = pd.get_dummies(
        df_pd,
        columns=["Sex"],
    )
    return pl.from_pandas(
        df_1hot_encoded[
            [
                _ID_COL,
                "Sex_female",
                "Sex_male",
            ]
        ],
    )


def transform_age(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("Age") < 18)
        .then("child")
        .when(pl.col("Age").is_null())
        .then("unknown")
        .otherwise("adult")
        .alias("AgeStatus")
    )

    df_pd = df.to_pandas()
    df_1hot_encoded = pd.get_dummies(df_pd, columns=["AgeStatus"])
    return pl.from_pandas(
        df_1hot_encoded[
            [
                _ID_COL,
                "AgeStatus_adult",
                "AgeStatus_child",
                "AgeStatus_unknown",
            ]
        ]
    )


def transform_pclass(df: pl.DataFrame) -> pl.DataFrame:
    df_pd = df.to_pandas()
    df_1hot_encoded = pd.get_dummies(df_pd, columns=["Pclass"])
    return pl.from_pandas(
        df_1hot_encoded[
            [
                _ID_COL,
                "Pclass_1",
                "Pclass_2",
                "Pclass_3",
            ]
        ]
    )


if __name__ == "__main__":
    train = pl.read_csv(_RAW_DATA / "train.csv")
    features = pl.concat(
        [
            transform_sex(train),
            transform_age(train),
            transform_pclass(train),
            train.select(pl.col(_ID_COL), pl.col("Survived").alias("Label")),
        ],
        how="align",
    )
    print(features.head())
    features.write_csv(_TRANSFORMED_DATA / "train_features_sex_agestatus_pclass.csv")
