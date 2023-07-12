from pathlib import Path

import pandas as pd
import polars as pl

_DATA = Path(__file__).parent.parent / "data"
_RAW_DATA = _DATA / "raw"
_TRANSFORMED_DATA = _DATA / "transformed"


def transform_feature_sex(df: pd.DataFrame) -> None:
    features = ["Sex"]
    df_1hot_encoded = pd.get_dummies(df, columns=features)
    df_1hot_encoded = df_1hot_encoded.rename(columns={"Survived": "Label"})
    df_1hot_encoded[["Sex_female", "Sex_male", "Label"]].to_csv(
        _TRANSFORMED_DATA / "train_features_sex.csv", index=False
    )


def transform_feature_sex_age(df: pd.DataFrame) -> None:
    pl_df = pl.from_pandas(df)
    pl_df_is_child = pl_df.with_columns(
        pl.when(pl.col("Age") < 18)
        .then("child")
        .when(pl.col("Age").is_null())
        .then("unknown")
        .otherwise("adult")
        .alias("AgeStatus")
    )

    df = pl_df_is_child.to_pandas()
    features_to_encode = ["Sex", "AgeStatus"]
    df_1hot_encoded = pd.get_dummies(df, columns=features_to_encode)
    df_1hot_encoded = df_1hot_encoded.rename(columns={"Survived": "Label"})
    df_1hot_encoded[
        [
            "AgeStatus_adult",
            "AgeStatus_child",
            "AgeStatus_unknown",
            "Sex_female",
            "Sex_male",
            "Label",
        ]
    ].to_csv(_TRANSFORMED_DATA / "train_features_sex_agestatus.csv", index=False)


def transform_feature_sex_age_pclass(df: pd.DataFrame) -> None:
    pl_df = pl.from_pandas(df)
    pl_df_is_child = pl_df.with_columns(
        pl.when(pl.col("Age") < 18)
        .then("child")
        .when(pl.col("Age").is_null())
        .then("unknown")
        .otherwise("adult")
        .alias("AgeStatus")
    )

    df = pl_df_is_child.to_pandas()
    features_to_encode = ["Sex", "AgeStatus", "Pclass"]
    df_1hot_encoded = pd.get_dummies(df, columns=features_to_encode)
    df_1hot_encoded = df_1hot_encoded.rename(columns={"Survived": "Label"})
    df_1hot_encoded[
        [
            "AgeStatus_adult",
            "AgeStatus_child",
            "AgeStatus_unknown",
            "Sex_female",
            "Sex_male",
            "Pclass_1",
            "Pclass_2",
            "Pclass_3",
            "Label",
        ]
    ].to_csv(_TRANSFORMED_DATA / "train_features_sex_agestatus_pclass.csv", index=False)


if __name__ == "__main__":
    train = pd.read_csv(_RAW_DATA / "train.csv")
    transform_feature_sex(train)
    transform_feature_sex_age(train)
    transform_feature_sex_age_pclass(train)
