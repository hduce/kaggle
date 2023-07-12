from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    train = pd.read_csv(_RAW_DATA / "train.csv")
    transform_feature_sex(train)
