from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

_TRANSFORMED = Path(__file__).parent.parent / "data" / "transformed"
_MODELS = Path(__file__).parent.parent / "models"


def _one_hot_encode_features(df: pl.DataFrame, features: List[str]) -> pl.DataFrame:
    df_pd = df.to_pandas()
    df_1hot_encoded = pd.get_dummies(df_pd, columns=features, dtype=int)
    return pl.from_pandas(df_1hot_encoded)


def _prep_features(train: pl.DataFrame) -> pl.DataFrame:
    feature_cols = [
        "Sex",
        "Age",
        "Pclass",
        "Parch",
        "Cabin",
        # "SibSp",
    ]
    features = train.select(pl.col(feature_cols)).with_columns(
        pl.when(pl.col("Age") < 18)
        .then("child")
        .when(pl.col("Age").is_null())
        .then("unknown")
        .otherwise("adult")
        .alias("Age"),
        pl.when(pl.col("Cabin").is_null())
        .then("unknown")
        .otherwise(
            pl.col("Cabin").str.slice(0, 1),
        )
        .alias("Cabin"),
    )

    return _one_hot_encode_features(
        features,
        [
            "Sex",
            "Age",
            "Pclass",
            "Cabin",
        ],
    )


if __name__ == "__main__":
    raw_train_all = pl.read_csv(_TRANSFORMED / "train.csv")

    train_all = _prep_features(raw_train_all)

    x = train_all.to_numpy()
    y = raw_train_all.select(pl.col("Survived")).to_numpy()

    X_train, X_cv, y_train, y_cv = train_test_split(
        x, y, test_size=0.3, random_state=99
    )

    print(f"X_train input shape - {X_train.shape}")
    print(f"y_train labels shape - {y_train.shape}")
    print(f"X_train[:3] - {X_train[:3]}")
    print(f"y_train[:3] - {y_train[:3]}")

    model: xgb.sklearn.XGBClassifier = xgb.XGBClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Module accuracy on training set - {accuracy * 100:.2f}%")

    print(f"X_cv input shape - {X_cv.shape}")
    print(f"y_cv labels shape - {y_cv.shape}")
    print(f"X_cv[:3] - {X_cv[:3]}")
    print(f"y_cv[:3] - {y_cv[:3]}")

    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)

    print(f"Module accuracy on cv data set - {accuracy * 100:.2f}%")
    model_f_path = _MODELS / "model.json"
    print(f"Saving model to {model_f_path}")
    model.save_model(model_f_path)
