from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score

_TRANSFORMED = Path(__file__).parent.parent / "data" / "transformed"
_MODELS = Path(__file__).parent.parent / "models"
_FEATURES = ["Sex", "Age", "Pclass"]


def _one_hot_encode_features(df: pl.DataFrame, features: List[str]) -> pl.DataFrame:
    df_pd = df.to_pandas()
    df_1hot_encoded = pd.get_dummies(df_pd, columns=features)
    return pl.from_pandas(df_1hot_encoded)


def _prep_features(train: pl.DataFrame) -> pl.DataFrame:
    features = (
        train.select(pl.col(_FEATURES))
        .with_columns(
            pl.when(pl.col("Age") < 18)
            .then("child")
            .when(pl.col("Age").is_null())
            .then("unknown")
            .otherwise("adult")
            .alias("AgeStatus")
        )
        .drop(columns="Age")
    )

    return _one_hot_encode_features(features, ["Sex", "AgeStatus", "Pclass"])


def train_xgboost(train: pl.DataFrame) -> None:
    x_df = _prep_features(train)
    print(f"Training data post feature prep: {x_df.head(3)}")

    x = x_df.to_numpy()
    y = train.select(pl.col("Survived")).to_numpy()

    print(f"X input shape - {x.shape}")
    print(f"y labels shape - {y.shape}")
    print(f"X[:3] - {x[:3]}")
    print(f"y[:3] - {y[:3]}")

    model: xgb.sklearn.XGBClassifier = xgb.XGBClassifier()

    model.fit(x, y)

    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print(f"Module accuracy on training set - {accuracy * 100:.2f}%")

    model_f_path = _MODELS / "model.json"
    print(f"Saving model to {model_f_path}")
    model.save_model(model_f_path)


def eval_xgboost_model(data: pl.DataFrame) -> None:
    model: xgb.sklearn.XGBClassifier = xgb.XGBClassifier()
    model.load_model(_MODELS / "model.json")

    x_df = _prep_features(data)
    print(f"Data post feature prep: {x_df.head(3)}")
    x = x_df.to_numpy()

    y = data.select(pl.col("Survived")).to_numpy()

    print(f"X input shape - {x.shape}")
    print(f"y labels shape - {y.shape}")
    print(f"X[:3] - {x[:3]}")
    print(f"y[:3] - {y[:3]}")

    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print(f"Module accuracy on eval data set - {accuracy * 100:.2f}%")


if __name__ == "__main__":
    training_data = pl.read_csv(_TRANSFORMED / "train.csv")
    train_xgboost(training_data)

    cv = pl.read_csv(_TRANSFORMED / "cv.csv")
    eval_xgboost_model(cv)
