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


def _prep_features(df: pl.DataFrame) -> pl.DataFrame:
    features = df.select(
        pl.col(
            [
                "Sex",
                "Age",
                "Pclass",
                "Parch",
                "Cabin",
                "SibSp",
                "Fare",
                "Embarked",
            ],
        )
    )
    features = features.with_columns(
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
        pl.when(pl.col("Embarked").is_null())
        .then("unknown")
        .otherwise(
            pl.col("Embarked"),
        )
        .alias("Embarked"),
    )

    return _one_hot_encode_features(
        features,
        [
            "Sex",
            "Age",
            "Pclass",
            "Cabin",
            "Embarked",
        ],
    )


if __name__ == "__main__":
    raw_train_all = pl.read_csv(_TRANSFORMED / "train.csv")

    train_all = _prep_features(raw_train_all)
    print(f"Prepped training features {train_all.head(3)}")

    x = train_all.to_numpy()
    y = raw_train_all.select(pl.col("Survived")).to_numpy()

    X_train, X_cv, y_train, y_cv = train_test_split(
        x, y, test_size=0.3, random_state=99
    )

    # Just to be safe
    del x, y

    print(f"X_train input shape - {X_train.shape}")
    print(f"y_train labels shape - {y_train.shape}")

    model: xgb.sklearn.XGBClassifier = xgb.XGBClassifier(
        learning_rate=0.5,
        n_estimators=120,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Module accuracy on training set - {accuracy * 100:.2f}%")

    print(f"X_cv input shape - {X_cv.shape}")
    print(f"y_cv labels shape - {y_cv.shape}")

    y_cv_pred = model.predict(X_cv)
    accuracy = accuracy_score(y_cv, y_cv_pred)

    print(f"Module accuracy on cv data set - {accuracy * 100:.2f}%")
    model_f_path = _MODELS / "model.json"
    print(f"Saving model to {model_f_path}")
    model.save_model(model_f_path)

    ## Test data
    test_all_df = pl.read_csv(_TRANSFORMED / "test.csv")
    test_features = _prep_features(test_all_df)
    print(f"Prepped test features {test_features.head(3)}")

    print("Filling Cabin_T and EmbarkedUnknown columns")
    test_features = test_features.with_columns(
        pl.lit(0).alias("Cabin_T"),
        pl.lit(0).alias("Embarked_unknown"),
    )

    X_test = test_features.to_numpy()
    print(f"X_test input shape - {X_test.shape}")

    y_test_pred = model.predict(X_test)
    test_predictions = test_all_df.select(pl.col("PassengerId"))
    test_predictions = test_predictions.hstack(
        pl.from_numpy(y_test_pred, schema={"Survived": pl.Int8})
    )

    test_pred_f = _TRANSFORMED / "test_predictions.csv"
    print(f"Writing test predictions to {test_pred_f}")
    test_predictions.write_csv(test_pred_f)
