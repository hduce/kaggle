from pathlib import Path

from sklearn.metrics import accuracy_score
import xgboost as xgb
import polars as pl

_TRANSFORMED = Path(__file__).parent.parent / "data" / "transformed"
_MODELS = Path(__file__).parent.parent / "models"


def train_xgboost(train: pl.DataFrame) -> None:
    x = train.select(pl.col("*").exclude("Label"))
    y = train.select(pl.col("Label"))

    print(f"X input shape - {x.shape}")
    print(f"y labels shape - {y.shape}")
    print(f"X.head(3) - {x.head(3)}")
    print(f"y.head(3) - {y.head(3)}")

    model = xgb.XGBClassifier()

    model.fit(x, y)

    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print(f"Module accuracy on training set - {accuracy * 100:.2f}%")

    model_f_path = _MODELS / "model_features_sex_agestatus.json"
    print(f"Saving model to {model_f_path}")
    model.save_model(model_f_path)


if __name__ == "__main__":
    training_data = pl.read_csv(_TRANSFORMED / "train_features_sex_agestatus.csv")
    train_xgboost(training_data)
