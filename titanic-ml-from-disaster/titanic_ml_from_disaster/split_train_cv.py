from pathlib import Path

import polars as pl

_DATA = Path(__file__).parent.parent / "data"
_RAW_DATA = _DATA / "raw"
_TRANSFORMED = _DATA / "transformed"
_SEED = 99

if __name__ == "__main__":
    raw_train: pl.DataFrame = pl.read_csv(_RAW_DATA / "train.csv")

    raw_train_shuffled: pl.DataFrame = raw_train.sample(
        fraction=1, shuffle=True, seed=_SEED
    )
    num_rows = raw_train.height
    train_size = int(num_rows * 0.7)
    cv_size = num_rows - train_size

    train = raw_train_shuffled.head(train_size)
    cv = raw_train_shuffled.tail(cv_size)

    train.write_csv(_TRANSFORMED / "train.csv")
    cv.write_csv(_TRANSFORMED / "cv.csv")
