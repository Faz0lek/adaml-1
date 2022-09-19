import typing
import pandas as pd


COLUMN_NAMES = ["date", "time", "epoch", "mote_id", "temperature", "humidity", "light", "voltage"]

DATA_COLUMNS = ["temperature", "humidity", "light", "voltage"]

COLUMN_DTYPES = [object, object, pd.Int64Dtype(), pd.Int64Dtype(), pd.Float64Dtype(), pd.Float64Dtype(), pd.Float64Dtype(), pd.Float64Dtype()]


def load_data(path: str, compression="gzip") -> pd.DataFrame:
    dtypes = {name: dtype for name, dtype in zip(COLUMN_NAMES, COLUMN_DTYPES)}

    df = pd.read_csv(path, delimiter=" ", names=COLUMN_NAMES, compression=compression, dtype=dtypes)

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"])

    return df
