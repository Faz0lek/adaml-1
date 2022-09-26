import typing
import pandas as pd
import numpy as np

from functools import reduce


COLUMN_NAMES = ["date", "time", "epoch", "mote_id", "temperature", "humidity", "light", "voltage"]

DATA_COLUMNS = ["temperature", "humidity", "light", "voltage"]

COLUMN_DTYPES = [object, object, "Int32", "Int32", np.float32, np.float32, np.float32, np.float32]


def load_data(path: str, compression="gzip") -> pd.DataFrame:
    dtypes = {name: dtype for name, dtype in zip(COLUMN_NAMES, COLUMN_DTYPES)}

    df = pd.read_csv(path, delimiter=" ", names=COLUMN_NAMES, compression=compression, dtype=dtypes)

    # Add datetime column
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time", "epoch"])
    
    # Some basic filtering based on theoretical values
    df = df[(df["mote_id"] >= 1) & (df["mote_id"] <= 54)]
    df = df[df["temperature"] >= -274]
    df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]
    df = df[df["light"] >= 0]
    
    df = df.set_index("datetime")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.pivot(columns="mote_id", values="temperature").add_prefix("T-").reset_index()
    light_df = df.pivot(columns="mote_id", values="light").add_prefix("L-").reset_index()
    humidity_df = df.pivot(columns="mote_id", values="humidity").add_prefix("H-").reset_index()
    voltage_df = df.pivot(columns="mote_id", values="voltage").add_prefix("V-").reset_index()
    
    dfs = [temp_df, light_df, humidity_df, voltage_df]
    
    df = reduce(lambda left, right: pd.merge(left, right, on=["datetime"]), dfs)

    df = df.resample("2H", on="datetime").mean()
    df = df.dropna(how="all")
    df = df.dropna(thresh=int(54*4*0.8))
    df = df.dropna(axis=1, thresh=len(df.index) - 10)
    
    df = df.interpolate(axis=1)
    df = (df - df.mean()) / df.std()
    
    return df
    