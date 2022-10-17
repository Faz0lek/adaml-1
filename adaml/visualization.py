""" This script was used to find theoretical values of variables and to visualize the data
    and it's distribution.

"""

import typing
import argparse
import os

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import helper


__author__ = "Martin Kostelnik, Marianne Jakonen, Ahmed Ansari"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to a file containing data.")
    parser.add_argument("--out-path", required=True, help="Path to an output folder.")
    parser.add_argument("--compression", default="infer", help="Compression format.")

    args = parser.parse_args()
    return args


def find_ranges(df: pd.DataFrame) -> pd.DataFrame:
    mins, maxs = df.min(axis=0), df.max(axis=0)

    range_df = pd.concat([mins, maxs], axis=1).reset_index().rename({"index": "column", 0: "min", 1: "max"}, axis=1)
    print(range_df)


def plot_boxplots(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    sns.set_theme()

    for ax, (col_name, col_data) in zip(axes.flat, df.iteritems()):
        sns_ax = sns.boxplot(ax=ax, data=col_data[col_data.notna()], showfliers=False)
        sns_ax.set(xticklabels=[], title=col_name)

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "boxplots.pdf"))


def plot_density(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    limits = [(-10, 100), (0, 100), (0, 3000), (2, 3)]

    for ax, col_name, limit in zip(axes.flat, df, limits):
        df[col_name].plot(kind="kde", ax=ax, xlim=limit, title=col_name, ylabel=None)

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "density.pdf"))


def plot_timeseries(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    df = df.resample("8H", on="datetime").mean().reset_index()

    df[df[helper.DATA_COLUMNS[0]] < 120].plot(kind="line", x="datetime", y=helper.DATA_COLUMNS[0], ax=axes.flat[0], grid=True, ylabel=helper.DATA_COLUMNS[0], xlabel=None, legend=False)
    df[df[helper.DATA_COLUMNS[1]] >= 0].plot(kind="line", x="datetime", y=helper.DATA_COLUMNS[1], ax=axes.flat[1], grid=True, ylabel=helper.DATA_COLUMNS[1], xlabel=None, legend=False)
    df.plot(kind="line", x="datetime", y=helper.DATA_COLUMNS[2], ax=axes.flat[2], grid=True, ylabel=helper.DATA_COLUMNS[2], xlabel=None, legend=False)
    df[(df[helper.DATA_COLUMNS[3]] < 5) & (df[helper.DATA_COLUMNS[3]] > 1)].plot(kind="line", x="datetime", y=helper.DATA_COLUMNS[3], ax=axes.flat[3], grid=True, ylabel=helper.DATA_COLUMNS[3], xlabel=None, legend=False)

    for ax in axes.flat:
        ax.set(xlabel=None)
        ax.axes.xaxis.set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "timeseries.pdf"))


if __name__ == "__main__":
    args = parse_arguments()
    
    os.makedirs(args.out_path, exist_ok=True)

    df = helper.load_data(args.data_path, args.compression)

    find_ranges(df)
    plot_boxplots(df[helper.DATA_COLUMNS], args.out_path)
    plot_density(df[helper.DATA_COLUMNS], args.out_path)
    plot_timeseries(df, args.out_path)
