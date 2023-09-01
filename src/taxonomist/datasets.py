"""
Defines custom functions for reading dataset data from train-test-splitted csv-files
"""
from pathlib import Path
import pandas as pd


def preprocess_dataset(data_folder, dataset_name, csv_path=None, fold=None, label=None):
    data_folder = Path(data_folder)
    fnames = {}
    labels = {}
    for set_ in ["train", "val", "test"]:
        if dataset_name == "rodi":
            fnames[set_], labels[set_] = process_split_csv_rodi(
                data_folder, csv_path, set_, fold, label
            )
        elif dataset_name == "finbenthic2":
            fnames[set_], labels[set_] = process_split_csv_finbenthic2(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "my_dataset":
            """

            YOUR CODE GOES HERE

            """
            fnames[set_], labels[set_] = None, None

        else:
            raise Exception("Unknown dataset name")

    return fnames, labels


def process_split_csv_finbenthic2(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(data_folder, "Images", x["individual"], x["img"]).resolve(),
        axis=1,
    ).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels


def process_split_csv_rodi(data_folder, csv_path, set_, fold, label):
    "RODI -specific function for reading train-test-split csvs"
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df["image"].apply(lambda x: data_folder.resolve() / x).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels
