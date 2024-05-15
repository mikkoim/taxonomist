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

        elif dataset_name == "finbenthic1":
            fnames[set_], labels[set_] = process_split_csv_finbenthic1(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "cifar10":
            fnames[set_], labels[set_] = process_split_csv_cifar10(
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


def process_split_csv_finbenthic1(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(data_folder, "Cropped images", x["taxon"], x["img"]).resolve(),
        axis=1,
    ).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

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

def process_split_csv_cifar10(data_folder, csv_path, set_, fold, label):
    "CIFAR10 -specific function for reading train-test-split csvs"
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df["Filename"].apply(lambda x: data_folder.resolve() / x).values

    # Print file paths for debugging
    print("File paths:", fnames)

    # Check if the files exist
    for fname in fnames:
        assert fname.exists(), f"File '{fname}' does not exist"  # Add informative message

    labels = df[label].values

    return fnames, labels


# def process_split_csv_cifar10(data_folder, csv_path, set_, fold, label):
#     "CIFAR10-specific function for reading train-test-split csvs"
#     df0 = pd.read_csv(csv_path)
    
#     # Check if the column named with the fold value exists
#     fold_column = f"split_{fold}"
#     if fold_column not in df0.columns:
#         raise ValueError(f"Column '{fold_column}' not found in DataFrame")

#     # Filter DataFrame based on fold and set_
#     df = df0[df0[fold_column] == set_]

#     # Check if the set_ values exist in the fold column
#     if df.empty:
#         raise ValueError(f"No data found for fold '{fold_column}' and set '{set_}'")

#     # Construct file paths
#     fnames = df.apply(lambda x: data_folder.resolve() / str(x["ID"]) / x["Filename"], axis=1).values

#     # Print file paths for debugging
#     print("File paths:", fnames)

#     # Check if the files exist
#     for fname in fnames:
#         assert fname.exists(), f"File '{fname}' does not exist"  # Add informative message

#     labels = df[label].values

#     return fnames, labels



# def process_split_csv_cifar10(data_folder, csv_path, set_, fold, label):
#     "CIFAR10-specific function for reading train-test-split csvs"
#     df0 = pd.read_csv(csv_path)
    
#     # Check if the column named with the fold value exists
#     fold_column = f"{fold}"
#     if fold_column not in df0.columns:
#         raise ValueError(f"Column '{fold_column}' not found in DataFrame")

#     # Filter DataFrame based on fold and set_
#     df = df0[df0[fold_column] == set_]

#     # Check if the set_ values exist in the fold column
#     if df.empty:
#         raise ValueError(f"No data found for fold '{fold_column}' and set '{set_}'")

#     # Construct file paths
#     #fnames = df["Filename"].apply(lambda x: data_folder.resolve() / "Images" / x).values
#     fnames = df["Filename"].apply(lambda x: data_folder.resolve() / x).values

#     # Print file paths for debugging
#     print("File paths:", fnames)

#     # Check if the files exist
#     for fname in fnames:
#         assert fname.exists(), f"File '{fname}' does not exist"  # Add informative message

#     labels = df[label].values

#     return fnames, labels






def process_split_csv_cifar10(data_folder, csv_path, set_, fold, label):
    "CIFAR10 -specific function for reading train-test-split csvs"
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df["Filename"].apply(lambda x: data_folder.resolve() / x).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels
