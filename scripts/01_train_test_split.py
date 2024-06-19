import argparse
from pathlib import Path
from distutils.util import strtobool

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

DESCRIPTION = """
Takes a dataset csv file and splits it to train, test and validation splits.
Assumes that a single row corresponds to a single sample, i.e. image.
Keeps the csv structure as is, but adds N columns, where N is the number of 
cross-validation splits.
Each column has string values 'train', 'test' and 'val', denoting the split 
the sample 
belongs to.


The 'test' sets are mutually exclusive, and together make up the full dataset.


For categorical variables, the splits are stratified, and samples are 
distributed equally
based on the 'target_col' parameter.


If dataset has groups that might induce data leakage, groups can be separated 
across splits
with the 'group_col' parameter. All samples of a group will then belong to a 
single split.

Creates a log of class counts in different splits along the final file.
Output file is named with the original file, number of splits and the target 
column info
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument(
        "--csv_path", type=str, help="Path to input csv file", required=True
    )

    parser.add_argument(
        "--target_col",
        type=str,
        help="Target variable column. Stratification " "is performed based on this",
        required=True,
    )

    parser.add_argument(
        "--group_col",
        type=str,
        help="Group column. Groups are non-overlapping " "across train-test-val splits",
        required=True,
    )

    parser.add_argument(
        "--n_splits", type=int, help="Number of splits. Default 5", default=5
    )

    parser.add_argument(
        "--verbose",
        type=int,
        help="If set to 1, prints information on data " "splits to console",
        default=1,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="The random seed for the split. Default is 42",
        default=42,
    )
    parser.add_argument(
        "--shuffle",
        type=lambda x: bool(strtobool(x)),
        help="Whether to shuffle each class's samples before splitting into batches.",
        nargs="?",
        const=True,
        default=True,
        required=False,
    )

    parser.add_argument("--out_folder", type=str, default=".")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv_path)

    # Splits
    try:
        random_state = args.random_state if args.shuffle else None
        cv = StratifiedGroupKFold(
            n_splits=args.n_splits, shuffle=args.shuffle, random_state=random_state
        )

        # Try to calculate the splits for exception catchment
        next(iter(cv.split(df, df[args.target_col], df[args.group_col])))
        print("Target categorical, using StratifiedGroupKFold")
        is_categorical = True

    except ValueError:  # If above fails
        cv = GroupKFold(n_splits=args.n_splits)
        print("Target continuous, using GroupKFold")
        is_categorical = False

    train_df = {}
    test_df = {}
    val_df = {}
    # Go through each cv fold
    for fold, (temp_idx, test_idx) in enumerate(
        cv.split(df, df[args.target_col], df[args.group_col])
    ):
        test_df[fold] = df.iloc[test_idx]
        temp_df = df.iloc[temp_idx]

        # Split rest to train and validation
        train_idx, val_idx = next(
            iter(cv.split(temp_df, temp_df[args.target_col], temp_df[args.group_col]))
        )
        train_df[fold] = temp_df.iloc[train_idx]
        val_df[fold] = temp_df.iloc[val_idx]

    # Create new columns
    for fold in range(args.n_splits):
        df = df.assign(**{str(fold): "0"})

    # Assign column values
    for fold in range(args.n_splits):
        train_list = train_df[fold][args.group_col].unique()
        test_list = test_df[fold][args.group_col].unique()
        val_list = val_df[fold][args.group_col].unique()

        df.loc[df[args.group_col].isin(train_list), str(fold)] = "train"
        df.loc[df[args.group_col].isin(test_list), str(fold)] = "test"
        df.loc[df[args.group_col].isin(val_list), str(fold)] = "val"

    out_fname = out_folder / (
        csv_path.stem + f"_{args.n_splits}splits_{args.target_col}.csv"
    )
    df.to_csv(out_fname, index=False)
    print(out_fname)

    # Print information on data splits
    if (args.verbose == 1) and is_categorical:
        log_fname = out_folder / (out_fname.stem + "_log.txt")
        with open(log_fname, "w") as file:
            for f in range(args.n_splits):
                infodf = pd.DataFrame()
                infodf.index = df[args.target_col].unique()

                def vc(f, set_):
                    return df[df[str(f)] == set_][args.target_col].value_counts()

                infodf[f"train_{str(f)}"] = vc(f, "train")
                infodf[f"val_{str(f)}"] = vc(f, "val")
                infodf[f"test_{str(f)}"] = vc(f, "test")
                p = str(f"Fold {f}\n")
                print(p)
                file.writelines([p])

                p = str(infodf.fillna(0).astype(int))
                print(p)
                file.writelines([p])

                m = infodf[infodf.isna().any(axis=1)]
                if len(m) > 0:
                    p = f"\nMissing classes:\n{str(m.fillna(0).astype(int))}"
                    print(p)
                    file.writelines([p])
                print()
                file.write("\n\n")
