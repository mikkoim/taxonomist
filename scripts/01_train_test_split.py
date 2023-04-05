from pathlib import Path
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--group_col", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--out_folder", type=str, default=".")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv_path)

    # Splits
    try:
        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

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
        df.assign(**{str(fold): 0})

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
        for f in range(args.n_splits):
            infodf = pd.DataFrame()
            infodf.index = df[args.target_col].unique()
            vc = lambda f, set_: df[df[str(f)] == set_][args.target_col].value_counts()

            infodf[f"train_{str(f)}"] = vc(f, "train")
            infodf[f"val_{str(f)}"] = vc(f, "val")
            infodf[f"test_{str(f)}"] = vc(f, "test")
            print(f"Fold {f}")
            print(infodf.fillna(0).astype(int))
            m = infodf[infodf.isna().any(axis=1)]
            if len(m) > 0:
                print("\nMissing classes:")
                print(m)
            print()
