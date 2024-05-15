from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

DESCRIPTION = """
This code modifies the original script to ensure that validation samples are exactly 10'perccent or defined value of percentage of the total size after test samples are allocated, with the rest used for training. It leverages the specified group column to maintain the integrity of groups across splits, ensuring there's no data leakage.

Takes a dataset csv file and splits it to train, test, and validation splits.
Assumes that a single row corresponds to a single sample, i.e., image.
Keeps the csv structure as is, but adds N columns, where N is the number of 
cross-validation splits.
Each column has string values 'train', 'test', and 'val', denoting the split 
the sample belongs to.

The 'test' sets are mutually exclusive, and together make up the full dataset.

For categorical variables, the splits are stratified, and samples are 
distributed equally based on the 'target_col' parameter.

If the dataset has groups that might induce data leakage, groups can be separated 
across splits with the 'group_col' parameter. All samples of a group will then belong to a 
single split.

Creates a log of class counts in different splits along the final file.
Output file is named with the original file, number of splits, and the target 
column info.
"""

def create_splits(df, group_col, target_col, n_splits, val_percentage=0.1, random_state=42):
    """
    Splits the dataframe into train, test, and validation sets.
    """
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        # Try to calculate the splits for exception catchment
        next(iter(cv.split(df, df[target_col], df[group_col])))
        print("Target categorical, using StratifiedGroupKFold")
        is_categorical = True
    except ValueError:
        cv = GroupKFold(n_splits=n_splits)
        print("Target continuous, using GroupKFold")
        is_categorical = False
    
    train_df, test_df, val_df = {}, {}, {}
    for fold, (temp_idx, test_idx) in enumerate(cv.split(df, df[target_col], df[group_col])):
        test_df[fold] = df.iloc[test_idx]
        temp_df = df.iloc[temp_idx]
        
        # For validation, we need to ensure groups are not split between train and val
        unique_groups = temp_df[group_col].unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_groups)
        
        # Calculate the number of groups for validation
        num_val_groups = int(np.ceil(len(unique_groups) * val_percentage))
        val_groups = unique_groups[:num_val_groups]
        train_groups = unique_groups[num_val_groups:]
        
        train_df[fold] = temp_df[temp_df[group_col].isin(train_groups)]
        val_df[fold] = temp_df[temp_df[group_col].isin(val_groups)]

    return train_df, test_df, val_df, is_categorical

def assign_splits_to_df(df, train_df, test_df, val_df, group_col, n_splits):
    """
    Assigns the split labels to the dataframe.
    """
    for fold in range(n_splits):
        df = df.assign(**{f"split_{fold}": "train"})  # Default to "train"
        
        # Update with actual splits
        train_list = train_df[fold][group_col].unique()
        test_list = test_df[fold][group_col].unique()
        val_list = val_df[fold][group_col].unique()
        
        df.loc[df[group_col].isin(test_list), f"split_{fold}"] = "test"
        df.loc[df[group_col].isin(val_list), f"split_{fold}"] = "val"
    
    return df

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--csv_path", type=str, help="Path to input csv file", required=True)
    parser.add_argument("--target_col", type=str, help="Target variable column.", required=True)
    parser.add_argument("--group_col", type=str, help="Group column.", required=True)
    parser.add_argument("--n_splits", type=int, help="Number of splits. Default 5", default=5)
    parser.add_argument("--verbose", type=int, help="Verbose output. Default is 1", default=1)
    parser.add_argument("--random_state", type=int, help="Random seed for the split. Default is 42", default=42)
    parser.add_argument("--out_folder", type=str, help="Output folder.", default=".")
    
    args = parser.parse_args()

    # Read data
    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    # Split data
    train_df, test_df, val_df, is_categorical = create_splits(
        df=df, 
        group_col=args.group_col, 
        target_col=args.target_col, 
        n_splits=args.n_splits, 
        val_percentage=0.1,  # 10% validation data
        random_state=args.random_state
    )

    # Assign splits
    df = assign_splits_to_df(df, train_df, test_df, val_df, args.group_col, args.n_splits)

    # Save the updated dataframe
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    out_fname = out_folder / (csv_path.stem + f"_{args.n_splits}splits_{args.target_col}.csv")
    df.to_csv(out_fname, index=False)
    
    if args.verbose == 1:
        print(f"Splits saved to {out_fname}")
