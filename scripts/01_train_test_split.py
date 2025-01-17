import argparse

from taxonomist.input_config import add_train_test_split_args
from taxonomist.preprocessing import train_test_split, TrainTestSplitArgs

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
    parser = add_train_test_split_args(parser)

    args = parser.parse_args()

    train_test_split(TrainTestSplitArgs(**vars(args)))

