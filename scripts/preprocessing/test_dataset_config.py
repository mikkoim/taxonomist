import argparse
import sys
from warnings import warn
import taxonomist as src

DESCRIPTION = """
Tests that the data loading function in the config file works as intended
"""


def main(args):
    dataset_config_module = src.utils.load_module_from_path(args.dataset_config_path)

    fpaths, labels = dataset_config_module.preprocess_dataset(
        data_folder=args.data_folder,
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
        fold=args.fold,
        label=args.label_column,
    )

    for set_ in ["train", "val", "test"]:
        print(f"{set_}: found {len(fpaths[set_])} fpaths, {len(labels[set_])} labels")
        print("Examples:")
        print(fpaths[set_][:5])
        print(labels[set_][:5])

        if not isinstance(fpaths[set_], (list, np.ndarray)):
            warn(
                f"fpaths {set_} is not a list or a numpy array. Check that it can be indexed correctly"
            )
        if not isinstance(labels[set_], (list, np.ndarray)):
            warn(
                f"labels {set_} is not a list or a numpy array. Check that it can be indexed correctly"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--data_folder")
    parser.add_argument("--dataset_config_path")
    parser.add_argument("--dataset_name")
    parser.add_argument("--csv_path")
    parser.add_argument("--fold")
    parser.add_argument("--label_column")

    args = parser.parse_args()
    main(args)

