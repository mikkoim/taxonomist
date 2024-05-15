
from pathlib import Path
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_folder", type=str, default=".")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.csv_path)
    # Do the necessary cleaning here

    # Save the labels to a label map
    with open(out_folder / "cifar10_label_map.txt", "w") as f:
        for lab in df.Label.value_counts().index.values:
            f.write(lab + "\n")

    out_fname = out_folder / "01_cifar10_processed.csv"
    df.to_csv(out_fname, index=False)
    print(out_fname)
