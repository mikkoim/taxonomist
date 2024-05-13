from pathlib import Path
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_folder", type=str, default=".")

    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.csv_path, encoding="ISO-8859-1", sep=";")

    df = df[
        [
            "Sample Name/Number",
            "Species Name",
            "Image File Name",
            "Sample Station",
            "Location",
            "Area",
            "Perimeter",
        ]
    ]

    df = df.assign(
        individual=df["Sample Name/Number"].apply(lambda x: x.split("_")[:-1][0])
    )

    out_fname = out_folder / "01_biodiscover_processed.csv"
    df.to_csv(out_fname, index=False)
    print(out_fname)
