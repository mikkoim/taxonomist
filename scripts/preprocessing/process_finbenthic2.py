from pathlib import Path
import pandas as pd
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IDA_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, default=".")

    args = parser.parse_args()

    folder = Path(args.IDA_folder)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    img_folder = folder / "Images"

    df = pd.read_csv(folder / "Machine_learning_splits.txt", sep=" ")

    # Save taxa lists
    def save_class_map(col, name):
        taxa = np.sort(df[col].unique().astype(str))
        with open(out_folder / name, "w") as f:
            for lab in taxa:
                if lab != "nan":
                    f.write(lab + "\n")

    save_class_map("Class", "label_map_01_taxon.txt")
    save_class_map("Class_species", "label_map_02_species.txt")
    save_class_map("Class_genus", "label_map_03_genus.txt")
    save_class_map("Class_family", "label_map_04_family.txt")
    save_class_map("Class_order", "label_map_05_order.txt")

    df1 = df.drop(
        [
            "SPLIT1",
            "SPLIT2",
            "SPLIT3",
            "SPLIT4",
            "SPLIT5",
            "SPLIT6",
            "SPLIT7",
            "SPLIT8",
            "SPLIT9",
            "SPLIT10",
        ],
        axis=1,
    )

    df1 = df1.rename(
        {
            "tunnus": "individual",
            "Class": "taxon",
            "Class_species": "species",
            "Class_genus": "genus",
            "Class_family": "family",
            "Class_order": "order",
            "Img_id": "img",
        },
        axis=1,
    )

    out_fname = out_folder / "01_finbenthic2_processed.csv"
    df1.to_csv(out_fname, index=False)
    print(out_fname)
