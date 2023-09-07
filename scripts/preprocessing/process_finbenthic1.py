from pathlib import Path
import pandas as pd
import argparse


def process_dataset_partition(dfp, file_df):
    dfp = dfp.assign(fname=dfp["im_num"].apply(lambda x: x.split("\\")[2]))
    dfp = dfp.assign(im_num_true=dfp.fname.apply(lambda x: x.split("_")[0]))
    dfp = dfp.set_index(["im_path", "im_num_true"]).sort_index()

    filtered_df = file_df.loc[dfp.index].reset_index()
    filtered_df = filtered_df.rename(columns={"im_path": "taxon", "fname": "img"})
    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, default=".")

    args = parser.parse_args()

    print("Create folders...")
    folder = Path(args.folder)
    out_folder_base = Path(args.out_folder)
    out_folder = {}
    out_folder[1] = out_folder_base / "finbenthic1-1"
    out_folder[2] = out_folder_base / "finbenthic1-2"
    out_folder[3] = out_folder_base / "finbenthic1-3"

    out_folder[1].mkdir(exist_ok=True, parents=True)
    out_folder[2].mkdir(exist_ok=True, parents=True)
    out_folder[3].mkdir(exist_ok=True, parents=True)

    img_folder = folder / "Cropped images"

    # Load partition info
    cols = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        "data_partitions",
        "im_num",
        "im_path",
        "cls_name",
        "cls_id",
    ]
    df = {}
    df[1] = pd.read_csv(
        folder / "dataset1_partitions.txt", skiprows=1, sep=" ", names=cols
    )
    df[2] = pd.read_csv(
        folder / "dataset2_partitions.txt", skiprows=1, sep=" ", names=cols
    )
    df[3] = pd.read_csv(
        folder / "dataset3_partitions.txt", skiprows=1, sep=" ", names=cols
    )

    # Read files
    all_taxons = [f.name for f in img_folder.glob("*") if f.is_dir()]

    print("Reading files...")
    filelist = []

    # Finds all image files in the folders
    for taxon in all_taxons:
        taxon_folder = img_folder / taxon
        files = list(taxon_folder.glob("*.png"))
        fdf = pd.DataFrame({"fname": files, "im_path": taxon})

        fdf = fdf.assign(
            im_num_true=fdf["fname"].apply(lambda x: x.stem.split("_")[0]),
            fname=fdf["fname"].apply(lambda x: x.name),
        )
        filelist.append(fdf)

    file_df = pd.concat(filelist)
    file_df = file_df.assign(
        individual=file_df.apply(lambda x: x["im_path"] + x["im_num_true"], axis=1)
    )
    file_df = file_df.set_index(["im_path", "im_num_true"]).sort_index()

    # Filters the images found in the folders based on the partition
    for key, item in df.items():
        filt_df = process_dataset_partition(item, file_df)
        label_map_fname = out_folder[key] / f"label_map_finbenthic1-{key}.txt"
        pd.DataFrame(filt_df["taxon"].unique().astype("str")).sort_values(0).to_csv(
            label_map_fname, header=False, index=False
        )

        out_fname = out_folder[key] / f"01_finbenthic1-{key}_processed.csv"
        filt_df.to_csv(out_fname, index=False)
        print(out_fname)
    print("Done!")
