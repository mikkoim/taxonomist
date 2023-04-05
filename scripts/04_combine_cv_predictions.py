import pandas as pd
from pathlib import Path
import numpy as np

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--reference_csv", type=str, required=False)
    parser.add_argument("--reference_target", type=str)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--around", type=int)

    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    out_folder = Path(args.model_folder) / "predictions"
    out_folder.mkdir(exist_ok=True, parents=True)

    if args.reference_csv:
        ref_df = pd.read_csv(args.reference_csv)

    csv_list = []
    idx_list = []
    for fold in range(args.start_fold, args.start_fold + args.n_folds):
        pred_folder = model_folder / f"f{fold}" / "predictions" / f"{args.tag}"
        f = next(pred_folder.glob("*.csv"))
        df_fold = pd.read_csv(f)
        csv_list.append(df_fold)

        if args.reference_csv:
            # To get the position from the ground truth csv
            idx = ref_df[ref_df[str(fold)] == "test"].index.values
            idx_list.append(idx)

            print(f"fold: {fold} | idx length: {len(idx)} | df length: {len(df_fold)}")
        else:
            print(f"fold: {fold} | df length: {len(df_fold)}")

    # Rearrange
    df = pd.concat(csv_list, ignore_index=True)

    if args.reference_csv:
        idx = np.concatenate(idx_list)
        df.index = idx
        df = df.sort_index()

        if df.y_true.dtype == "O":
            assert np.all(ref_df[args.reference_target] == df.y_true)
        else:
            assert np.allclose(ref_df[args.reference_target], df.y_true)

    # Possible rounding
    if args.around:
        print(f"Rounding values to {args.around} decimals")
        df = df.round(args.around)

    out_fname = out_folder / f"{model_folder.name}_{args.tag}.csv"
    df.to_csv(out_fname, index=False)
    print(f"Done! Saved output to {out_fname}")
