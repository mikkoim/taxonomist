import pandas as pd
from pathlib import Path
import numpy as np

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--reference_csv", type=str)
    parser.add_argument("--reference_target", type=str)
    parser.add_argument("--reference_group", type=str)
    parser.add_argument("--around", type=int)

    args = parser.parse_args()

    out_folder = Path(args.predictions).parents[0]

    csv_stem = Path(args.predictions).stem

    df = pd.read_csv(args.predictions)
    ref_df = pd.read_csv(args.reference_csv)

    assert np.allclose(ref_df[args.reference_target], df.y_true)

    # Combine predictions and reference
    comb_df = pd.concat((df, ref_df), axis=1)

    # Grouping
    group_df = comb_df.groupby(args.reference_group)[["y_true", "y_pred"]].mean()

    print(f"Grouped {len(df)} rows to {len(group_df)} groups")

    if args.around:
        print(f"Rounding values to {args.around} decimals")
        group_df = group_df.applymap(lambda x: np.around(x, args.around))

    out_name = out_folder / f"{csv_stem}_grouped.csv"
    group_df.to_csv(out_name)
    print(f"Saved to {out_name}")
