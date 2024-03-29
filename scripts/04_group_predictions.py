import pandas as pd
from pathlib import Path
import numpy as np

import argparse

DESCRIPTION = """
Performs aggregation to predictions, based on a group variable in the original dataset.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", type=str)
    parser.add_argument("--reference_csv", type=str)
    parser.add_argument("--reference_target", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--reference_group", type=str)
    parser.add_argument("--agg_func", type=str)
    parser.add_argument(
        "--around", type=int, help="Round the output. Only on regression"
    )

    args = parser.parse_args()

    out_folder = Path(args.predictions).parents[0]

    csv_stem = Path(args.predictions).stem

    df = pd.read_csv(args.predictions)
    ref_df = pd.read_csv(args.reference_csv)

    if len(ref_df) != len(df):
        if args.fold is None:
            raise ValueError(
                "Predictions and reference don't match."
                " Set a fold parameter if grouping a single fold"
            )
        ref_df = ref_df[ref_df[str(args.fold)] == args.set].reset_index(drop=True)

    # Check that reference matches
    ref_a = ref_df[args.reference_target]
    ref_b = df.y_true
    try:
        if not np.allclose(ref_a, ref_b):
            raise ValueError("Reference column does not match ground truth.")

    except TypeError:  # categorical variable
        if not (ref_a == ref_b).all():
            raise ValueError("Reference column does not match ground truth.")

    # Combine predictions and reference
    comb_df = pd.concat((df, ref_df), axis=1)

    # Grouping
    group_df = comb_df.groupby(args.reference_group)[["y_true", "y_pred"]]

    if args.agg_func == "mode":

        def agg_func(x):
            return pd.Series.mode(x)[0]

    else:
        agg_func = args.agg_func

    group_df = group_df.agg(agg_func)
    print(f"Grouped {len(df)} rows to {len(group_df)} groups")

    if args.around:
        print(f"Rounding values to {args.around} decimals")
        group_df = group_df.map(lambda x: np.around(x, args.around))

    out_name = out_folder / f"{csv_stem}_grouped.csv"
    group_df.to_csv(out_name)
    print(f"Saved to {out_name}")
