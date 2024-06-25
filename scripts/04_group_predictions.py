import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DESCRIPTION = """
Performs aggregation to predictions, based on a group variable in the original dataset.
"""

def quantile_mean(series):
    """Returns the mean after values outside the 5th and 95th percentile are removed"""
    q5 = series.quantile(0.05)
    q95 = series.quantile(0.95)
    return series[(q5 <= series) & (series <= q95)].mean()

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
    if len(ref_a) != len(ref_b):
        raise ValueError(
            "Predictions and reference sizes dont match. "
            f"Reference size is {len(ref_a)} and prediction size is {len(ref_b)}."
        )

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
    elif args.agg_func == "quantile_mean":
        agg_func = quantile_mean

    else:
        agg_func = args.agg_func

    group_df = group_df.agg(agg_func)
    print(f"Grouped {len(df)} rows to {len(group_df)} groups")

    if args.around:
        print(f"Rounding values to {args.around} decimals")
        try:
            group_df = group_df.map(lambda x: np.around(x, args.around))
        except np.core._exceptions._UFuncNoLoopError:
            raise Exception("Can't round values. Only regression tasks can be rounded")

    out_name = out_folder / f"{csv_stem}_grouped.csv"
    group_df.to_csv(out_name)
    print(f"Saved to {out_name}")
