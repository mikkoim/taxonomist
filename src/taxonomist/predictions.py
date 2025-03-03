import pandas as pd
from scipy.special import softmax
import numpy as np
from dataclasses import dataclass
from pathlib import Path


class TaxonomistPredictions:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.fnames = None

    def set_y_true(self, y_true):
        self.y_true = y_true

    def set_y_pred(self, y_pred):
        self.y_pred = y_pred

    def set_fnames(self, fnames):
        self.fnames = fnames

    def set_logits(self, logits):
        self.logits = logits
        self.n_classes = logits.shape[1]

    def set_class_map(self, class_map):
        self.class_map = class_map
        n_in_class_map = len(class_map["fwd_dict"])
        if n_in_class_map != self.n_classes:
            self.classes = class_map["inv"](list(range(self.n_classes)))
        else:
            self.classes = range(self.n_classes)

    def get_y_true_y_pred(self):
        df = pd.DataFrame({"y_true": self.y_true, "y_pred": self.y_pred})
        df.index = self.fnames
        df.index.name = "fname"
        return df

    def get_softmax(self):
        return softmax(self.logits, axis=1)

    def get_full_df(self, softmax=False):
        df_pred = self.get_y_true_y_pred()
        if softmax:
            df_prob = pd.DataFrame(self.get_softmax(), columns=self.classes)
        else:
            df_prob = pd.DataFrame(self.logits, columns=self.classes)

        df = pd.concat(
            (df_pred.reset_index(drop=True), df_prob.reset_index(drop=True)), axis=1
        )
        df.index = self.fnames
        df.index.name = "fname"
        return df

@dataclass
class CombineCVPredictionsArgs:
    model_folder: str
    tag: str
    reference_csv: str
    reference_target: str
    suffix: str = ".csv"
    n_folds: int = 5
    start_fold: int = 0
    around: int = None

def combine_cv_predictions(args: CombineCVPredictionsArgs):
    model_folder = Path(args.model_folder)
    out_folder = Path(args.model_folder) / "predictions"
    out_folder.mkdir(exist_ok=True, parents=True)

    if args.reference_csv:
        ref_df = pd.read_csv(args.reference_csv)
        if args.reference_target not in ref_df.columns:
            raise ValueError(f"Reference target not in reference csv columns: {ref_df.columns.tolist()}")

    csv_list = []
    idx_list = []
    for fold in range(args.start_fold, args.start_fold + args.n_folds):
        pred_folder = model_folder / f"f{fold}" / "predictions" / f"{args.tag}"
        if not pred_folder.exists():
            raise ValueError(
                f"The path {pred_folder} does not exist. Check fold `{fold}` and tag `{args.tag}`"
            )

        csvs_in_folder = list(pred_folder.glob(f"*{args.suffix}"))
        if len(csvs_in_folder) != 1:
            raise ValueError(
                f"The number of files that match the suffix {args.suffix} is {len(csvs_in_folder)}: {list(csvs_in_folder)}. Ensure the suffix uniquely identifies the files to be combined. Remember to include the file extension."
            )
        f = csvs_in_folder[0]
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
            if not np.all(ref_df[args.reference_target] == df.y_true):
                raise ValueError(f"The data does not match the reference table\n"
                                 f"ref target:\n{ref_df[args.reference_target]}\ndata y_true:\n"
                                 f"{df.y_true}")
        else:
            if not np.allclose(ref_df[args.reference_target], df.y_true):
                raise ValueError(f"The data does not match the reference table\n"
                                 f"ref target:\n{ref_df[args.reference_target]}\ndata y_true:\n"
                                 f"{df.y_true}")

    # Possible rounding
    if args.around:
        print(f"Rounding values to {args.around} decimals")
        df = df.round(args.around)

    out_fname = out_folder / f"{model_folder.name}_{args.tag}.csv"
    df.to_csv(out_fname, index=False)
    print(f"Done! Saved output to {out_fname}")


def _quantile_mean(series):
    """Returns the mean after values outside the 5th and 95th percentile are removed"""
    if len(series) == 2:
        return series.mean()
    q5 = series.quantile(0.05)
    q95 = series.quantile(0.95)
    return series[(q5 <= series) & (series <= q95)].mean()


def _read_table(fpath):
    if (fpath.endswith(".csv")) or (fpath.endswith(".csv.zip")):
        return pd.read_csv(fpath)
    elif (fpath.endswith(".parquet")) or (fpath.endswith(".parquet.gzip")):
        return pd.read_parquet(fpath)
    else:
        raise ValueError("File extension not supported")


def _group_preds(comb_df, args):
    group_df = comb_df.groupby(args.reference_group)[["y_true", "y_pred"]]

    if args.agg_func == "mode":

        def agg_func(x):
            return pd.Series.mode(x)[0]
    elif args.agg_func == "quantile_mean":
        agg_func = _quantile_mean

    else:
        agg_func = args.agg_func

    group_df = group_df.agg(agg_func)
    return group_df


def _group_logits(comb_df, cols, args):
    y_true = comb_df.groupby(args.reference_group)["y_true"].first()
    group_df = comb_df.groupby(args.reference_group)[cols]
    y_scores = group_df.agg(args.agg_func)
    y_pred = y_scores.idxmax(axis=1).rename("y_pred")
    group_df = pd.concat([y_true, y_pred, y_scores], axis=1)
    return group_df

@dataclass
class GroupPredictionsArgs:
    predictions: str
    reference_csv: str
    reference_target: str
    reference_group: str
    fold: int = None
    fold_col_prefix: str = ""
    set: str = "test"
    group_logits: bool = False
    agg_func: str = "mode"
    suffix: str = ""
    around: int = None

def group_predictions(args: GroupPredictionsArgs):
    out_folder = Path(args.predictions).parents[0]

    csv_stem = Path(args.predictions).stem

    df = _read_table(args.predictions)
    ref_df = _read_table(args.reference_csv)

    if len(ref_df) != len(df):
        if args.fold is None:
            raise ValueError(
                "Predictions and reference don't match."
                " Set a fold parameter if grouping a single fold"
            )
        ref_df = ref_df[
            ref_df[f"{args.fold_col_prefix}{str(args.fold)}"] == args.set
        ].reset_index(drop=True)

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

    if args.group_logits:
        group_df = _group_logits(comb_df, df.columns[3:], args)
    else:
        group_df = _group_preds(comb_df, args)

    # Grouping
    print(f"Grouped {len(df)} rows to {len(group_df)} groups")

    if args.around:
        print(f"Rounding values to {args.around} decimals")
        try:
            group_df = group_df.map(lambda x: np.around(x, args.around))
        except np.core._exceptions._UFuncNoLoopError:
            raise Exception("Can't round values. Only regression tasks can be rounded")

    out_name = out_folder / f"{csv_stem}_grouped{args.suffix}.csv"
    group_df.to_csv(out_name)
    print(f"Saved to {out_name}")

