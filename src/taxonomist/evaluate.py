from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import sklearn.metrics
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict

def load_metric(metric):
    # Regression
    if metric == "mse":
        return sklearn.metrics.mean_squared_error

    elif metric == "rmse":
        return sklearn.metrics.root_mean_squared_error

    elif metric == "mae":
        return sklearn.metrics.mean_absolute_error

    elif metric == "mape":
        return sklearn.metrics.mean_absolute_percentage_error

    elif metric == "mdape":
        return lambda y, yhat: np.median(np.abs(y - yhat) / y)

    elif metric == "r2":
        return sklearn.metrics.r2_score

    elif metric == "log-r2":

        def logr2(y, yhat):
            if np.any(yhat < 0):
                return 0
            else:
                return sklearn.metrics.r2_score(np.log(y), np.log(yhat))

        return logr2

    # Classification
    elif metric == "accuracy":
        return sklearn.metrics.accuracy_score

    elif metric == "precision_macro":
        return lambda y, yhat: sklearn.metrics.precision_score(
            y, yhat, average="macro", zero_division=False
        )
    elif metric == "precision_micro":
        return lambda y, yhat: sklearn.metrics.precision_score(
            y, yhat, average="micro", zero_division=False
        )
    elif metric == "precision_weighted":
        return lambda y, yhat: sklearn.metrics.precision_score(
            y, yhat, average="weighted", zero_division=False
        )

    elif metric == "recall_macro":
        return lambda y, yhat: sklearn.metrics.recall_score(
            y, yhat, average="macro", zero_division=False
        )
    elif metric == "recall_micro":
        return lambda y, yhat: sklearn.metrics.recall_score(
            y, yhat, average="micro", zero_division=False
        )
    elif metric == "recall_weighted":
        return lambda y, yhat: sklearn.metrics.recall_score(
            y, yhat, average="weighted", zero_division=False
        )

    elif metric == "f1_macro":
        return lambda y, yhat: sklearn.metrics.f1_score(
            y, yhat, average="macro", zero_division=False
        )
    elif metric == "f1_micro":
        return lambda y, yhat: sklearn.metrics.f1_score(
            y, yhat, average="micro", zero_division=False
        )
    elif metric == "f1_weighted":
        return lambda y, yhat: sklearn.metrics.f1_score(
            y, yhat, average="weighted", zero_division=False
        )


def calc_metrics(df, metrics):
    values = {}
    for metric in metrics:
        func = load_metric(metric)
        values[metric] = func(df.y_true, df.y_pred)
    return values


def calc_bootstrap(df, metrics, n_repeats, alpha=0.95):
    """Test set bootstrapping"""

    def _bootstrap(df):
        bs = df.sample(n=len(df), replace=True)
        return calc_metrics(bs, metrics)

    bs_value_list = Parallel(n_jobs=4)(
        delayed(_bootstrap)(df) for _ in tqdm(range(n_repeats))
    )

    bs_values = pd.DataFrame(bs_value_list).melt(var_name="metric")

    bs_errors = {}
    for metric in conf.metrics:
        # err = np.quantile(np.abs(values[metric] - bs_values.query("metric==@metric")['value'].values), alpha)
        values = bs_values.query("metric==@metric")["value"].values
        q_err = np.quantile(values, [1 - alpha, alpha])
        std = np.std(values)
        bs_errors[metric] = {}
        bs_errors[metric]["q"] = q_err
        bs_errors[metric]["std"] = std

    return bs_errors, bs_values

@dataclass
class EvaluateArgs:
    predictions: str
    metric_config: str
    reference_csv: str
    reference_target: str
    n_folds: int
    n_bootstrap: int
    no_bootstrap: bool
    bootstrap_alpha: float
    no_save: bool
    out_prefix: str
    around: int

def evaluate(args: EvaluateArgs):
    csv_path = Path(args.predictions)

    # Load metric config
    conf = OmegaConf.load(args.metric_config)
    print(OmegaConf.to_yaml(conf))

    # Load predictions and calculate metrics
    df = pd.read_csv(csv_path)
    values = calc_metrics(df, conf.metrics)
    if not args.no_bootstrap:
        bs_errors, bs_values = calc_bootstrap(
            df=df,
            metrics=conf.metrics,
            n_repeats=args.n_bootstrap,
            alpha=args.bootstrap_alpha,
        )
    else:
        bs_errors = None
        bs_values = None

    pprint(values)

    # Save values to a dataframe
    row_list = []
    for metric in conf.metrics:
        row = {}
        row["fold"] = "full"
        row["metric"] = metric
        row["value"] = values[metric]
        if not args.no_bootstrap:
            row["q_l"] = bs_errors[metric]["q"][0]
            row["q_u"] = bs_errors[metric]["q"][1]
            row["std"] = bs_errors[metric]["std"]
            assert row["q_l"] <= row["value"]
            assert row["value"] <= row["q_u"]
        row_list.append(row)
    results = pd.DataFrame(row_list)

    # If a reference csv is provided, calculate metrics for each cv fold
    if args.reference_csv:
        print("Using reference csv for cross-validation fold metrics")
        ref_df = pd.read_csv(args.reference_csv)
        assert np.allclose(ref_df[args.reference_target], df.y_true)

        row_list = []
        for fold in range(args.n_folds):
            idx = ref_df[ref_df[str(fold)] == "test"].index.values
            df_fold = df.iloc[idx]

            values = calc_metrics(df_fold, conf.metrics)
            if not args.no_bootstrap:
                bs_errors, _ = calc_bootstrap(
                    df=df_fold,
                    metrics=conf.metrics,
                    n_repeats=args.n_bootstrap,
                    alpha=args.bootstrap_alpha,
                )
            for metric in conf.metrics:
                row = {}
                row["fold"] = str(fold)
                row["metric"] = metric
                row["value"] = values[metric]
                if not args.no_bootstrap:
                    row["q_l"] = bs_errors[metric]["q"][0]
                    row["q_u"] = bs_errors[metric]["q"][1]
                    row["std"] = bs_errors[metric]["std"]
                    assert row["q_l"] <= row["value"]
                    assert row["value"] <= row["q_u"]
                row_list.append(row)

        results_cv = pd.DataFrame(row_list)
        results = pd.concat((results, results_cv))

    if args.around:
        print(f"Rounding values to {args.around} decimals")
        results["value"] = results["value"].apply(lambda x: np.around(x, args.around))
        if not args.no_bootstrap:
            results["q_l"] = results["q_l"].apply(lambda x: np.around(x, args.around))
            results["q_u"] = results["q_u"].apply(lambda x: np.around(x, args.around))
            results["std"] = results["std"].apply(lambda x: np.around(x, args.around))

    if not args.no_save:
        csv_stem = csv_path.stem
        out_folder = csv_path.parents[1] / "metrics"
        out_folder.mkdir(exist_ok=True, parents=True)

        out_fname = out_folder / f"{args.out_prefix}_{csv_stem}.csv"
        results.to_csv(out_fname, index=False)
        print(f"Saved to {out_fname}")

@dataclass
class CompareArgs:
    config: str
    out_folder: str
    print_config: bool
    print_versions: bool

def compare(args: CompareArgs):
    if args.out_folder:
        out_folder = Path(args.out_folder)
        out_pred = out_folder / "predictions"
        out_metr = out_folder / "metrics"

        out_pred.mkdir(exist_ok=True, parents=True)
        out_metr.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.load(args.config)
    if args.print_config:
        print(OmegaConf.to_yaml(conf))

    # Create tags
    tag_set = set()
    for model, model_conf in conf.models.items():
        for tag in model_conf["tags"].keys():
            tag_set.add(tag)

    # Read metrics and predictions
    dfs = []
    dfs_pred = defaultdict(lambda: defaultdict(list))
    length_tags = set()
    for model, model_conf in conf.models.items():
        dataset = model_conf["dataset"]
        length = model_conf["length"]

        # Read metrics
        df_metrics = pd.read_csv(model_conf["metrics"])
        df_metrics["name"] = model
        df_metrics["dataset"] = dataset
        df_metrics["length"] = length
        for tag, value in model_conf["tags"].items():
            df_metrics[tag] = value
        dfs.append(df_metrics)

        # Read predictions
        df_pred = pd.read_csv(model_conf["predictions"])[["y_true", "y_pred"]]
        df_pred = df_pred.rename({"y_pred": model}, axis=1)
        # Different lengths of datasets and predictions (for example groupings)
        # are taken care of with dataset and length tags
        dfs_pred[dataset][length].append(df_pred)
        length_tags.add(length)

    # combine predictions to a single dataframe
    for dataset, ds_preds in dfs_pred.items():
        for length in length_tags:
            if len(ds_preds[length]) != 0:
                df_ds_l = pd.concat(dfs_pred[dataset][length], axis=1)

                # Remove duplicate columns
                df_ds_l = df_ds_l.loc[:, ~df_ds_l.columns.duplicated()].copy()

                # Save preds
                if args.out_folder:
                    out_fname = out_pred / (
                        Path(args.config).stem + f"_{dataset}_{length}.csv"
                    )
                    df_ds_l.to_csv(out_fname, index=False)
                    print(out_fname)

    # Save combined evaluation dataframes
    df = pd.concat(dfs).reset_index(drop=True)
    if args.out_folder:
        out_fname = out_metr / (Path(args.config).stem + ".csv")
        df.to_csv(out_fname, index=False)

    df_full = df.query("fold =='full'")
    df_full_pivot = df_full.pivot(
        values="value", index=["dataset", "length", "name"], columns="metric"
    )
    print(df_full_pivot.to_string())