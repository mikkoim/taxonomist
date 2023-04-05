"""
Combines evaluation results for several models

Input:
    yaml file containing results to be compared in the structure. User input <inside brackets>:
    
    models:
        <model_path01>:
            versions:
                <version01>:
                    short_name: <short name for this model and version>
                    version_tag: <tag for this version separating it from 
                                different length versions, for example after grouping, etc>
                    dataset: <dataset used for calculating results, (not necessarily same as
                                for training the model)>
                <version02>:
                    ...
        <model_path02>:
            ...
    metrics:
        <metric01>:
            minimize: <boolean whether smaller is better for this metric>
        <metric02>:
            ...
                    
Output:
    prints the best performing models
"""

import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import defaultdict, OrderedDict


def print_top_scores(df):
    def print_top(d):
        dd = pd.DataFrame.from_dict(d, orient="index").T.fillna("-")
        col_order = dd.apply(lambda x: np.sum(x != "-")).sort_values(ascending=False)
        print(dd.loc[:, col_order.index].to_markdown(index=False))

    d = defaultdict(list)
    d_ci = defaultdict(list)

    print("## Models ordered on a metric, within the CI of the best model")
    for metric in sorted(conf.metrics.keys()):
        ascending = conf.metrics[metric].minimize
        if ascending:
            comp = lambda x, y: x < y
            q_limit = "q_u"
        else:
            comp = lambda x, y: x > y
            q_limit = "q_l"

        df_sort = df.query("metric==@metric").sort_values(
            by="value", ascending=ascending
        )

        within_ci = df_sort[comp(df_sort["value"], df_sort.iloc[0][q_limit])]
        print(metric)
        print(
            within_ci[["name", "value", "version", "q_l", "q_u"]].to_markdown(
                index=False
            )
        )
        print()

        d[df_sort.iloc[0]["name"]].append(metric)
        for n in within_ci.name:
            d_ci[n].append(metric)

    print()
    print("## Overview")
    print_top(d_ci)
    print()

    print("## Only the best models")
    print_top(d)

    return d, d_ci


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--print_versions", action="store_true")

    args = parser.parse_args()

    folder = Path(args.root_dir)

    if args.out_folder:
        out_folder = Path(args.out_folder)
        out_pred = out_folder / "predictions"
        out_metr = out_folder / "metrics"

        out_pred.mkdir(exist_ok=True, parents=True)
        out_metr.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.load(args.config)
    if args.print_config:
        print(OmegaConf.to_yaml(conf))

    # Go through models in conf
    dfs = []
    dfs_pred = defaultdict(lambda: defaultdict(list))
    versions = set()
    version_tags = set()
    for model in conf.models.keys():
        metric_folder = folder / model / "metrics"
        pred_folder = folder / model / "predictions"

        # Go through different prediction versions
        for version in conf.models[model].versions.keys():
            vdict = conf.models[model].versions[version]
            name = vdict["short_name"]
            version_tag = vdict["version_tag"]
            dataset = vdict["dataset"]

            # Read metrics
            df0 = pd.read_csv(
                metric_folder / f"metrics_{Path(model).name}_{version}.csv"
            )
            df0["version"] = version
            df0["name"] = name
            df0["version_tag"] = version_tag
            df0["dataset"] = dataset
            dfs.append(df0)

            # Read predictions
            df_pred = pd.read_csv(pred_folder / f"{Path(model).name}_{version}.csv")
            df_pred = df_pred.rename({"y_pred": name}, axis=1)
            dfs_pred[dataset][version_tag].append(df_pred)

            versions.add(version)
            version_tags.add(version_tag)

    # combine predictions to a single dataframe
    for dataset in dfs_pred.keys():
        for version_tag in version_tags:
            if len(dfs_pred[dataset][version_tag]) != 0:
                df_ds_v = pd.concat(dfs_pred[dataset][version_tag], axis=1)
                df_ds_v = df_ds_v.loc[:, ~df_ds_v.columns.duplicated()].copy()

                # Save preds
                if args.out_folder:
                    out_fname = out_pred / (
                        Path(args.config).stem + f"_{dataset}_{version_tag}.csv"
                    )
                    df_ds_v.to_csv(out_fname, index=False)
                    print(out_fname)

    # Save combined evaluation dataframes
    df = pd.concat(dfs).reset_index(drop=True)
    if args.out_folder:
        out_fname = out_metr / (Path(args.config).stem + ".csv")
        df.to_csv(out_fname, index=False)
        print(out_fname)

    # Print comparison of models
    df_full = df.query("fold=='full'")
    df_full_pivot = df_full.pivot(
        values="value", index=["dataset", "version_tag", "name"], columns="metric"
    )
    print(df_full_pivot.to_string())

    print("\nPrinting results for all models")
    d = print_top_scores(df_full)
    print("\n")

    if args.print_versions:
        print("Printing results for each version tag separately:")
        for version_tag in df_full["version_tag"].unique():
            df_tmp = df_full.query("version_tag==@version_tag")
            print(f"\n##### VERSION TAG: {version_tag}")
            d_tmp = print_top_scores(df_tmp)
