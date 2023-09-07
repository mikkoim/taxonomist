import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from collections import defaultdict

DESCRIPTION = """
Combines evaluation results for several models

Input:
    yaml file containing results to be compared in the structure. User input <inside brackets>:

    models:
        <model-1-name>:
            predictions: <path to model-1 predictions>
            metrics: <path to model-1 metrics>
            dataset: <The dataset that was used for testing (NOT training).
                        Used to differentiate different lenght outputs>
            length: <a length tag that differentiates different length outputs.
                     for example: "grouped" or "separate">
            tags:
                <tag-1>: <Tags are arbitary and can have arbitary values>
                <tag-2>: <tag 2 value>
        <model-2-name>:
            predictions:
            metrics:
            dataset:
            length:
                    
Output:
    Creates two sub-folders to out_folder: predictions and metrics. Predictions are grouped together
    by dataset-length pairs
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument("--config", type=str)
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--print_versions", action="store_true")

    args = parser.parse_args()

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
