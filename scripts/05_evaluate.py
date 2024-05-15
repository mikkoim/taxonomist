import pandas as pd
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
from pprint import pprint
from joblib import Parallel, delayed

import sklearn.metrics

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,multilabel_confusion_matrix, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from imblearn.metrics import geometric_mean_score
import os

DESCRIPTION = """
Calculates metrics to prediction outputs.

Input:
A csv file containing true and predicted labels
A config file containing metrics that are calculated

Output:
A dataframe containing 
    metrics
    bootstrapped confidence intervals
for
    full cv predictions
    each fold separately
"""
lb = LabelBinarizer()

def load_metric(metric):
    # Regression
    if metric == "mse":
        return sklearn.metrics.mean_squared_error

    elif metric == "rmse":
        return lambda y, yhat: sklearn.metrics.mean_squared_error(
            y, yhat, squared=False
        )

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
            if np.any(df.y_pred < 0):
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
    
    elif metric == "roc_auc":
        # Return a function that calculates ROC AUC when called
        def roc_auc_func(y_true, y_pred):
            y_true_encoded = lb.fit_transform(y_true)
            y_pred_encoded = lb.transform(y_pred)
            if y_true_encoded.shape[1] == 1:
                # Binary classification
                return roc_auc_score(y_true_encoded, y_pred_encoded[:, 1])
            else:
                # Multiclass classification
                return roc_auc_score(y_true_encoded, y_pred_encoded, average='macro', multi_class='ovo')
        return roc_auc_func

    elif metric == "g_mean":
        return lambda y, yhat: geometric_mean_score(y, yhat, average='macro')

    elif metric == "bacc":
        return lambda y, yhat: balanced_accuracy_score(y, yhat)

    elif metric == "mcc":
        def mcc_multiclass(y_true, y_pred):
            # Calculate the multilabel confusion matrix
            mcm = multilabel_confusion_matrix(y_true, y_pred)
            mccs = []

            for i in range(len(mcm)):
                tn, fp, fn, tp = mcm[i].ravel()
                # Calculate MCC for each class and handle edge cases
                with np.errstate(all='ignore'):
                    mcc_num = (tp * tn) - (fp * fn)
                    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    mcc = mcc_num / mcc_den if mcc_den != 0 else 0
                    mccs.append(mcc)
                    
            # Calculate the mean MCC across all classes
            mcc_mean = np.mean(mccs)
            return mcc_mean
        return mcc_multiclass


def calc_metrics(df):
    values = {}
    for metric in conf.metrics:
        func = load_metric(metric)
        values[metric] = func(df.y_true, df.y_pred)
    return values


def calc_bootstrap(df, n_repeats, alpha=0.95):
    """Test set bootstrapping"""

    def _bootstrap(df):
        bs = df.sample(n=len(df), replace=True)
        return calc_metrics(bs)

    bs_value_list = Parallel(n_jobs=4)(
        delayed(_bootstrap)(df) for _ in tqdm(range(n_repeats))
    )

    bs_values = pd.DataFrame(bs_value_list).melt(var_name="metric")

    bs_errors = {}
    for metric in conf.metrics:
        # err = np.quantile(np.abs(values[metric] - bs_values.query("metric==@metric")['value'].values), alpha)
        err = np.quantile(
            bs_values.query("metric==@metric")["value"].values, [1 - alpha, alpha]
        )
        bs_errors[metric] = err

    return bs_errors, bs_values


def print_fewest_classes_info(df, class_counts, confusion_matrix):
    # Get unique classes in the order they appear in the confusion matrix
    unique_classes = np.unique(df['y_true'])
    fewest_samples_classes = class_counts.nsmallest(10).index.tolist()

    print("Fewest Samples Classes Information:")
    for cls in fewest_samples_classes:
        class_index = np.where(unique_classes == cls)[0][0]  # Accurate index retrieval
        total_samples = class_counts.loc[cls]
        correctly_classified = confusion_matrix[class_index, class_index]
        print(f"Class: {cls}, Total Samples: {total_samples}, Correctly Classified: {correctly_classified}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument("--predictions", type=str)
    parser.add_argument("--metric_config", type=str)

    parser.add_argument("--reference_csv", default=None, type=str)
    parser.add_argument("--reference_target", default=None, type=str)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--n_bootstrap", default=1000)
    parser.add_argument("--bootstrap_alpha", default=0.95)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="metrics")
    parser.add_argument("--around", default=4, type=int)

    args = parser.parse_args()

    csv_path = Path(args.predictions)  

    # Ensure out_folder is defined at this point
    csv_stem = csv_path.stem
    out_folder = csv_path.parents[1] / "metrics"
    out_folder.mkdir(exist_ok=True, parents=True) 

    # Load metric config
    conf = OmegaConf.load(args.metric_config)
    print(OmegaConf.to_yaml(conf))

    # Load predictions and calculate metrics
    df = pd.read_csv(csv_path)
    values = calc_metrics(df)
    if not args.no_bootstrap:
        bs_errors, bs_values = calc_bootstrap(
            df, args.n_bootstrap, args.bootstrap_alpha
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
            row["q_l"] = bs_errors[metric][0]
            row["q_u"] = bs_errors[metric][1]
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

            values = calc_metrics(df_fold)
            if not args.no_bootstrap:
                bs_errors, _ = calc_bootstrap(
                    df_fold, args.n_bootstrap, args.bootstrap_alpha
                )
            for metric in conf.metrics:
                row = {}
                row["fold"] = str(fold)
                row["metric"] = metric
                row["value"] = values[metric]
                if not args.no_bootstrap:
                    row["q_l"] = bs_errors[metric][0]
                    row["q_u"] = bs_errors[metric][1]
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

        if not args.no_save:
            csv_stem = csv_path.stem
            out_folder = csv_path.parents[1] / "metrics"
            out_folder.mkdir(exist_ok=True, parents=True)

            out_fname = out_folder / f"{args.out_prefix}_{csv_stem}.csv"
            results.to_csv(out_fname, index=False)
            print(f"Saved to {out_fname}")

    # Code for classes with fewest samples
    
    class_counts = df['y_true'].value_counts()
    fewest_samples_classes = class_counts.nsmallest(10).index.tolist()
    df_fewest_samples = df[df['y_true'].isin(fewest_samples_classes)]
    fewest_samples_values = calc_metrics(df_fewest_samples)
    fewest_samples_results = pd.DataFrame({
        "metric": metric_name,
        "value": metric_value
    } for metric_name, metric_value in fewest_samples_values.items())

    fewest_samples_out_fname = out_folder / f"{args.out_prefix}_{csv_stem}_minor_classes.csv"
    fewest_samples_results.to_csv(fewest_samples_out_fname, index=False)
    print(f"Saved minor classes metrics to {fewest_samples_out_fname}")


    # Calculate the confusion matrix with sorted unique labels
    unique_classes = np.sort(df['y_true'].unique())
    y_true = df['y_true'].apply(lambda x: np.where(unique_classes == x)[0][0]).tolist()
    y_pred = df['y_pred'].apply(lambda x: np.where(unique_classes == x)[0][0]).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(unique_classes)))  # Specify labels explicitly

    # Calculate class counts
    class_counts_minor = df['y_true'].value_counts()

    # Now print fewest classes info 
    print_fewest_classes_info(df, class_counts_minor, cm)

    # Write the DataFrame to a CSV file
    df_fewest_classes_info.to_csv('fewest_classes_info.csv', index=False)
    """
    #For 5 least miniority classes

    # Assuming `df` contains the true labels in a column named 'y_true'

    #Step 1
    # Count the number of samples per class
    class_counts = df['y_true'].value_counts()

    # Get the 5 classes with the fewest samples
    fewest_samples_classes = class_counts.nsmallest(5).index.tolist()

    #STEP 2
    # Filter the dataframe for rows where the class is one of the five with the fewest samples
    df_fewest_samples = df[df['y_true'].isin(fewest_samples_classes)]

    # Calculate metrics only for the filtered dataframe
    fewest_samples_values = calc_metrics(df_fewest_samples)

    #STEP 3
    fewest_samples_results = pd.DataFrame({
        "metric": metric_name,
        "value": metric_value
    } for metric_name, metric_value in fewest_samples_values.items())

    fewest_samples_out_fname = out_folder / f"{args.out_prefix}_{csv_stem}_minor_classes.csv"
    fewest_samples_results.to_csv(fewest_samples_out_fname, index=False)
    print(f"Saved minor classes metrics to {fewest_samples_out_fname}")
    #END
    """
