# The Python code for the script without markdown and ready to be saved as a .py file

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from imblearn.metrics import geometric_mean_score
import os
import argparse

def calculate_metrics(predictions_path, out_folder, out_prefix):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(predictions_path)

    # Extract the true and predicted labels
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values

    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
   
    print(accuracy)
    # Calculate F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f1)
    # Encode labels for ROC AUC calculation
    lb = LabelBinarizer()
    y_true_encoded = lb.fit_transform(y_true)
    y_pred_encoded = lb.transform(y_pred)

    # Handle binary and multiclass cases for ROC AUC
    if y_true_encoded.shape[1] == 1:
        roc_auc = roc_auc_score(y_true_encoded, y_pred_encoded)
    else:
        roc_auc = roc_auc_score(y_true_encoded, y_pred_encoded, average='macro', multi_class='ovo')
    print(roc_auc)
    # Calculate G-Mean
    g_mean = geometric_mean_score(y_true, y_pred, average='weighted')
    print(g_mean)



    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'ROC AUC', 'G-Mean'],
        'Value': [accuracy, f1, roc_auc, g_mean]
    })

    # Define the output file path
    output_file_path = os.path.join(out_folder, f"{out_prefix}_evaluation_metrics.csv")

    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv(output_file_path, index=False)
    print(f"Saved the evaluation metrics to {output_file_path}")

# The following is for script execution only and won't run in the notebook.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True, help='CSV file with predictions.')
    parser.add_argument('--out_folder', type=str, required=True, help='Directory to save the output file.')
    parser.add_argument('--out_prefix', type=str, default='metrics', help='Prefix for the output file.')

    args = parser.parse_args()
    calculate_metrics(args.predictions, args.out_folder, args.out_prefix)

