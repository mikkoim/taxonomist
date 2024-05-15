import argparse
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix, output_file):
    plt.figure(figsize=(20, 14))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(output_file)

def save_conf_matrix_csv(conf_matrix, output_file_csv):
    df_cm = pd.DataFrame(conf_matrix)
    df_cm.to_csv(output_file_csv, index=False)
    print(f"Confusion matrix saved to CSV file: {output_file_csv}")

# Argument parsing
parser = argparse.ArgumentParser(description='Plot and save confusion matrix from predictions.')
parser.add_argument('--predictions', type=str, help='Path to the CSV file with predictions.')
args = parser.parse_args()

# Data loading
df = pd.read_csv(args.predictions)
y_true = df['y_true']
y_pred = df['y_pred']

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Define the output directory with a 'conf_matrix' subfolder
output_dir = os.path.join(os.path.dirname(args.predictions), 'conf_matrix')
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Construct output filenames within the 'conf_matrix' subfolder
output_file_plot = os.path.join(output_dir, "conf_matrix_plot.png")
output_file_csv = os.path.join(output_dir, "conf_matrix.csv")

# Plot and save the confusion matrix plot
plot_confusion_matrix(conf_matrix, output_file_plot)

# Save the confusion matrix to a CSV file
save_conf_matrix_csv(conf_matrix, output_file_csv)

