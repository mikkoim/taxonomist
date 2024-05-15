# from pathlib import Path
# import pandas as pd
# import argparse

# def create_class_imbalance(df, label_column):
#     """
#     Reduces the number of samples for each class to create an artificial class imbalance.
#     Args:
#         df (DataFrame): The original dataframe.
#         label_column (str): The column name containing the labels.
#     Returns:
#         DataFrame: A new dataframe with reduced class samples.
#     """
#     class_counts = df[label_column].value_counts().sort_index()
#     imbalance_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 1.0]  # Adjust as needed for the number of classes
#     imbalance_factors = imbalance_factors[:len(class_counts)][::-1]  # Adjust list size and reverse to start reduction from the last class

#     # Sample each class based on the specified imbalance factor
#     new_frames = []
#     for (label, count), factor in zip(class_counts.items(), imbalance_factors):
#         num_samples = int(count * factor)
#         sampled_df = df[df[label_column] == label].sample(n=num_samples, random_state=42)
#         new_frames.append(sampled_df)

#     return pd.concat(new_frames, ignore_index=True)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv_path", type=str, required=True)
#     parser.add_argument("--out_folder", type=str, default=".")
    
#     args = parser.parse_args()

#     csv_path = Path(args.csv_path)
#     out_folder = Path(args.out_folder)
#     out_folder.mkdir(exist_ok=True, parents=True)

#     df = pd.read_csv(args.csv_path)

#     # Assuming 'Label' is the column containing class labels
#     df_imbalanced = create_class_imbalance(df, 'Label')

#     # Save the labels to a label map, preserving the original order of appearance
#     unique_labels = df['Label'].unique()
#     label_map_path = out_folder / "cifar10_label_map.txt"
#     with open(label_map_path, "w") as f:
#         for label in unique_labels:
#             f.write(label + "\n")

#     out_fname = out_folder / "01_cifar10_processed_imbalanced.csv"
#     df_imbalanced.to_csv(out_fname, index=False)
#     print(out_fname)


from pathlib import Path
import pandas as pd
import argparse

def load_or_create_label_map(df, label_column, out_folder):
    """
    Loads or creates a label map from the DataFrame or an existing file.
    Args:
        df (DataFrame): The DataFrame from which to derive class labels.
        label_column (str): The column name containing the labels.
        out_folder (Path): The output folder path where the label map is saved.
    Returns:
        list: A list of class labels in the order they should be used.
    """
    label_map_path = out_folder / "cifar10_label_map.txt"
    if label_map_path.exists():
        with open(label_map_path, 'r') as f:
            labels = [line.strip() for line in f]
    else:
        labels = df[label_column].unique()
        with open(label_map_path, "w") as f:
            for label in labels:
                f.write(label + "\n")
    return labels

def create_class_imbalance(df, label_column, labels):
    """
    Reduces the number of samples for each class to create an artificial class imbalance.
    Args:
        df (DataFrame): The original dataframe.
        label_column (str): The column name containing the labels.
        labels (list): A list of labels in the specific order to be followed.
    Returns:
        DataFrame: A new dataframe with reduced class samples.
    """
    class_counts = df[label_column].value_counts()
    imbalance_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # Must be adjusted based on number of classes
    # Ensure the imbalance factors list matches the number of classes
    imbalance_factors = imbalance_factors[:len(labels)][::-1]  # Reverse to start reduction from the last class

    new_frames = []
    for label, factor in zip(reversed(labels), imbalance_factors):
        num_samples = int(class_counts[label] * factor)
        sampled_df = df[df[label_column] == label].sample(n=num_samples, random_state=42)
        new_frames.append(sampled_df)

    return pd.concat(new_frames, ignore_index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_folder", type=str, default=".")
    
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.csv_path)

    # Load or create the label map, ensuring class order is maintained
    labels = load_or_create_label_map(df, 'Label', out_folder)

    # Create class imbalance following the specific order
    df_imbalanced = create_class_imbalance(df, 'Label', labels)

    out_fname = out_folder / "01_cifar10_processed_imbalanced.csv"
    df_imbalanced.to_csv(out_fname, index=False)
    print(out_fname)
