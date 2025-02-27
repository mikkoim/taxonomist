from pathlib import Path

# All imports are optional, and you can use any other library you prefer
import pandas as pd
from tqdm import tqdm
from taxonomist.data import make_webdataset
from torch import nn
import open_clip
from datasets import load_dataset

"""
Defines custom functions for reading dataset data from train-test-splitted csv-files,

Also defines possible custom models for training.
"""


def preprocess_dataset(data_folder, dataset_name, csv_path=None, fold=None, label=None):
    """Returns a filepath dataset. MODIFY ONLY SELECT PARTS.

    How to use: Define a new elif branch with loading instructions for your dataset.
    The function must return two dicts that contain the filenames and labels.

    Labels are returned as their string representations. Mapping to indices is handled
    with the class_map defined elsewhere.

    fnames: {"train": <list of trainset filepaths>,
             "val": <list of validation filepaths>,
             "test": <list of testing filepaths>}

    labels: {"train": <labels corresponding to above, as strings>,
             "val": ...,
             "test": ...}
    """
    data_folder = Path(data_folder)
    fnames = {}
    labels = {}
    for set_ in ["train", "val", "test"]:
        if dataset_name == "rodi":
            fnames[set_], labels[set_] = process_split_csv_rodi(
                data_folder, csv_path, set_, fold, label
            )
        elif dataset_name == "finbenthic2":
            fnames[set_], labels[set_] = process_split_csv_finbenthic2(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "finbenthic1":
            fnames[set_], labels[set_] = process_split_csv_finbenthic1(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "biodiscover":
            """
            You should change the name of the dataset to match the true name 
            of your BioDiscover dataset
            """
            fnames[set_], labels[set_] = process_split_csv_biodiscover(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "my_dataset":
            """

            YOUR CODE GOES HERE

            """
            fnames[set_], labels[set_] = [None, None], [None, None]

        else:
            raise Exception("Unknown dataset name")

    return fnames, labels

def return_custom_datasets(data_folder: str,
                           dataset_name: str, 
                           csv_path: str = None,
                           fold: int = None,
                           label: str = None,
                           label_transform: callable = None,
                           transforms: callable = None):
    """Returns a custom dataset. MODIFY ONLY SELECT PARTS.

    How to use: Define a new elif-branch for your dataset. Assumes the branch defines 
    dataset objects in a dictionary structure of form:
        datasets = {"train": my_train_dataset,
                    "val": my_val_dataset,
                    "test": my_test_dataset}
    The dictionary keys must be exactly as above.

    Passed values:

    data_folder (str): A string for the data folder or other data source
    dataset_name (str): A string for the dataset name, used to select the correct branch
    csv_path (str): A string for the path to a csv or other metadata file.
    fold (int): An integer for the fold number
    label (str): A string for the label column name
    label_transform (callable): A function for transforming the label. Passed from
        the main script, and created from the class_map.
    transforms (callable): A function for transforming the image data. Passed from
        the main script, and created from the data augmentation settings.
    """

    if dataset_name == "rodi-wds":
        datasets = process_wds_dataset(
            data_folder, csv_path, fold, label, label_transform, transforms
        )
    elif dataset_name == "aquamonitor-jyu-regression":
        datasets = process_aquamonitor_jyu_regression(label, transforms)

    elif dataset_name == "my_custom_dataset":
        # Define your dataset here
        datasets = {"train": None, "val": None, "test": None}
    else:
        raise Exception("Unknown dataset name")
    
    assert "train" in datasets.keys()
    assert "test" in datasets.keys()
    assert "val" in datasets.keys()
    return datasets

def return_custom_model(model_name: str,
                         freeze_base: bool,
                         pretrained: bool,
                         n_classes: int):
    """Returns a custom model.

    How to use: Define a new elif-branch for your model. Assumes the branch defines
    a model object that can be used as a classifier. The model must output logits in the
    shape of (batch_size, n_classes).

    Passed values:

    model_name (str): A string for the model name
    freeze_base (bool): A boolean for freezing the base model
    pretrained (bool): A boolean for using pretrained weights (if applicable)
    n_classes (int): An integer for the number of classes
    """
    if model_name == "bioclip":
        model = create_bioclip_model(model_name='hf-hub:imageomics/bioclip',
                                     freeze_base=freeze_base,
                                     n_classes=n_classes)
    elif model_name == "my_custom_model":
        """
        
        YOUR CODE GOES HERE
        
        """
        model = None
    return model

################# USER DEFINED FUNCTIONS ###############################

class OpenClipModel(nn.Module):
    """Wrapper for a OpenClip model with a projection head"""
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        freeze_base: bool = True,
        h_dim: int = 512,
        n_classes: int = 1,
    ):
        super().__init__()
        self.base_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.h_dim = h_dim

        if freeze_base:
            self.freeze_base()

        self.init_proj_head(n_classes)

    def init_proj_head(self, n_classes):
        self.proj_head = nn.Sequential(nn.Linear(self.h_dim, n_classes))

    def freeze_base(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Base model frozen")

    def forward(self, x):
        h = self.base_model.encode_image(x)
        return self.proj_head(h)

    def base_forward(self, x):
        return self.base_model.encode_image(x)

    def proj_forward(self, h):
        return self.proj_head(h)


def create_bioclip_model(model_name, freeze_base, n_classes):
    model = OpenClipModel(model_name=model_name,
                          freeze_base=freeze_base,
                          n_classes=n_classes
                          )
    return model

def process_wds_dataset(data_folder,
                             csv_path,
                             fold,
                             label,
                             label_transform,
                             transforms):

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.removesuffix(".png")
    label_dict = dict(zip(df["image"], df[label]))

    ds = make_webdataset(data_folder=data_folder,
                         df=df,
                         image_column="image",
                         image_key="png",
                         fold_column=str(fold),
                         filename2label=label_dict,
                         batch_size=256,
                         shuffle_buffer=1000,
                         label_transform=label_transform,
                         transforms=transforms)

    return ds

def process_aquamonitor_jyu_regression(label, transforms):
    import torch
    ds = load_dataset("mikkoim/aquamonitor-jyu", cache_dir="huggingface_cache")
    metadata = pd.read_parquet("https://huggingface.co/datasets/mikkoim/aquamonitor-jyu/resolve/main/aquamonitor-jyu.parquet.gzip")

    metadata["img"] = metadata["img"].str.removesuffix(".jpg")
    label_dict = dict(zip(metadata["img"], metadata[label]))

    tf_train = transforms["train"]
    tf_test = transforms["test"]

    def train_transform(batch):
        return {"fname": batch["__key__"],
                "x": [tf_train(x) for x in batch["jpg"]],
                "y": torch.as_tensor([label_dict[x] for x in batch["__key__"]], dtype=torch.float32)}

    def test_transform(batch):
        return {"fname": batch["__key__"],
                "x": [tf_test(x) for x in batch["jpg"]],
                "y": torch.as_tensor([label_dict[x] for x in batch["__key__"]], dtype=torch.float32)}
    
    ds_train = ds["train"].with_transform(train_transform)
    ds_val = ds["validation"].with_transform(test_transform)
    return {"train": ds_train, "val": ds_val, "test": ds_val}


def process_split_csv_aquamonitor(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_parquet(csv_path)
    df = df0[df0[f"fold{fold}"] == set_]

    fnames = df["img"].apply(lambda x: Path(data_folder, f"{x[:-4]}.jpg")).values

    labels = df[label].values.tolist()

    assert len(fnames) == len(labels)
    print(f"{set_} size: {len(fnames)}")
    print(f"Checking '{set_}' filenames exist...")
    print("Done.")

    return fnames, labels

def process_split_csv_biodiscover(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(
            data_folder,
            x["Species Name"],
            x["individual"],
            x["Sample Name/Number"],
            x["Image File Name"][:-4] + ".jpg",
        ),
        axis=1,
    ).values

    labels = df[label].values

    return fnames, labels


def process_split_csv_finbenthic1(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(data_folder, "Cropped images", x["taxon"], x["img"]).resolve(),
        axis=1,
    ).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels


def process_split_csv_finbenthic2(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(data_folder, "Images", x["individual"], x["img"]).resolve(),
        axis=1,
    ).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels


def process_split_csv_rodi(data_folder, csv_path, set_, fold, label):
    "RODI -specific function for reading train-test-split csvs"
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df["image"].apply(lambda x: data_folder.resolve() / x).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels
