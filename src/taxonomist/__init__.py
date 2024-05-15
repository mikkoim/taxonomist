import argparse
import bisect
import os
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path

import albumentations as A
import numpy as np
import PIL.Image as Image
import lightning.pytorch as pl
import pandas as pd
import scipy.stats
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from tqdm import tqdm

from .datasets import preprocess_dataset


def add_dataset_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Folder where the data can be found. This folder is "
        "used with the csv_path to produce final filenames for training",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The dataset name that is used to select the function that "
        "determines how data should be loaded",
        required=True,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the csv file that contains dataset table and label "
        "information for each sample. "
        "Used along data_folder to produce final filenames for training. "
        "The csv should contain train-test-validation split info for all "
        "cross-validation folds",
        required=True,
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Label column. Found from the csv_path file.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="The fold that is used for training. Found from the csv_path file.",
        required=True,
    )
    parser.add_argument(
        "--class_map",
        type=str,
        help="Refers to a list of classes found in the dataset. Provides "
        "an unambiguous reference between strings and indices, even if some folds "
        "don't contain all classes",
        default=None,
        required=False,
    )
    return parser


def add_dataloader_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--imsize", type=int, help="Inputs are resized to this size", default=None
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=None)
    parser.add_argument(
        "--aug",
        type=str,
        help="The augmentation method to be used",
        default="only-flips",
    )
    parser.add_argument(
        "--load_to_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Whether the full dataset is loaded to memory. Might be faster for "
        "systems with slow disks.",
        default=False,
    )
    parser.add_argument(
        "--tta",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Test-time augmentation is applied",
        default=False,
    )
    return parser


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model", type=str, help="The model name from the timm library"
    )
    parser.add_argument("--criterion", type=str, help="The loss function")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Optional path for a checkpoint. If this is "
        "specified with resume=True, logging will continue for that run. "
        "With resume=False the weights are loaded from this checkpoint "
        "and a new model is trained.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--freeze_base",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Whether the convolutional layers are frozen (not trained)",
        default=False,
    )
    parser.add_argument(
        "--pretrained",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="If True, the pretrained weights from timm are used. Usually ImageNet",
        default=True,
    )
    parser.add_argument(
        "--inverse_class_map",
        help="'none', if no inverse mapping should be done, 'same', if the "
        "inverse of the label map provided with the dataset is used",
        type=str,
        default="same",
    )
    parser.add_argument(
        "--feature_extraction",
        help="If set, only features will be extracted in the prediction script",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--return_logits",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="returns logits instead of softmax output in the prediction script",
        default=False,
    )
    return parser


def add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument("--min_epochs", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument(
        "--early_stopping",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument(
        "--auto_lr",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--precision", default=32)
    parser.add_argument(
        "--deterministic",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--resume",
        help="Whether to resume a training run or to start a new",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    return parser


def add_program_args(parser: argparse.ArgumentParser):
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--out_folder", type=str, default=".", required=False)
    parser.add_argument("--out_prefix", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--global_seed", type=int, default=123, required=False)
    return parser


"""
  __  __ ___ ____   ____ 
 |  \/  |_ _/ ___| / ___|
 | |\/| || |\___ \| |    
 | |  | || | ___) | |___ 
 |_|  |_|___|____/ \____|
                         
"""


def load_continuous_transform(name):
    if name == "log":
        label_tf = {"fwd": np.log, "inv": np.exp}
    else:
        raise Exception(f"Invalid transform name {name}")
    return label_tf


def load_class_map(fname):
    """Loads functions that map classes to indices and the inverse of this"""
    if not Path(fname).exists():
        class_map = load_continuous_transform(fname)
    else:
        with open(fname) as f:
            fwd = {label.strip(): i for i, label in enumerate(f)}
        inv = {v: k for k, v in fwd.items()}
        class_map = {}
        # class_map["fwd"] = lambda x: np.array([fwd[str(int(v))] for v in x])
        # class_map["inv"] = lambda x: np.array([inv[int(v)] for v in x])
        class_map["fwd"] = lambda x: np.array([fwd[v] for v in x])
        class_map["inv"] = lambda x: np.array([inv[v] for v in x])
        class_map["fwd_dict"] = fwd
        class_map["inv_dict"] = inv
    return class_map


def read_image(fpath: str):
    img = Image.open(fpath)
    return img


def show_img(T, name=None, to_numpy=False):
    """Displays an arbitary tensor/numpy array"""
    fname = name or "img-" + datetime.now().strftime("%H%M%S") + ".jpg"
    if isinstance(T, Image.Image):
        img = T
        I = np.array(img)
    else:
        if isinstance(T, torch.Tensor):
            if len(T.shape) == 4:
                T = torchvision.utils.make_grid(T)
            T = T.permute(1, 2, 0).numpy()
        elif isinstance(T, np.ndarray):
            T = T.astype(float)
        T -= T.min()
        T = T / (T.max() + 1e-8)
        I = (T * 255).astype(np.uint8)

    if to_numpy:
        return I
    else:
        img = Image.fromarray(I)
    img.save(fname)


def class_batch(ds, target, n=8):
    """Fetches n samples matching the target from the dataset ds"""
    all_inds = np.where(ds.y == target)[0]
    if len(all_inds) == 0:
        raise Exception("No label")
    inds = np.random.choice(all_inds, n)
    x_list = [ds[i][0] for i in inds]
    X = torch.stack(x_list)
    return X


def histogram_batch(ds, bins, b, n=8):
    """Fetches n samples from the histogram bin 'bin' for a continuous value ds.y"""

    all_inds = np.where(bins == b)[0]
    inds = np.random.choice(all_inds, n)
    x_list = [ds[i][0] for i in inds]
    X = torch.stack(x_list)
    return X


def visualize_dataset(ds, n=8, v=True, name=None, to_numpy=False):
    """Finds unique classes from the dataset ds and fetches n examples of all classes.
    Returns this as a array, or saves to an image
    """
    I_list = []
    fname = name or "img-" + datetime.now().strftime("%H%M%S") + ".jpg"

    # Continuous target
    if len(np.unique(ds.y)) > 50:
        _, bin_edges = np.histogram(ds.y, bins=50)
        bins = [bisect.bisect(bin_edges, x) for x in ds.y]
        for b in np.unique(bins):
            if v:
                print(bin_edges[b - 1])
            T = histogram_batch(ds, bins, b)
            I = show_img(T, to_numpy=True)
            I_list.append(I)
    # Categorical target
    else:
        for target in np.unique(ds.y):
            if v:
                print(target)
            T = class_batch(ds, target, n)
            I = show_img(T, to_numpy=True)
            I_list.append(I)

    I = np.vstack(I_list)

    if to_numpy:
        return I
    else:
        img = Image.fromarray(I)
        img.save(fname)


"""
  ____    _  _____  _    ____  _____ _____ ____  
 |  _ \  / \|_   _|/ \  / ___|| ____|_   _/ ___| 
 | | | |/ _ \ | | / _ \ \___ \|  _|   | | \___ \ 
 | |_| / ___ \| |/ ___ \ ___) | |___  | |  ___) |
 |____/_/   \_\_/_/   \_\____/|_____| |_| |____/ 
                                                 
"""


"""
                                  _ 
   __ _  ___ _ __   ___ _ __ __ _| |
  / _` |/ _ \ '_ \ / _ \ '__/ _` | |
 | (_| |  __/ | | |  __/ | | (_| | |
  \__, |\___|_| |_|\___|_|  \__,_|_|
  |___/                             
"""
#Functions for class_counts calculation in correct order:
import pandas as pd

# def load_class_map(class_map_path):
#     """
#     Load class mapping from a file.

#     Args:
#         class_map_path (str): Path to the class map file.

#     Returns:
#         dict: A dictionary mapping class names to indices.
#     """
#     with open(class_map_path, 'r') as file:
#         classes = [line.strip() for line in file.readlines()]
#     return {label: idx for idx, label in enumerate(classes)}

def encode_labels(labels, label_to_index):
    """
    Encode labels according to the provided mapping.

    Args:
        labels (list): List of labels to encode.
        label_to_index (dict): Mapping from label names to indices.

    Returns:
        list: List of encoded labels.
    """
    return [label_to_index[label] for label in labels]

def calculate_class_counts(encoded_labels, num_classes):
    """
    Calculate class counts from encoded labels.

    Args:
        encoded_labels (list of int): List containing encoded class labels as integers.
        num_classes (int): Total number of classes.

    Returns:
        list: List of counts for each class.
    """
    counts = [0] * num_classes
    for label in encoded_labels:
        counts[label] += 1
    return counts


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset for reading image/target pairs from a filepath list

    Args:
        filenames: list of filepaths
        y: list of targets
        preload_transform: transform to apply to the PIL image after loading and before
                            loading into memory
        transform: transform to apply to the image after loading
        load_to_memory: if True, the images are loaded into memory
    """

    def __init__(
        self,
        filenames: list,
        y: list,
        preload_transform=None,
        transform=None,
        load_to_memory=True,
    ):
        self.filenames = filenames
        self.y = y
        self.preload_transform = preload_transform
        self.transform = transform
        self.mem_dataset = None

        if load_to_memory:
            self.mem_dataset = []
            print("Loading dataset to memory...")
            for i in tqdm(range(len(filenames))):
                self.mem_dataset.append(self.__readfile(i))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Reads item either from memory or from disk"""
        if self.mem_dataset:
            X = self.mem_dataset[index]
        else:
            X = self.__readfile(index)

        if self.transform:
            X = self.transform(X)

        if self.y is not None:
            y = torch.as_tensor(self.y[index], dtype=torch.float32)
        else:
            y = None
        return X, y

    def __readfile(self, index):
        """Actual loading of the item"""
        fname = self.filenames[index]
        img = read_image(fname)
        if self.preload_transform:
            img = self.preload_transform(img)
        return img


class LitDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for an arbitary dataset

    Args:

        csv_path: path to the csv file containing the filenames

        data_folder: path to the folder containing the images

        fold: cross-validation fold to use

        label: column to use as the label

        label_transform: function to apply to the label list

        batch_size: batch size

        imsize: size of the images

        load_to_memory: whether to load the images to memory

        tta_n: The number of test-time-augmentation rounds
    """

    def __init__(
        self,
        data_folder: str,
        dataset_name: str = "imagefolder",
        csv_path: str = None,
        fold: int = None,
        label: str = None,
        batch_size: int = 128,
        aug: str = "only-flips",
        imsize: int = 224,
        label_transform=None,
        load_to_memory: bool = False,
        tta_n: int = 5,
        class_map_path: str = None,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.dataset_name = dataset_name

        self.csv_path = csv_path
        self.fold = fold
        self.label = label

        self.aug = aug
        self.batch_size = batch_size

        self.label_transform = label_transform
        self.load_to_memory = load_to_memory
        self.tta_n = tta_n

        self.class_map_path = class_map_path

        self.labels = {}  # Initialize the labels attribute

        self.cpu_count = int(
            os.getenv("SLURM_CPUS_PER_TASK") or torch.multiprocessing.cpu_count()
        )
        self.drop_last = lambda x: True if len(x) % batch_size == 1 else False

        self.aug_args = {"imsize": imsize}
        self.tf_test, self.tf_train = choose_aug(self.aug, self.aug_args)

    def setup(self, stage=None):
        fnames, labels = preprocess_dataset(
            data_folder=self.data_folder,
            dataset_name=self.dataset_name,
            csv_path=self.csv_path,
            fold=self.fold,
            label=self.label,

        )
        
    ###################################################################33
        """self.labels["train"] = labels["train"]
        self.labels["val"] = labels["val"]
        #self.labels = labels  # Make sure this assignment correctly populates the labels
        # Assuming labels["train"] contains targets for training dataset
        if stage == 'fit' or stage is None:
            # Assuming labels["train"] is a list of integers
            labels_train_np = np.array(labels["train"], dtype=np.int64)  # Convert to NumPy array of type int64
            train_targets = torch.tensor(labels_train_np)
            self.class_counts = calculate_class_counts(train_targets)
            print(self.class_counts)"""

        # if stage == 'fit' or stage is None:
        #     # Create an encoding for the labels
        #     unique_labels = set(labels["train"])
        #     label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        #     # Encode the training labels
        #     labels_encoded = [label_to_index[label] for label in labels["train"]]
        #     self.labels_encoded = {"train": labels_encoded}
            
        #     # Store the class counts
        #     self.class_counts = calculate_class_counts(labels_encoded)
        #     print(self.class_counts)

        # if stage == 'fit' or stage is None:
        #     # Load the class map
        #     class_map_path = args.class_map
        #     label_to_index = load_class_map(class_map_path)

        #     # Encode the training labels
        #     labels_encoded = encode_labels(labels["train"], label_to_index)
        #     self.labels_encoded = {"train": labels_encoded}
            
        #     # Store the class counts
        #     self.class_counts = calculate_class_counts(labels_encoded)
        #     print(self.class_counts, fold)
        if stage == 'fit' or stage is None:
            # Load the class map
            class_map_path = self.class_map_path
            class_map = load_class_map(class_map_path)

            # Forward map is used to encode labels
            label_to_index = class_map['fwd_dict']

            # Encode the training labels using the forward map
            labels_encoded = encode_labels(labels["train"], label_to_index)
            self.labels_encoded = {"train": labels_encoded}
            
            # Number of classes
            num_classes = len(label_to_index)
            
            # Store the class counts
            self.class_counts = calculate_class_counts(labels_encoded, num_classes)
            print(self.class_counts)

    ##############################################################3

        if self.label_transform:
            labels["train"] = self.label_transform(labels["train"])
            labels["val"] = self.label_transform(labels["val"])
            labels["test"] = self.label_transform(labels["test"])

        self.trainset = Dataset(
            fnames["train"],
            labels["train"],
            preload_transform=None,
            transform=self.tf_train,
            load_to_memory=self.load_to_memory,
        )

        self.valset = Dataset(
            fnames["val"],
            labels["val"],
            preload_transform=None,
            transform=self.tf_test,
            load_to_memory=self.load_to_memory,
        )

        self.testset = Dataset(
            fnames["test"],
            labels["test"],
            preload_transform=None,
            transform=self.tf_test,
            load_to_memory=self.load_to_memory,
        )

        tta_list = [self.testset] + [
            Dataset(
                fnames["test"],
                labels["test"],
                preload_transform=None,
                transform=self.tf_train,
                load_to_memory=self.load_to_memory,
            )
            for _ in range(self.tta_n - 1)
        ]

        self.ttaset = torch.utils.data.ConcatDataset(tta_list)

    #################################################################################
    """    # Inside LitDataModule
    def calculate_and_get_class_counts(self):
        # Assuming you have a way to access labels directly or through a dataset attribute
        # This is a placeholder; adapt it to your actual data structure
        all_labels = self.labels["train"] + self.labels["val"]
        train_targets = torch.tensor(all_labels)
        return calculate_class_counts(train_targets)"""
##############################################################################



    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last(self.trainset),
            num_workers=self.cpu_count,
        )

        return trainloader

    def val_dataloader(self):
        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            drop_last=self.drop_last(self.valset),
            num_workers=self.cpu_count,
        )

        return valloader

    def test_dataloader(self):
        testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.cpu_count,
        )

        return testloader

    def tta_dataloader(self):
        ttaloader = torch.utils.data.DataLoader(
            self.ttaset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.cpu_count,
        )
        return ttaloader

    def tta_process(self, y):
        A = y.reshape(self.tta_n, len(self.testset))
        return pd.DataFrame(A).T.mode(axis=1).iloc[:, 0].values

    def tta_process_output(self, output):
        A = output.T.reshape(output.shape[1], self.tta_n, len(self.testset))
        return A.mean(axis=1).T

    def visualize_datasets(self, folder):
        _now = datetime.now().strftime("%H%M%S")
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        visualize_dataset(self.trainset, v=False, name=folder / f"{_now}-train.jpg")
        visualize_dataset(self.valset, v=False, name=folder / f"{_now}-val.jpg")
        visualize_dataset(self.testset, v=False, name=folder / f"{_now}-test.jpg")


def choose_aug(aug, args):
    imsize = args["imsize"]
    a_end_tf = A.Compose(
        [
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    keep_aspect_resize = A.Compose(
        [
            A.LongestMaxSize(max_size=imsize),
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0),
        ],
        p=1.0,
    )
    if aug == "none":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = tf_test
    elif aug == "torch-only-flips":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        tf_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif aug == "aug-01":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice(
                    [
                        transforms.GaussianBlur(kernel_size=(3, 3)),
                        transforms.ColorJitter(brightness=0.5, hue=0.1),
                    ]
                ),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=(0, 360)),
                        transforms.RandomPerspective(distortion_scale=0.1),
                        transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.75, 0.9)
                        ),
                        transforms.RandomResizedCrop(size=(imsize, imsize)),
                    ]
                ),
                transforms.RandomChoice(
                    [
                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                        transforms.RandomAutocontrast(),
                        transforms.RandomEqualize(),
                    ]
                ),
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif aug == "color-jitter":
        tf_test = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = tf_test

    elif aug == "keep-aspect":
        transform_test = A.Compose([keep_aspect_resize, a_end_tf])
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = tf_test

    elif aug.startswith("flips"):
        keep_aspect = "keep-aspect" in aug
        rotate = "rotate" in aug
        transform_test = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                A.Flip(),
                A.RandomRotate90(p=0.5) if rotate else A.NoOp(),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    elif aug.startswith("color"):
        keep_aspect = "keep-aspect" in aug
        transform_test = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20
                        ),
                        A.ToGray(p=0.2),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    elif aug.startswith("geometric"):
        border = 0
        keep_aspect = "keep-aspect" in aug
        transform_test = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.OneOf(
                    [
                        A.Rotate(limit=360, border_mode=border),
                        A.Perspective(pad_mode=border),
                        A.Affine(
                            scale=(0.5, 0.9),
                            translate_percent=0.1,
                            shear=(-30, 30),
                            rotate=360,
                        ),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    elif aug.startswith("aug-02"):
        apply_eq = "EQ" in aug
        apply_bw = "BW" in aug
        keep_aspect = "keep-aspect" in aug
        border = 0
        transform_test = A.Compose(
            [
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                # Possible equalization
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                # Slow pixel tf
                A.Posterize(p=0.1),
                A.NoOp() if apply_eq else A.Equalize(p=0.2),
                A.CLAHE(0.2),
                A.OneOf(
                    [
                        A.GaussianBlur(),
                        A.Sharpen(),
                    ],
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                # Colors
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2
                        ),
                        A.NoOp() if apply_bw else A.ToGray(p=0.5),
                    ],
                    p=0.2,
                ),
                # Slow geometrical tf
                A.OneOf(
                    [
                        A.Rotate(limit=360, border_mode=border),
                        A.Perspective(pad_mode=border),
                        A.Affine(
                            scale=(0.5, 0.9),
                            translate_percent=0.1,
                            shear=(-30, 30),
                            rotate=360,
                        ),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                A.CoarseDropout(
                    max_holes=30,
                    max_height=15,
                    max_width=15,
                    min_holes=1,
                    min_height=2,
                    min_width=2,
                ),
                A.RandomSizedCrop(
                    min_max_height=(int(0.5 * imsize), int(0.8 * imsize)),
                    height=imsize,
                    width=imsize,
                    p=0.3,
                ),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    else:
        raise ValueError(f"Invalid augmentation value {aug}")

    return tf_test, tf_train


"""
  __  __  ___  ____  _____ _     ____  
 |  \/  |/ _ \|  _ \| ____| |   / ___| 
 | |\/| | | | | | | |  _| | |   \___ \ 
 | |  | | |_| | |_| | |___| |___ ___) |
 |_|  |_|\___/|____/|_____|_____|____/ 
                                       
"""


# def calculate_class_counts(encoded_labels):
#     """
#     Calculate class counts from encoded labels.

#     Args:
#         encoded_labels (list of int): List containing encoded class labels as integers.

#     Returns:
#         dict: A dictionary where each key is a class index and the value is the count of samples for that class.
#     """
#     counts = {}
#     for label in encoded_labels:
#         counts[label] = counts.get(label, 0) + 1

#     # Calculate the maximum class index to ensure the list covers all classes
#     max_class_index = max(counts.keys())

#     # Initialize a list with zeros for all possible class indices
#     class_counts = [0] * (max_class_index + 1)

#     # Populate the list with counts from the dictionary
#     for class_index, count in counts.items():
#         class_counts[class_index] = count

#     return class_counts

def cross_entropy(output, target):
    loss = F.cross_entropy(output, target.long())
    return loss

#alphas for Focal loss
def calculate_alpha(class_counts, num_classes):
    # Calculate alpha values based on class_counts here
    # For example, inverse frequency or any other method
    inverse_frequency = [1.0 / count for count in class_counts]
    # Normalize if necessary
    sum_inv_freq = sum(inverse_frequency)
    alpha = [inv_freq / sum_inv_freq for inv_freq in inverse_frequency]
    # Ensure the length matches the number of classes
    if len(alpha) != num_classes:
        raise ValueError("Length of alpha does not match number of classes")
    return alpha

class FocalLoss(torch.nn.Module):
    def __init__(self, class_counts, gamma=2.0, reduction='mean', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if num_classes is None:
            raise ValueError("num_classes must be set when alpha is not provided")
        
        alpha = calculate_alpha(class_counts, num_classes=num_classes)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        
        if hasattr(self, 'alpha'):
            # Update the alpha buffer if it already exists
            self.alpha = alpha
        else:
            # Register alpha as a buffer if it doesn't exist
            self.register_buffer('alpha', alpha)

    def forward(self, inputs, targets):
        # Ensure targets are of the correct type for indexing
        targets = targets.long()
        
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        
        # Now targets is guaranteed to be of a type suitable for indexing
        alpha_t = self.alpha[targets]
        print("The value of alpha inside focal loss is: ", alpha_t)
        loss = alpha_t * ((1 - pt) ** self.gamma) * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Define the custom loss function
class CILoss(torch.nn.Module):
    def __init__(self, class_counts, gamma=2.0, theta=0.5, device='cuda'):
        super(CILoss, self).__init__()
        self.gamma = gamma
        self.theta = theta
        self.device = device
        N_max = max(class_counts)
        self.weights = torch.tensor([((N_max/ N_j) + theta) for N_j in class_counts], dtype=torch.float).to(device)

    def forward(self, inputs, targets):
        # Ensure targets is of type torch.long for indexing
        targets = targets.long()

        # Convert inputs to softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Sort weights according to the target order
        sorted_weights = self.weights[targets]
        """
        # Create a tensor of weights based on target indices
        weights = self.weights[targets]"""
        #print("The value of sorted weights in CI loss is: ", sorted_weights)
        # Calculate the log of probabilities
        log_probs = torch.log(probs)
        # Gather the log probabilities for each target class
        log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Calculate the CI loss
        loss = -sorted_weights * torch.pow((1 - probs.gather(1, targets.unsqueeze(1)).squeeze(1)), self.gamma) * log_probs
        return loss.mean()




def choose_criterion(name, class_counts=None, n_classes=None, gamma=2.0, theta=0.5, device='cuda'):
    if name == "cross-entropy":
        return cross_entropy
    elif name == "class-imbalance":
        if class_counts is None:
            raise ValueError("class_counts must be provided for class-imbalance loss")
        return CILoss(class_counts, gamma=gamma, theta=theta, device=device)
    elif name == "focal":
        # Note: Adjust alpha and gamma as per your requirement
        return FocalLoss(class_counts, num_classes=n_classes, gamma=gamma)
    else:
        raise Exception(f"Invalid criterion name '{name}'")


class Model(nn.Module):
    """PyTorch module for an arbitary timm model, separating the base and projection head"""

    def __init__(
        self,
        model: str = "resnet18",
        freeze_base: bool = True,
        pretrained: bool = True,
        n_classes: int = 1,
    ):
        """Initializes the model

        Args:
            model (str): name of the model to use
            freeze_base (bool): if True, the base is frozen
            pretrained (bool): if True, use pretrained weights
            n_classes (int): output layer size
        """
        super().__init__()

        self.h_dim = (
            timm.create_model(model, pretrained=False, num_classes=1)
            .get_classifier()
            .in_features
        )
        self.base_model = timm.create_model(model, num_classes=0, pretrained=pretrained)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.proj_head = nn.Sequential(nn.Linear(self.h_dim, n_classes))

    def forward(self, x):
        h = self.base_model(x)
        return self.proj_head(h)

    def base_forward(self, x):
        return self.base_model(x)

    def proj_forward(self, h):
        return self.proj_head(h)


class LitModule(pl.LightningModule):
    """PyTorch Lightning module for training an arbitary model"""

    def __init__(
        self,
        model: str,
        freeze_base: bool = False,
        pretrained: bool = True,
        n_classes: int = 1,
        criterion: str = "mse",
        opt: dict = {"name": "adam"},
        lr: float = 1e-4,
        label_transform=None,
        class_counts=None,  # Add this parameter
        gamma=2.0,         # Add this parameter
        theta=0.5,         # Add this parameter
        device='cuda',     # Add this parameter
    ):
        """Initialize the module
        Args:
            model (str): name of the ResNet model to use

            freeze_base (bool): whether to freeze the base model

            pretrained (bool): whether to use pretrained weights

            n_classes (int): number of outputs. Set 1 for regression

            criterion (str): loss function to use

            lr (float): learning rate

            label_transform: possible transform that is done for the output labels
        """
        super().__init__()
        self.save_hyperparameters(ignore=["label_transform"])
        self.example_input_array = torch.randn((1, 3, 224, 224))
        self.model = Model(
            model=model,
            freeze_base=freeze_base,
            pretrained=pretrained,
            n_classes=n_classes,
        )
        self.lr = lr
        self.label_transform = label_transform
        self.criterion = choose_criterion(criterion, class_counts, n_classes, gamma, theta, device)
        self.opt_args = opt

        if criterion in ["cross-entropy", "class-imbalance", "focal"]:
            self.is_classifier = True
        else:
            self.is_classifier = False

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def predict_func(self, output):
        """Processes the output for prediction"""
        if self.is_classifier:
            return output.argmax(dim=1)
        else:
            return output.flatten()

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def configure_optimizers(self):
        """Sets optimizers based on a dict passed as argument"""
        if self.opt_args["name"] == "adam":
            return torch.optim.Adam(self.model.parameters(), self.lr)
        elif self.opt_args["name"] == "adamw":
            return torch.optim.AdamW(self.model.parameters(), self.lr)
        else:
            raise Exception("Invalid optimizer")

    def common_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        return x, y, out, loss

    def common_epoch_end(self, outputs, name: str):
        """Combination of outputs for calculating metrics"""
        y_true = torch.cat([x["y_true"] for x in outputs]).cpu().detach().numpy()
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().detach().numpy()

        if self.label_transform:
            y_true = self.label_transform(y_true)
            y_pred = self.label_transform(y_pred)

        if self.is_classifier:
            self.log(f"{name}/acc", accuracy_score(y_true, y_pred))
            self.log(
                f"{name}/f1",
                f1_score(y_true, y_pred, average="weighted", zero_division=0),
            )

        return y_true, y_pred

    # Training
    def training_step(self, batch, batch_idx):
        _, y, out, loss = self.common_step(batch, batch_idx)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        outputs = {"loss": loss, "y_true": y, "y_pred": self.predict_func(out)}
        self.training_step_outputs.append(outputs)
        return loss

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        _, _ = self.common_epoch_end(outputs, "train")
        self.training_step_outputs.clear()

    # Validation
    def validation_step(self, batch, batch_idx):
        _, y, out, val_loss = self.common_step(batch, batch_idx)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        outputs = {"y_true": y, "y_pred": self.predict_func(out)}
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        _, _ = self.common_epoch_end(outputs, "val")
        self.validation_step_outputs.clear()

    # Testing
    def test_step(self, batch, batch_idx):
        _, y, out, test_loss = self.common_step(batch, batch_idx)
        self.log("test/loss", test_loss, on_step=True, on_epoch=True)
        outputs = {"y_true": y, "y_pred": self.predict_func(out), "out": out}
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        if self.is_classifier:
            logits = torch.cat([x["out"] for x in outputs])
            self.softmax = logits.softmax(dim=1).cpu().detach().numpy()
            self.logits = logits.cpu().detach().numpy()

        self.y_true, self.y_pred = self.common_epoch_end(outputs, "test")


class FeatureExtractionModule(pl.LightningModule):
    def __init__(
        self,
        feature_extraction_mode: str,
        model: str,
        freeze_base: bool = False,
        pretrained: bool = True,
        n_classes: int = 0,
        criterion: str = "cross-entropy",
        opt: dict = {"name": "adam"},
        lr: float = 1e-4,
        label_transform=None,
    ):
        """
        The feature exctraction module implements the same interface as the basic LitModule
        for passing LitModule parameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["label_transform"])
        self.example_input_array = torch.randn((1, 3, 224, 224))

        self.feature_extraction_mode = feature_extraction_mode
        self.model = Model(
            model=model,
            freeze_base=freeze_base,
            pretrained=pretrained,
            n_classes=n_classes,
        )
        self.lr = lr
        self.label_transform = label_transform
        self.criterion = choose_criterion(criterion)
        self.opt_args = opt

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if self.feature_extraction_mode == "unpooled":
            return self.model.base_model.forward_features(x)
        elif self.feature_extraction_mode == "pooled":
            return self.model.base_forward(x)
        else:
            raise Exception(
                f"Invalid feature extraction mode {self.feature_extraction_mode}"
            )

    def test_step(self, batch, batch_idx):
        bx, by = batch
        out = self.forward(bx)
        outputs = {
            "y_true": by.cpu().detach().numpy(),
            "out": out.cpu().detach().numpy(),
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.y_true = [x["y_true"] for x in outputs]
        self.y_pred = [x["out"] for x in outputs]
