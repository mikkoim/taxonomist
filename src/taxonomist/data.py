import os
from datetime import datetime
from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from tqdm import tqdm

import math
from typing import Dict, List, Optional, Tuple

from torch import Tensor

from torchvision.transforms import functional as F, InterpolationMode

from .utils import load_module_from_path, read_image, visualize_dataset


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
        batch = {"x": X, "y": y, "fname": str(self.filenames[index])}
        return batch

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
        dataset_config_path: str = None,
        dataset_name: str = None,
        csv_path: str = None,
        fold: int = None,
        label: str = None,
        aug: str = "none",
        batch_size: int = 128,
        imsize: int = 224,
        label_transform=None,
        load_to_memory: bool = False,
        tta_n: int = 5,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.dataset_config_path = dataset_config_path
        self.dataset_name = dataset_name

        self.csv_path = csv_path
        self.fold = fold
        self.label = label

        self.aug = aug
        self.batch_size = batch_size
        self.imsize = imsize

        self.label_transform = label_transform
        self.load_to_memory = load_to_memory
        self.tta_n = tta_n

        self.cpu_count = int(
            os.getenv("SLURM_CPUS_PER_TASK") or torch.multiprocessing.cpu_count()
        )
        self.drop_last = lambda x: True if len(x) % batch_size == 1 else False

        self.aug_args = {"imsize": imsize}
        self.tf_test, self.tf_train = choose_aug(self.aug, self.aug_args)

    def setup(self, stage=None):
        dataset_config_module = load_module_from_path(self.dataset_config_path)

        fnames, labels = dataset_config_module.preprocess_dataset(
            data_folder=self.data_folder,
            dataset_name=self.dataset_name,
            csv_path=self.csv_path,
            fold=self.fold,
            label=self.label,
        )

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

def _custom_apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    """
    Custom apply_op
    """
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

class TrivialAugmentWideNoWarp(torch.nn.Module):
    """
    Modified version of TrivialAugmentWide, with warping operations removed.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _custom_apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

def choose_aug(aug, args):
    """Select data augmentation transformations based on the provided augmentation scheme."""
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

    elif aug == "trivialaugment":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    elif aug == "trivialaugment-noresize":
        # No resizing during training. Assumes images are the right size
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = transforms.Compose(
            [
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    
    elif aug == "trivialaugment-nowarp":
        def tf_train(img):
            aug = TrivialAugmentWideNoWarp()
            aug2 = A.Compose(
                    [
                        A.Resize(imsize, imsize, p=1.0),
                        A.Flip(),
                        A.RandomRotate90(p=0.5),
                        a_end_tf,
                    ]
            )
            x = aug(img)
            x = aug2(image=np.array(x))["image"]
            return x

        transform_test = A.Compose(
            [
                A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        

    elif aug == "randaugment":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

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
    
    elif aug == "flips-cont-rotate":
        transform_test = A.Compose(
            [
                A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.Resize(imsize, imsize, p=1.0),
                A.Flip(),
                A.RandomRotate90(p=1.0),
                A.Rotate(p=0.5, border_mode=0),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

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
