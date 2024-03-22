import argparse
from distutils.util import strtobool

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
        default="imagefolder",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the csv file that contains label information for each sample. "
        "Used along data_folder to produce final filenames for training."
        "The csv should contain train-test-validation split info for all "
        "cross-validation folds",
        default=None,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="The fold that is used for training. Found from the csv_path file.",
        default=None,
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Label column. Found from the csv_path file.",
        default=None,
    )
    parser.add_argument(
        "--class_map",
        type=str,
        help="Refers to a list of classes found in the dataset. Provides "
        "an unambiguous reference between strings and indices, even if some folds "
        "don't contain all classes",
        default=None,
    )
    return parser


def add_dataloader_args(parser: argparse.ArgumentParser):
    parser.add_argument("--imsize", type=int, help="Inputs are resized to this size")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--aug", type=str, default="only-flips")
    parser.add_argument(
        "--load_to_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--tta", type=lambda x: bool(strtobool(x)), nargs="?", const=True, default=False
    )
    parser.add_argument("--tta_n", type=int, default=5)
    return parser


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str)
    parser.add_argument("--criterion", type=str)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--freeze_base",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--pretrained",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=True,
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
    parser.add_argument("--precision", type=int, default=32)
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