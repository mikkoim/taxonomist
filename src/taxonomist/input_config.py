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
        "--tta",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Test-time augmentation is applied",
        default=False,
    )
    parser.add_argument("--tta_n", type=int, default=5)
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
    parser.add_argument(
        "--random_state",
        type=int,
        help="The random seed for the split. Default is 42",
        default=42,
    )
    return parser
