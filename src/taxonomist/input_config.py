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
        "--dataset_config_path",
        type=str,
        help="The path to the dataset config file that defines data loading functions. "
        "The file must contain the function 'preprocess_dataset' that specifies "
        "a python function that loads filenames and labels for the dataset",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The dataset name that is used to select the function in "
        "'dataset_config_path' that "
        "determines how data should be loaded",
        required=True,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the csv file that contains dataset table and label "
        "information for each sample. "
        "Used along 'data_folder to produce final filenames for training. "
        "The csv should contain train-test-validation split info for all "
        "cross-validation folds",
        required=True,
    )
    parser.add_argument(
        "--label_column",
        type=str,
        help="Label column. Found from the csv_path file.",
        default=None,
        required=False,
    )
    # Alias for above
    parser.add_argument("--label", dest="label_column")
    parser.add_argument(
        "--fold",
        type=int,
        help="The fold that is used for training. Found from the csv_path file.",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--class_map_name",
        type=str,
        help="Refers to a list of classes found in the dataset. Provides "
        "an unambiguous reference between strings and indices, even if some folds "
        "don't contain all classes",
        default=None,
        required=False,
    )
    # Alias for above
    parser.add_argument("--class_map", dest="class_map_name")
    return parser


def add_dataloader_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--imsize",
        type=int,
        help="Inputs are resized to this size",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to be used",
        default=32,
        required=False,
    )
    parser.add_argument(
        "--aug",
        type=str,
        help="Augmentation that is applied to the images",
        default="none",
        required=False,
    )
    parser.add_argument(
        "--load_to_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="If set, the dataset is loaded to memory",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--tta",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Test-time augmentation is applied",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--tta_n",
        type=int,
        help="The number of test-time augmentations",
        default=5,
        required=False,
    )
    return parser


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--timm_model_name",
        type=str,
        help="The model name from the timm library",
        default="mobilenetv3_large_100.ra_in1k",
        required=False,
    )
    # Alias for above
    parser.add_argument("--model", dest="timm_model_name")

    parser.add_argument(
        "--criterion", type=str, help="The loss function", default=None, required=False
    )

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
        required=False,
    )
    parser.add_argument(
        "--pretrained",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="If True, the pretrained weights from timm are used. Usually ImageNet",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--inverse_class_map",
        type=str,
        help="'none', if no inverse mapping should be done, 'same', if the "
        "inverse of the label map provided with the dataset is used",
        default="same",
        required=False,
    )
    parser.add_argument(
        "--feature_extraction",
        type=str,
        help="If set, only features will be extracted in the prediction script. "
        "If set, should be 'pooled' or 'unpooled'",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--return_logits",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="returns logits instead of softmax output in the prediction script",
        default=False,
        required=False,
    )
    return parser


def add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--min_epochs",
        type=int,
        help="Minimum number of epochs that are ran",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum amount of epochs that are ran",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--early_stopping",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Monitors validation loss and stop training when it stops improving.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="number of checks with no improvement after which training will be stopped",
        default=5,
        required=False,
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=None, required=False
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="Learning rate scheduler.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--opt",
        type=str,
        help="The optimizer name as a string",
        default="adam",
        required=False,
    )
    parser.add_argument(
        "--auto_lr",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Whether to run automatic learning rate tuning in the beginning of training",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--precision",
        help="The precision that is passed to lightning trained",
        default=32,
        required=False,
    )
    parser.add_argument(
        "--deterministic",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Whether pytorch lightning is set as deterministic",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--resume",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Whether to resume a training run or to start a new",
        default=False,
        required=False,
    )
    return parser


def add_program_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Logging directory. This name is passed to wandb.",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        help="Outputs are saved here.",
        default=".",
        required=False,
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        help="A prefix that is set to model or output names",
        default="",
        required=False,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="The random seed for the split. Default is 42",
        default=42,
        required=False,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser
