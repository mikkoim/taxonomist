import argparse
from distutils.util import strtobool
from dataclasses import dataclass
from typing import Optional, Union, List


@dataclass(frozen=True)
class TaxonomistModelArguments:
    task: str = "classification"
    data_folder: str = None
    dataset_config_path: str = None
    dataset_name: str = None
    csv_path: str = None
    custom_dataset: bool = False

    label_column: Optional[str] = None
    fold: int = 0
    class_map_name: str = None

    imsize: int = None
    batch_size: int = 32
    aug: str = "none"
    mixup: bool = False
    load_to_memory: bool = False
    tta: bool = False
    tta_n: int = 5

    model_name: str = "mobilenetv3_large_100.ra_in1k"
    custom_model: bool = False
    criterion: str = None
    ckpt_path: Optional[str] = None  # required if resume=True
    freeze_base: bool = False
    pretrained: bool = True
    inverse_class_map: str = "same"
    feature_extraction: str = None
    return_softmax: bool = False

    min_epochs: Optional[int] = None
    max_epochs: Optional[int] = None
    save_top_k: Optional[int] = 1
    early_stopping: bool = False
    early_stopping_patience: int = 5  # used if early_stopping=True
    lr: float = 1e-4
    opt: str = "adam"
    lr_scheduler: str = None

    auto_lr: bool = False
    swa: bool = False
    swa_lrs: float = 1e-2
    precision: Union[int, str] = 32
    deterministic: bool = False
    resume: bool = False

    accelerator: str = "auto"
    strategy: Union[str, int] = "auto"
    devices: Union[str, int] = "auto"
    num_nodes: int = 1

    log_dir: str = "logs"
    no_wandb: bool = False
    out_folder: str = "outputs"
    out_prefix: str = "metrics"
    random_state: int = 42
    debug: bool = False
    smoke_test: bool = False

    log_every_n_steps: Optional[int] = 10
    check_val_every_n_epoch: Optional[int] = 1
    val_check_interval: Optional[float] = 1.0
    suffix = None

    def __post_init__(self):
        validate_arguments(self)


def validate_arguments(args: TaxonomistModelArguments):
    """
    Validate the arguments of the TaxonomistModelArguments dataclass.
    Args:
        args (TaxonomistModelArguments): The arguments to validate

    Raises:
        ValueError: If the arguments are invalid.
    """

    # Check that the ckpt_path is set if resume is True
    if args.resume:
        if args.ckpt_path is None:
            raise ValueError("When resuming, a ckpt_path must be set")

    # Check that the task is one of the allowed values
    if not args.task in ["classification", "regression", "feature-extraction"]:
        raise ValueError(
            "task must be 'classification', 'regression', or 'feature-extraction'"
        )
    
    # Check that mixup is not used with a custom model
    if args.aug == "mixup" and args.custom_model:
        raise ValueError("Mixup augmentation is not supported with custom models. "
                         "Workaround is to define mixup in a custom dataset function.")
    
    # Check that mixup is only used with classification
    if args.aug == "mixup" and args.task != "classification":
        raise ValueError("Mixup augmentation is only supported for classification tasks.")
    
    # Check that if the task is regression, the criterion is not cross-entropy
    if args.task == "regression" and args.criterion == "cross-entropy":
        raise ValueError("Cross-entropy loss is not supported for regression tasks.")


def add_dataset_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Folder where the data can be found. This folder is "
        "used with the csv_path to produce final filenames for training",
        required=False,
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        help="The path to the dataset config file that defines data loading functions. "
        "The file must contain the function 'preprocess_dataset' that specifies "
        "a python function that loads filenames and labels for the dataset",
        required=False,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The dataset name that is used to select the function in "
        "'dataset_config_path' that "
        "determines how data should be loaded",
        required=False,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the csv file that contains dataset table and label "
        "information for each sample. "
        "Used along 'data_folder to produce final filenames for training. "
        "The csv should contain train-test-validation split info for all "
        "cross-validation folds",
        required=False,
    )
    parser.add_argument(
        "--custom_dataset",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="If True, a custom dataset is loaded using the 'return_dataset' function "
        "from the dataset config file. If False, the default behavior of loading "
        "filenames and labels based on a csv file is used.",
        default=False,
        required=False,
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
        "--mixup",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="If True, MixUp augmentation is applied",
        default=False,
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
        "--model_name",
        type=str,
        help="The model name from the timm library",
        default="mobilenetv3_large_100.ra_in1k",
        required=False,
    )
    # Alias for above
    parser.add_argument("--model", dest="model_name")
    parser.add_argument(
        "--custom_model",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="If True, a custom model is loaded using the 'return_model' function "
        "from the dataset config file. If False, the default behavior of using "
        "a timm model is used.",
        default=False,
        required=False,
    )

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
        "--return_softmax",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="calculates and returns the softmax instead of logits in the prediction script",
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
        "--save_top_k",
        type=int,
        help="Saves the top k best models. Default 1",
        default=1,
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
        "--log_every_n_steps",
        type=int,
        help="How often to add logging rows",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        help="Check val every n train epochs.",
        default=1,
        required=False,
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        help="How often within one training epoch to check the validation set. Can specify as float or int. (from Lightning)",
        default=1.0,
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
        "--swa",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="Whether to use Stochastic Weight Averaging during training. Default False",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--swa_lrs",
        type=float,
        help="The SWA learning rate to use",
        default=1e-2,
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
    parser.add_argument(
        "--accelerator",
        help="Lighting Trainer accelerator. Default 'auto'",
        default="auto",
        required=False,
    )
    parser.add_argument(
        "--strategy",
        help="Lighting Trainer training strategy, for example 'ddp'. Default 'auto'.",
        default="auto",
        required=False,
    )
    parser.add_argument(
        "--devices",
        help="Lightning Trainer devices to use. Default 'auto'",
        default="auto",
        required=False,
    )
    parser.add_argument(
        "--num_nodes",
        help="Number of GPU nodes for distributed training. Default 1.",
        type=int,
        default=1,
        required=False,
    )
    return parser


def add_program_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--task",
        type=str,
        help="A task specifier. In predict stage, can be 'predict' or 'feature-extraction'"
        "In train stage, can be 'classification' or 'regression'",
        default="classification",
        required=False,
    )
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
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser

def add_train_test_split_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--csv_path", type=str, help="Path to input csv file", required=True
    )

    parser.add_argument(
        "--target_col",
        type=str,
        help="Target variable column. Stratification is performed based on this",
        required=True,
    )

    parser.add_argument(
        "--group_col",
        type=str,
        help="Group column. Groups are non-overlapping across train-test-val splits",
        required=True,
    )

    parser.add_argument(
        "--n_splits", type=int, help="Number of splits. Default 5", default=5
    )

    parser.add_argument(
        "--verbose",
        type=int,
        help="If set to 1, prints information on data splits to console",
        default=1,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="The random seed for the split. Default is 42",
        default=42,
    )
    parser.add_argument(
        "--shuffle",
        type=lambda x: bool(strtobool(x)),
        help="Whether to shuffle each class's samples before splitting into batches.",
        nargs="?",
        const=True,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--generate_class_map",
        action="store_true",
        help="If set, generates a class map based on the target_col",
    )

    parser.add_argument("--out_folder", type=str, default=".")
    return parser

def add_combine_cv_predictions_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--tag", help="The augmentation/dataset identifier", type=str)
    parser.add_argument("--reference_csv", type=str, required=False)
    parser.add_argument("--reference_target", type=str)
    parser.add_argument(
        "--suffix",
        help="file suffix that identifies the csv files to be grouped",
        type=str,
        default=".csv",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--around", type=int)
    return parser

def add_group_predictions_args(parser: argparse.ArgumentParser):
    parser.add_argument("--predictions", type=str)
    parser.add_argument("--reference_csv", type=str)
    parser.add_argument("--reference_target", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--fold_col_prefix", type=str, default="")
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--reference_group", type=str)
    parser.add_argument("--agg_func", type=str)
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--group_logits", action="store_true")
    parser.add_argument(
        "--around", type=int, help="Round the output. Only on regression"
    )
    return parser

def add_evaluate_args(parser: argparse.ArgumentParser):
    parser.add_argument("--predictions", type=str)
    parser.add_argument("--metric_config", type=str)

    parser.add_argument("--reference_csv", default=None, type=str)
    parser.add_argument("--reference_target", default=None, type=str)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap_alpha", default=0.95)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="metrics")
    parser.add_argument("--around", default=None, type=int)
    return parser

def add_compare_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str)
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--print_versions", action="store_true")
    return parser