import pickle
import uuid
import gzip
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

import wandb

from .data import LitDataModule
from .model import FeatureExtractionModule, LitModule
from .utils import load_class_map, TaxonomistUid
from .predictions import TaxonomistPredictions


@dataclass(frozen=True)
class TaxonomistModelArguments:
    task: str
    data_folder: str
    dataset_config_path: str
    dataset_name: str
    csv_path: str
    custom_dataset: bool = False

    label_column: Optional[str] = None
    fold: int = 0
    class_map_name: str = None

    imsize: int = None
    batch_size: int = 32
    aug: str = "none"
    load_to_memory: bool = False
    tta: bool = False
    tta_n: int = 5

    timm_model_name: str = "mobilenetv3_large_100.ra_in1k"
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
    if args.resume:
        if args.ckpt_path is None:
            raise ValueError("When resuming, a ckpt_path must be set")

    if not args.task in ["classification", "regression", "feature-extraction"]:
        raise ValueError("task must be 'classification', 'regression', or 'feature-extraction'")

class TaxonomistCheckpoint:
    """
    A class used to represent a checkpoint in the Taxonomist model.

    Attributes:
        ckpt_path (Path): Path to the checkpoint file.
        ckpt (dict): The checkpoint dictionary.
    """

    def __init__(self, ckpt_path: str):
        """
        Initialize the TaxonomistCheckpoint with the given checkpoint path.

        Args:
            ckpt_path (str): Path to the checkpoint file.

        Raises:
            ValueError: If the checkpoint path does not exist.
        """
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise ValueError(f"The checkpoint path '{str(self.ckpt_path)}' does not exist")

        self.ckpt = torch.load(
            self.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    
    def __repr__(self):
        return f"TaxonomistCheckpoint({self.ckpt_path})"

    def is_last(self) -> bool:
        """
        Check if the checkpoint is the last checkpoint (i.e., has suffix '_last').

        Returns:
            bool: True if the checkpoint is the last, False otherwise.
        """
        return self.name.endswith("_last")

    @property
    def name(self) -> str:
        """
        Get the name of the checkpoint file without the extension.

        Returns:
            str: The name of the checkpoint file.
        """
        return self.ckpt_path.stem
    
    @property
    def basename(self) -> str:
        """
        Get the basename of the checkpoint file.

        Returns:
            str: The basename of the checkpoint file.
        """
        if self.is_last():
            return "_".join(self.name.split("_")[:-5])
        else:
            return "_".join(self.name.split("_")[:-4])
    
    @property
    def modelname(self) -> str:
        if self.is_last():
            return "_".join(self.name.split("_")[:-3])
        else:
            return "_".join(self.name.split("_")[:-2])
    
    @property
    def folder(self) -> Path:
        """
        Get the folder of the checkpoint file.

        Returns:
            Path: The folder of the checkpoint file.
        """
        return self.ckpt_path.parents[0]

    @property
    def uid(self) -> str:
        """
        Get the unique identifier of the checkpoint file.

        Returns:
            str: The unique identifier of the checkpoint file.
        """
        if self.is_last():
            return self.name.split("_")[-4]
        else:
            return self.name.split("_")[-3]

class PathManager:
    def __init__(self, stage: str, args: TaxonomistModelArguments, ckpt: TaxonomistCheckpoint=None):
        self.stage = stage
        self.args = args
        self.ckpt = ckpt

        if (self.args.ckpt_path is not None) and (ckpt is None):
            raise ValueError("A checkpoint path was provided but the checkpoint object is None")

        if self.ckpt is not None:
            self.has_checkpoint = True
        else:
            self.has_checkpoint = False

        # Setup the path variables
        # Example: outputs/rodi/rodi_resnet18/f0
        #           ^      ^          ^       ^
        #          root  dataset  basename  fold

        self._create_names()
        if stage == "training":
            self._create_train_out_folder()
        elif stage == "prediction":
            self._create_pred_out_folder()
            self.predict_fpath = self._create_predict_fpath(self.args.task)
        
        self.visualization_path = self.out_folder / f"aug-{self.args.aug}-{self.uid}"
        self.config_path = self.out_folder / f"config_{self.uid}.yml"
    
    @property
    def tag(self):
        tag = f"{self.args.dataset_name}_{self.args.aug}"
        if self.args.tta:
            tag += "_tta"
        return tag

    def _create_train_out_folder(self):
        out_folder = (
            Path(self.args.out_folder)
            / str(self.args.dataset_name)
            / str(self.basename)
            / f"f{self.args.fold}"
        )
        out_folder.mkdir(exist_ok=True, parents=True)
        self.out_folder = out_folder
        print(f"Output folder: {out_folder}")
    
    def _create_pred_out_folder(self):
        """
        Creates an output folder for a given task.
        """
        if self.args.task == "classification":
            folder_name = "predictions"
        elif self.args.task == "feature-extraction":
            folder_name = "features"
        else:
            raise ValueError("Task must be 'classification' or 'feature-extraction'")

        if self.has_checkpoint:
            out_folder = Path(self.ckpt.folder, folder_name, self.tag)
        else:
            out_folder = (
                Path(self.args.out_folder)
                / str(self.args.dataset_name)
                / str(self.args.timm_model_name)
                / f"f{self.args.fold}"
                / folder_name
                / self.tag
            )

        out_folder.mkdir(exist_ok=True, parents=True)
        self.out_folder = out_folder
        print(f"Output folder: {out_folder}")

    def _create_predict_fpath(self, task: str) -> Path:

        if task == "feature-extraction":
            name = f"{self.modelname}_{self.args.feature_extraction}.p.gz"
            return self.out_folder / name
        else:
            name = f"{self.modelname}_{self.args.aug}"
            if self.args.tta:
                name += "_tta"

            return Path(self.out_folder, name + ".csv")
    
    def _create_names(self):
        """
        Creates the basename and the modelname
        """
        if self.stage == "training":
            if self.args.resume:
                if f"{self.args.out_prefix}_{self.args.timm_model_name}" != self.ckpt.basename:
                    raise ValueError(
                        f"Input arguments {self.args.out_prefix} and {self.args.timm_model_name} do not match checkpoint basename {self.ckpt.basename}."
                        " Please double check you are resuming the correct model"
                    )

                self.modelname = self.ckpt.modelname
                self.basename = self.ckpt.basename
                self.uid = self.ckpt.uid
            else:
                self._set_new_names()

        elif self.stage == "prediction":
            if self.has_checkpoint is False:
                self._set_new_names()
            else:
                self.modelname = self.ckpt.modelname
                self.basename = self.ckpt.basename
                self.uid = self.ckpt.uid
        else:
            raise ValueError("staget must be 'training' or 'prediction'")
        
    def _set_new_names(self):
        """
        Creates the basename and modelname for a new model.
        """
        uid = datetime.now().strftime("%y%m%d-%H%M") + f"-{str(uuid.uuid4())[:4]}"
        self.uid = uid
        self.basename = f"{self.args.out_prefix}_{self.args.timm_model_name}"
        self.modelname = f"{self.basename}_f{self.args.fold}_{uid}"


    
class TaxonomistModel:
    def __init__(self, args: TaxonomistModelArguments):
        validate_arguments(args)
        self.args = args
        self._handle_checkpoint()
        self._handle_class_map() 

        if args.deterministic:
            pl.seed_everything(seed=args.random_state, workers=True)

    
    def _handle_checkpoint(self):
        """
        Creates a TaxonomistCheckpoint object.

        Uses:
            self.args.ckpt_path
            self.args.resume

        Sets:
            self.ckpt
            self.has_checkpoint
        """
        if self.args.ckpt_path is None:
            self.ckpt = None
            self.has_checkpoint = False
        else:
            self.ckpt = TaxonomistCheckpoint(self.args.ckpt_path)
            self.has_checkpoint = True


    def _handle_class_map(self):
        """
        Creates a class map for the model.

        Uses:
            self.args.class_map_name
        
        Sets:
            self.class_map
            self.n_classes
        """
        if (self.args.class_map_name is not None) and (
            self.args.class_map_name != "none"
        ):
            class_map = load_class_map(self.args.class_map_name)
            n_classes = len(class_map["fwd_dict"])
        else:
            class_map = {"fwd": None, "inv": None, "fwd_dict": None, "inv_dict": None}
            n_classes = 1
        
        self.class_map = class_map
        self.n_classes = n_classes

    
    def _create_data_module(self):
        dm = LitDataModule(
            data_folder=self.args.data_folder,
            dataset_config_path=self.args.dataset_config_path,
            dataset_name=self.args.dataset_name,
            csv_path=self.args.csv_path,
            custom_dataset=self.args.custom_dataset,
            fold=self.args.fold,
            label=self.args.label_column,
            label_transform=self.class_map["fwd"],
            imsize=self.args.imsize,
            batch_size=self.args.batch_size,
            aug=self.args.aug,
            load_to_memory=self.args.load_to_memory,
            tta_n=self.args.tta_n,
        )
        dm.setup()
        dm.visualize_datasets(self.path_manager.visualization_path)
        return dm

    def _create_train_model(self):
        self._create_lr_scheduler_params()
        self._create_opt_params()

        model = LitModule(
            model=self.args.timm_model_name,
            freeze_base=self.args.freeze_base,
            pretrained=self.args.pretrained,
            criterion=self.args.criterion,
            opt=self.opt_params,
            n_classes=self.n_classes,
            lr=self.args.lr,
            lr_scheduler=self.lr_scheduler_params,
            label_transform=self.class_map["inv"],
        )
        
        if self.has_checkpoint and (self.args.resume is False):
            self._load_checkpoint(model)
        return model

    def _create_predict_model(self):
        model = LitModule(**self.ckpt.ckpt["hyper_parameters"])
        
        if self.args.inverse_class_map == "none":
            model.label_transform = None
        else:
            model.label_transform = self.class_map["inv"]
        
        model.freeze()
        return model
    
    def _create_feature_extraction_model(self):
        if self.has_checkpoint:
            model = FeatureExtractionModule(
                feature_extraction_mode=self.args.feature_extraction,
                **self.ckpt.ckpt["hyper_parameters"],
            )
        else:
            model = FeatureExtractionModule(
                feature_extraction_mode=self.args.feature_extraction,
                model=self.args.model,
                pretrained=True,
            )
        model.freeze()
        return model
    
    def _create_model(self, stage:str):
        if stage == "training":
            model = self._create_train_model()
        elif stage == "prediction":
            if self.args.task == "feature-extraction":
                model = self._create_feature_extraction_model()
            else:
                model = self._create_predict_model()
        else:
            raise ValueError("stage must be 'training' or 'prediction'")
        return model

    def _load_checkpoint(self, model):
        ckpt = self.ckpt.ckpt
        try:
            model.load_state_dict(ckpt["state_dict"])
        except RuntimeError:
            print(
                "Checkpoint and model parameters don't match. Loading without last layer"
            )
            model = LitModule(**ckpt["hyper_parameters"])
            model.load_state_dict(ckpt["state_dict"], strict=False)
            if self.args.freeze_base:
                model.model.freeze_base()
            model.model.init_proj_head(self.n_classes)

    def _create_callbacks(self):
        # Best model saving
        checkpoint_callback_best = ModelCheckpoint(
            monitor="val/loss",
            dirpath=self.path_manager.out_folder,
            save_top_k=self.args.save_top_k,
            filename=f"{self.path_manager.modelname}_" + "epoch{epoch:02d}_val-loss{val/loss:.2f}",
            auto_insert_metric_name=False,
        )

        # Last model saving
        checkpoint_callback_last = ModelCheckpoint(
            monitor="epoch",
            mode="max",
            dirpath=self.path_manager.out_folder,
            filename=f"{self.path_manager.modelname}_"
            + "epoch{epoch:02d}_val-loss{val/loss:.2f}_last",
            auto_insert_metric_name=False,
        )

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [checkpoint_callback_best, checkpoint_callback_last, lr_monitor]

        # Optional callbacks

        # Early stopping
        if self.args.early_stopping:
            print(
                f"Using early stopping with patience {self.args.early_stopping_patience}"
            )
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss", patience=self.args.early_stopping_patience
                )
            )
        if self.args.swa:
            print(
                f"Using Stochastic Weight Averaging with learning rate {self.args.swa_lrs}"
            )
            callbacks.append(StochasticWeightAveraging(swa_lrs=self.args.swa_lrs))
        return callbacks

    def _create_lr_scheduler_params(self):
        if self.args.lr_scheduler is None:
            self.lr_scheduler_params = None
        else:
            self.lr_scheduler_params = {"name": self.args.lr_scheduler, "T_max": self.args.max_epochs}
        print(f"lr_scheduler_params: {self.lr_scheduler_params}")
    
    def _create_opt_params(self):
        if self.args.opt is None:
            raise ValueError("opt must be set")
        
        self.opt_params = {"name": self.args.opt}
        print(f"opt_params: {self.opt_params}")
    

    def _create_logger(self, model):

        if self.args.no_wandb:
            return True

        wandb_resume = True if self.args.resume else None
        print(f"wandb_resume: {wandb_resume}")
        logger = WandbLogger(
            project=self.args.log_dir,
            name=self.path_manager.modelname,
            id=self.uid,
            resume=wandb_resume,
            allow_val_change=wandb_resume,
        )

        logger.watch(model)
        wandb.init()
        wandb.config.update(self.args, allow_val_change=True)
        wandb.config.update({"basename": self.basename})
        # logger = TensorBoardLogger(args.log_dir,
        #                            name=basename,
        #                            version=uid)
        # logger.log_hyperparams(vars(args))
        # logger.log_graph(model)
        return logger

    def _create_trainer(self, stage, callbacks=None, logger=None):
        if stage == "training":
            if self.args.smoke_test:
                limit_train_batches = 4
                limit_val_batches = 4
                limit_test_batches = 4
            else:
                limit_train_batches = 1.0
                limit_val_batches = 1.0
                limit_test_batches = 1.0

            # Training
            trainer = pl.Trainer(
                accelerator=self.args.accelerator,  # auto
                strategy=self.args.strategy,  # auto
                devices=self.args.devices,  # auto
                num_nodes=self.args.num_nodes,  # 1
                max_epochs=self.args.max_epochs,
                min_epochs=self.args.min_epochs,
                logger=logger,
                log_every_n_steps=self.args.log_every_n_steps,
                check_val_every_n_epoch=self.args.check_val_every_n_epoch,
                val_check_interval=self.args.val_check_interval,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                limit_test_batches=limit_test_batches,
                callbacks=callbacks,
                precision=self.args.precision,
                deterministic=self.args.deterministic,
            )
            return trainer
        elif stage == "prediction":
            trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                fast_dev_run=2 if self.args.smoke_test else False,
                logger=False,
            )
            return trainer
        else:
            raise ValueError("stage must be 'training' or 'prediction'")

    def _tune_lr(self, trainer, model, dm):
        tuner = Tuner(trainer)
        tuner.lr_find(model, dm)
        print(f"New lr: {model.hparams.lr}")
        wandb.config.update({"new_lr": model.hparams.lr}, allow_val_change=True)

    def _save_config(self):
        with open(self.path_manager.config_path, "w") as f:
            f.write(yaml.dump(vars(wandb.config)["_items"]))
    
    
    def _handle_predictions(self, model, dm):
        if not self.args.tta:
            y_true = model.y_true
            y_pred = model.y_pred
            fnames = model.fnames
            logits = model.logits
        else:
            y_true = dm.tta_process(model.y_true)
            y_pred = dm.tta_process(model.y_pred)
            fnames = dm.tta_process(model.fnames)
            logits = dm.tta_process_output(model.logits)
        

        preds = TaxonomistPredictions()
        preds.set_y_true(y_true)
        preds.set_y_pred(y_pred)
        preds.set_fnames(fnames)
        preds.set_logits(logits)
        preds.set_class_map(self.class_map)
        breakpoint()

        if self.args.task == "regression":
            df = preds.get_y_true_y_pred()
        else:
            df = preds.get_full_df(softmax=self.args.return_softmax)
        
        out_fpath = self.path_manager.predict_fpath

        df.to_csv(out_fpath, index=True)
        print(out_fpath)
    
    def _handle_features(self, model, dm):
        y_true = model.y_true
        features = model.features
        fnames = model.fnames

        out_fpath = self.path_manager.predict_fpath
        with gzip.open(out_fpath, "wb") as f:
            pickle.dump({"fname": fnames, "y_true": y_true, "features": features}, f)
        print(out_fpath)

    def train(self):
        self.path_manager = PathManager("training", self.args, self.ckpt)

        dm = self._create_data_module()
        model = self._create_model(stage="training")
        callbacks = self._create_callbacks()

        logger = self._create_logger(model)

        trainer = self._create_trainer(stage="training", callbacks=callbacks, logger=logger)

        if self.args.auto_lr:
            self._tune_lr(trainer, model, dm)

        if not self.args.no_wandb:  # we can't access wandb.config
            self._save_config()

        trainer.fit(model, dm, ckpt_path=self.ckpt.ckpt_path if self.args.resume else None)
        trainer.test(model, datamodule=dm, ckpt_path="best")
        breakpoint()

        print(
            f"Best model: {callbacks[0].best_model_path} | score: {callbacks[0].best_model_score}"
        )
        return trainer

    def predict(self):
        self.path_manager = PathManager("prediction", self.args, self.ckpt)

        dm = self._create_data_module()
        model = self._create_model(stage="prediction")
        trainer = self._create_trainer(stage="prediction")
        
        # Predictions. Sets y_true, y_pred, fnames, and logits in the model
        if self.args.tta:
            trainer.test(model, dataloaders=dm.tta_dataloader())
        else:
            trainer.test(model, dm)

        if self.args.task == "feature-extraction":
            self._handle_features(model, dm)
        else:
            self._handle_predictions(model, dm)
