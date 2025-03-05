import pickle
import uuid
import gzip
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

from .input_config import TaxonomistModelArguments
from .data import LitDataModule
from .model import FeatureExtractionModule, LitModule
from .utils import load_class_map
from .predictions import TaxonomistPredictions


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
            raise ValueError(
                f"The checkpoint path '{str(self.ckpt_path)}' does not exist"
            )

        self.ckpt = torch.load(
            self.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=True
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
    """
    Manages file paths and folder structures for training and prediction tasks.

    Attributes:
        stage (str): The operational stage, either 'training' or 'prediction'.
        args (TaxonomistModelArguments): Input arguments specifying model configurations.
        ckpt (TaxonomistCheckpoint, optional): An optional checkpoint object for resuming training or prediction.
    """

    def __init__(
        self,
        stage: str,
        args: TaxonomistModelArguments,
        ckpt: TaxonomistCheckpoint = None,
    ):
        """
        Initialize the PathManager with the given stage, arguments, and checkpoint.

        Args:
            stage (str): The operational stage, either 'training' or 'prediction'.
            args (TaxonomistModelArguments): Input arguments specifying model configurations.
            ckpt (TaxonomistCheckpoint, optional): An optional checkpoint object for resuming training or prediction.
        """
        self.stage = stage
        self.args = args
        self.ckpt = ckpt

        if (self.args.ckpt_path is not None) and (ckpt is None):
            raise ValueError(
                "A checkpoint path was provided but the checkpoint object is None"
            )

        self._create_names()
        if stage == "training":
            self._create_train_out_folder()
        elif stage == "prediction":
            self._create_pred_out_folder()
            self.predict_fpath = self._create_predict_fpath(self.args.task)

        self.visualization_path = self.out_folder / f"aug-{self.args.aug}-{self.uid}"
        self.config_path = self.out_folder / f"config_{self.uid}.yml"

    @property
    def has_checkpoint(self):
        """
        Check if the model has a checkpoint.
        """
        return self.ckpt is not None

    @property
    def tag(self):
        """
        Get the tag for the model based on the dataset name and augmentation.

        Returns:
            str: The tag for the model.
        """
        tag = f"{self.args.dataset_name}_{self.args.aug}"
        if self.args.tta:
            tag += "_tta"
        return tag

    def _create_train_out_folder(self):
        """
        Creates an output folder for training.

        The output folder is created based on the dataset name, model name, and fold.

        Side Effects:
            - Sets `out_folder`.
        """
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
        Creates an output folder for prediction.

        The output folder is created based on the dataset name, model name, and task.

        Behavior:
            - If the model has a checkpoint, the output folder is created in the checkpoint folder.
            - If the model does not have a checkpoint, the output folder is created based on the input arguments.

        Side Effects:
            - Sets `out_folder`.
        """
        if (self.args.task == "classification") or (self.args.task == "regression"):
            folder_name = "predictions"
        elif self.args.task == "feature-extraction":
            folder_name = "features"
        else:
            raise ValueError(
                "Task must be 'classification', 'regression', or 'feature-extraction'"
            )

        if self.has_checkpoint:
            out_folder = Path(self.ckpt.folder, folder_name, self.tag)
        else:
            out_folder = (
                Path(self.args.out_folder)
                / str(self.args.dataset_name)
                / str(self.args.model_name)
                / f"f{self.args.fold}"
                / folder_name
                / self.tag
            )

        out_folder.mkdir(exist_ok=True, parents=True)
        self.out_folder = out_folder
        print(f"Output folder: {out_folder}")

    def _create_predict_fpath(self, task: str) -> Path:
        """
        Creates the output file path for predictions based on the task.

        Returns:
            Path: The output file path for predictions.
        """

        if task == "feature-extraction":
            name = f"{self.modelname}_{self.args.feature_extraction}.p.gz"
            return self.out_folder / name
        elif (task == "classification") or (task == "regression"):
            name = f"{self.ckpt.name}_{self.args.aug}"
            if self.args.tta:
                name += "_tta"
            return Path(self.out_folder, name + ".csv")
        else:
            raise ValueError(
                f"Task must be 'classification', 'regression', or 'feature-extraction'. Got {task}"
            )

    def _create_names(self):
        """
        Creates the `basename`, `modelname`, and `uid` for the model based on the stage and arguments.

        Behavior:
            - If the stage is 'training' and the model is resuming, the names are taken from the checkpoint.
            - If the stage is 'training' and the model is not resuming, new names are created.
            - If the stage is 'prediction' and the model has no checkpoint, new names are created.
            - If the stage is 'prediction' and the model has a checkpoint, the names are taken from the checkpoint.

        Raises:
            ValueError: If the input arguments do not match the checkpoint basename when resuming.

        Side Effects:
            - Sets `modelname`, `basename`, and `uid`.

        """
        if self.stage == "training":
            if self.args.resume:
                if (
                    f"{self.args.out_prefix}_{self.args.model_name}"
                    != self.ckpt.basename
                ):
                    raise ValueError(
                        f"Input arguments {self.args.out_prefix} and {self.args.model_name} do not match checkpoint basename {self.ckpt.basename}."
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

        UID is created based on the current time and a random UUID.

        Side Effects:
            - Sets `basename`, `modelname`, and `uid`.
        """
        uid = datetime.now().strftime("%y%m%d-%H%M") + f"-{str(uuid.uuid4())[:4]}"
        self.uid = uid
        self.basename = f"{self.args.out_prefix}_{self.args.model_name}"
        self.modelname = f"{self.basename}_f{self.args.fold}_{uid}"


class TaxonomistModel:
    """
    The taxonomist main class for training and prediction.

    Attributes:
        args (TaxonomistModelArguments): Input arguments specifying model configurations.
        ckpt (TaxonomistCheckpoint): The checkpoint object.
        has_checkpoint (bool): True if the model has a checkpoint, False otherwise.
        class_map (dict): The class map for the model.
        n_classes (int): The number of classes

    Methods:
        train: Perform the training.
        predict: Perform the prediction.
    """

    def __init__(self, args: TaxonomistModelArguments):
        """
        Initialize the TaxonomistModel with the given arguments.

        Args:
            args (TaxonomistModelArguments): Input arguments specifying model configurations.
        """
        self.args = args
        self._handle_checkpoint()
        self._handle_class_map()

        if args.deterministic:
            pl.seed_everything(seed=args.random_state, workers=True)

    def _handle_checkpoint(self):
        """
        Creates a TaxonomistCheckpoint object.

        Side Effects:
            - Sets `ckpt` and `has_checkpoint`.
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

        Side Effects:
            - Sets `class_map` and `n_classes`.
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

    def _create_data_module(self, visualize=True):
        """
        Creates a LitDataModule object for the model.

        Side Effects:
            - Visualizes the datasets.

        Returns:
            LitDataModule: The LitDataModule object.
        """
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
            n_classes=self.n_classes,
            mixup=self.args.mixup,
        )
        dm.setup()
        if visualize:
            dm.visualize_datasets(self.path_manager.visualization_path)
        return dm

    def _load_checkpoint(self, model):
        """
        Loads model weights from a checkpoint.

        Handles mismatches in the last layer by initializing a new projection head.

        Side Effects:
            - Modifies model weights
            - Initializes a new projection head if there is a mismatch
        """
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

        print(f"Loaded model weights from {self.ckpt.ckpt_path}")

    def _create_train_model(self):
        """
        Creates a LitModule object for training.

        Side Effects:
            - Loads model weights from a checkpoint if not resuming and a checkpoint exists.
        """
        self._create_lr_scheduler_params()
        self._create_opt_params()

        model = LitModule(
            model=self.args.model_name,
            custom_model=self.args.custom_model,
            dataset_config_path=self.args.dataset_config_path,
            freeze_base=self.args.freeze_base,
            pretrained=self.args.pretrained,
            criterion=self.args.criterion,
            opt=self.opt_params,
            n_classes=self.n_classes,
            lr=self.args.lr,
            lr_scheduler=self.lr_scheduler_params,
            label_transform=self.class_map["inv"],
            no_train_metrics=True if self.args.mixup else False,
        )

        if self.has_checkpoint and (self.args.resume is False):
            self._load_checkpoint(model)
        return model

    def _create_predict_model(self):
        """
        Creates a LitModule object for prediction.

        Side Effects:
            - Loads model weights from a checkpoint if a checkpoint exists.
        """
        model = LitModule(**self.ckpt.ckpt["hyper_parameters"])

        if self.has_checkpoint:
            self._load_checkpoint(model)

        if self.args.inverse_class_map == "none":
            model.label_transform = None
        else:
            model.label_transform = self.class_map["inv"]

        model.freeze()
        return model

    def _create_feature_extraction_model(self):
        """
        Creates a FeatureExtractionModule object for feature extraction.

        Side Effects:
            - Loads model weights from a checkpoint if a checkpoint exists.
        """
        if self.has_checkpoint:
            model = FeatureExtractionModule(
                feature_extraction_mode=self.args.feature_extraction,
                **self.ckpt.ckpt["hyper_parameters"],
            )
            self._load_checkpoint(model)
        else:
            model = FeatureExtractionModule(
                feature_extraction_mode=self.args.feature_extraction,
                model=self.args.model_name,
                pretrained=True,
            )
        model.freeze()
        return model

    def _create_model(self, stage: str) -> Union[LitModule, FeatureExtractionModule]:
        """
        Handler for creating a model based on the stage.

        Args:
            stage (str): The operational stage, either 'training' or 'prediction'.

        Returns:
            Union[LitModule, FeatureExtractionModule]: The model object.
        """
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

    def _create_callbacks(self):
        """
        Creates a list of callbacks for the model.

        Returns:
            List: A list of callbacks."""
        # Best model saving
        checkpoint_callback_best = ModelCheckpoint(
            monitor="val/loss",
            dirpath=self.path_manager.out_folder,
            save_top_k=self.args.save_top_k,
            filename=f"{self.path_manager.modelname}_"
            + "epoch{epoch:02d}_val-loss{val/loss:.2f}",
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
        """
        Creates the learning rate scheduler parameters for the model.
        """
        if self.args.lr_scheduler is None:
            self.lr_scheduler_params = None
        else:
            self.lr_scheduler_params = {
                "name": self.args.lr_scheduler,
                "T_max": self.args.max_epochs,
            }
        print(f"lr_scheduler_params: {self.lr_scheduler_params}")

    def _create_opt_params(self):
        """
        Creates the optimizer parameters for the model.
        """
        if self.args.opt is None:
            raise ValueError("opt must be set")

        self.opt_params = {"name": self.args.opt,
                           "lr": self.args.lr,
                           "weight_decay": self.args.opt_weight_decay,
                           "momentum": self.args.opt_momentum,
                           "beta1": self.args.opt_beta1,
                           "beta2": self.args.opt_beta2}
        print(f"opt_params: {self.opt_params}")

    def _create_logger(self, model):
        """
        Creates a logger for the model.
        """

        if self.args.no_wandb:
            return True

        wandb_resume = True if self.args.resume else None
        print(f"wandb_resume: {wandb_resume}")
        logger = WandbLogger(
            project=self.args.log_dir,
            name=self.path_manager.modelname,
            id=self.path_manager.uid,
            resume=wandb_resume,
            allow_val_change=wandb_resume,
        )

        logger.watch(model)
        wandb.init()
        wandb.config.update(self.args, allow_val_change=True)
        wandb.config.update({"basename": self.path_manager.basename})
        # logger = TensorBoardLogger(args.log_dir,
        #                            name=basename,
        #                            version=uid)
        # logger.log_hyperparams(vars(args))
        # logger.log_graph(model)
        return logger

    def _create_trainer(self, stage, callbacks=None, logger=None):
        """
        Creates a trainer for the model.

        Args:
            stage (str): The operational stage, either 'training' or 'prediction'.
            callbacks (List, optional): A list of callbacks for the model.
            logger (WandbLogger, optional): A logger for the model.
        Returns:
            Trainer: A trainer object.
        """
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
        """
        Tunes the learning rate for the model.
        """
        tuner = Tuner(trainer)
        tuner.lr_find(model, dm)
        print(f"New lr: {model.hparams.lr}")
        wandb.config.update({"new_lr": model.hparams.lr}, allow_val_change=True)

    def _save_config(self):
        """Saves the configuration to a YAML file."""
        with open(self.path_manager.config_path, "w") as f:
            f.write(yaml.dump(vars(wandb.config)["_items"]))

    def _handle_predictions(self, model, dm):
        if not self.args.tta:
            y_true = model.y_true
            y_pred = model.y_pred
            fnames = model.fnames
            if self.args.task == "classification":
                logits = model.logits
        else:
            y_true = dm.tta_process(model.y_true)
            y_pred = dm.tta_process(model.y_pred)
            fnames = dm.tta_process(model.fnames)
            if self.args.task == "classification":
                logits = dm.tta_process_output(model.logits)

        preds = TaxonomistPredictions()
        preds.set_y_true(y_true)
        preds.set_y_pred(y_pred)
        preds.set_fnames(fnames)
        if self.args.task == "classification":
            preds.set_logits(logits)
            preds.set_class_map(self.class_map)

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
        """
        Perform the training.
        """

        # Setup the path manager
        self.path_manager = PathManager("training", self.args, self.ckpt)

        # Setup the data and the model
        dm = self._create_data_module(visualize=False if self.args.custom_dataset else True)
        model = self._create_model(stage="training")
        callbacks = self._create_callbacks()

        logger = self._create_logger(model)

        trainer = self._create_trainer(
            stage="training", callbacks=callbacks, logger=logger
        )

        if self.args.auto_lr:
            self._tune_lr(trainer, model, dm)

        if not self.args.no_wandb:  # we can't access wandb.config
            self._save_config()

        trainer.fit(
            model, dm, ckpt_path=self.ckpt.ckpt_path if self.args.resume else None
        )
        trainer.test(model, datamodule=dm, ckpt_path="best")

        print(
            f"Best model: {callbacks[0].best_model_path} | score: {callbacks[0].best_model_score}"
        )

    def predict(self):
        self.path_manager = PathManager("prediction", self.args, self.ckpt)

        dm = self._create_data_module(visualize=False if self.args.custom_dataset else True)
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
