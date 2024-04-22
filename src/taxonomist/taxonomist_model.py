import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

import wandb

from .data import LitDataModule
from .model import FeatureExtractionModule, LitModule
from .utils import load_class_map


@dataclass(frozen=True)
class TaxonomistModelArguments:
    data_folder: str
    dataset_config_path: str
    dataset_name: str
    csv_path: str

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
    return_logits: bool = False

    min_epochs: Optional[int] = None
    max_epochs: Optional[int] = None
    early_stopping: bool = False
    early_stopping_patience: int = 5  # used if early_stopping=True
    lr: float = 1e-4
    opt: str = "adam"
    lr_scheduler: str = None

    auto_lr: bool = False
    precision: int = 32
    deterministic: bool = False
    resume: bool = False

    log_dir: str = "logs"
    out_folder: str = "outputs"
    out_prefix: str = "metrics"
    random_state: int = 42
    debug: bool = False
    smoke_test: bool = False

    log_every_n_steps: Optional[int] = 10
    suffix = None


class TaxonomistModel:
    def __init__(self, args: TaxonomistModelArguments):
        self.args = args
        self.basename = f"{args.out_prefix}_{args.timm_model_name}"
        self.uid = self._parse_uid()
        self.outname = f"{self.basename}_f{args.fold}_{self.uid}"

        if args.deterministic:
            pl.seed_everything(seed=args.random_state, workers=True)

    def _parse_uid(self):
        # It is possible to resume to an existing run that was cancelled/stopped if
        # argument ckpt_path is provided that contains the weights of when the run was
        # stopped/cancelled
        if not self.args.resume:
            uid = datetime.now().strftime("%y%m%d-%H%M") + f"-{str(uuid.uuid4())[:4]}"
        else:
            if not self.args.ckpt_path:
                raise ValueError("When resuming, a ckpt_path must be set")
            # Parse the uid from filename
            print(f"Using checkpoint from {self.args.ckpt_path}")
            ckpt_name = Path(self.args.ckpt_path).stem
            uid = ckpt_name.split("_")[-3]
            assert self.basename == "_".join(ckpt_name.split("_")[:-4])
        return uid

    def _create_out_folder(self, training=True):
        if training:
            out_folder = (
                Path(self.args.out_folder)
                / Path(self.args.dataset_name)
                / self.basename
                / f"f{self.args.fold}"
            )
        else:
            tag = f"{self.args.dataset_name}_{self.args.aug}"
            if self.args.tta:
                tag += "_tta"
            folder_type = "features" if self.args.feature_extraction else "predictions"
            if self.args.ckpt_path:
                out_folder = Path(self.args.ckpt_path).parents[0] / folder_type / tag
            else:
                model_stem = self.args.model
                out_folder = (
                    Path(self.args.out_folder)
                    / self.args.dataset_name
                    / model_stem
                    / f"f{self.args.fold}"
                    / folder_type
                    / tag
                )

        out_folder.mkdir(exist_ok=True, parents=True)
        return out_folder

    def _create_prediction_out_folder(self):
        tag = f"{self.args.dataset_name}_{self.args.aug}"
        if self.args.tta:
            tag += "_tta"

        out_folder = Path(self.args.ckpt_path).parents[0] / "predictions" / tag
        out_folder.mkdir(exist_ok=True, parents=True)

    def _load_class_map(self):
        # Class / label map loading
        if (self.args.class_map_name is not None) and (
            self.args.class_map_name != "none"
        ):
            class_map = load_class_map(self.args.class_map_name)
            n_classes = len(class_map["fwd_dict"])
        else:
            class_map = {"fwd": None, "inv": None, "fwd_dict": None, "inv_dict": None}
            n_classes = 1
        return class_map, n_classes

    def _create_data_module(self, class_map):
        dm = LitDataModule(
            data_folder=self.args.data_folder,
            dataset_config_path=self.args.dataset_config_path,
            dataset_name=self.args.dataset_name,
            csv_path=self.args.csv_path,
            fold=self.args.fold,
            label=self.args.label_column,
            label_transform=class_map["fwd"],
            imsize=self.args.imsize,
            batch_size=self.args.batch_size,
            aug=self.args.aug,
            load_to_memory=self.args.load_to_memory,
            tta_n=self.args.tta_n,
        )
        return dm

    def _create_model(self, n_classes, class_map, lr_scheduler=None, ckpt=None, training=True):
        if training:
            model = LitModule(
                model=self.args.timm_model_name,
                freeze_base=self.args.freeze_base,
                pretrained=self.args.pretrained,
                criterion=self.args.criterion,
                opt={"name": self.args.opt},
                n_classes=n_classes,
                lr=self.args.lr,
                lr_scheduler=lr_scheduler,
                label_transform=class_map["inv"],
            )
            return model
        else:
            if self.args.feature_extraction:
                print("Loading a FeatureExtractionModule...")
                if self.args.ckpt_path:
                    model = FeatureExtractionModule(
                        feature_extraction_mode=self.args.feature_extraction,
                        **ckpt["hyper_parameters"],
                    )
                else:
                    model = FeatureExtractionModule(
                        feature_extraction_mode=self.args.feature_extraction,
                        model=self.args.model,
                        pretrained=True,
                    )
            else:  # Normal prediction
                model = LitModule(**ckpt["hyper_parameters"])

            if self.args.ckpt_path:
                model.load_state_dict(ckpt["state_dict"])

            # Inverse class map loading
            if self.args.inverse_class_map == "same":
                model.label_transform = class_map["inv"]
            elif self.args.inverse_class_map == "none":
                model.label_transform = None
            else:
                raise ValueError("inverse_class_map must be 'same' or 'none")

            model.freeze()
            return model

    def _load_model(self, ckpt):
        return LitModule(**ckpt["hyper_parameters"])

    def _load_checkpoint(self, model):
        ckpt = torch.load(
            self.args.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.load_state_dict(ckpt["state_dict"])

    def _create_callbacks(self, out_folder):
        # Training callbacks
        checkpoint_callback_best = ModelCheckpoint(
            monitor="val/loss",
            dirpath=out_folder,
            filename=f"{self.outname}_" + "epoch{epoch:02d}_val-loss{val/loss:.2f}",
            auto_insert_metric_name=False,
        )
        checkpoint_callback_last = ModelCheckpoint(
            monitor="epoch",
            mode="max",
            dirpath=out_folder,
            filename=f"{self.outname}_"
            + "epoch{epoch:02d}_val-loss{val/loss:.2f}_last",
            auto_insert_metric_name=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [checkpoint_callback_best, checkpoint_callback_last, lr_monitor]
        if self.args.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss", patience=self.args.early_stopping_patience
                )
            )
        return callbacks
    
    def _create_lr_scheduler(self):
        if self.args.lr_scheduler is None:
            return None
        
        lr_scheduler = {"name": self.args.lr_scheduler,
                        "T_max": self.args.max_epochs}
        print(f"Set lr scheduler: {lr_scheduler}")
        return lr_scheduler

    def _create_logger(self, model):
        wandb_resume = True if self.args.resume else None
        print(wandb_resume)
        logger = WandbLogger(
            project=self.args.log_dir,
            name=self.outname,
            id=self.uid,
            resume=wandb_resume,
            allow_val_change=wandb_resume,
        )

        logger.watch(model)
        wandb.init()
        wandb.config.update(self.args, allow_val_change=True)
        # logger = TensorBoardLogger(args.log_dir,
        #                            name=basename,
        #                            version=uid)
        # logger.log_hyperparams(vars(args))
        # logger.log_graph(model)
        return logger

    def _create_trainer(self, callbacks, logger, training=True):
        if training:
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
                max_epochs=self.args.max_epochs,
                min_epochs=self.args.min_epochs,
                logger=logger,
                log_every_n_steps=10,
                devices="auto",
                accelerator="auto",
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                limit_test_batches=limit_test_batches,
                callbacks=callbacks,
                precision=self.args.precision,
                deterministic=self.args.deterministic,
            )
            return trainer
        else:
            trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                fast_dev_run=2 if self.args.smoke_test else False,
                logger=False,
            )
            return trainer

    def _perform_training(self, trainer, model, dm, resume_ckpt):
        trainer.fit(model, dm, ckpt_path=resume_ckpt)
        trainer.test(model, datamodule=dm, ckpt_path="best")

    def _tune_lr(self, trainer, model, dm):
        tuner = Tuner(trainer)
        tuner.lr_find(model, dm)
        print(f"New lr: {model.hparams.lr}")
        wandb.config.update({"new_lr": model.hparams.lr}, allow_val_change=True)

    def _save_config(self, out_folder, uid):
        with open(out_folder / f"config_{uid}.yml", "w") as f:
            f.write(yaml.dump(vars(wandb.config)["_items"]))

    def _predict(self, trainer, model, dm, class_map, n_classes, out_folder=None):
        # Actual prediction
        if not self.args.tta:
            trainer.test(model, dm)
            y_true, y_pred = model.y_true, model.y_pred

        else:
            dm.setup()
            trainer.test(model, dataloaders=dm.tta_dataloader())
            y_true = dm.tta_process(model.y_true)
            y_pred = dm.tta_process(model.y_pred)

        if out_folder:
            if self.args.ckpt_path:
                model_stem = Path(self.args.ckpt_path).stem
            else:
                model_stem = self.args.model

            out_stem = f"{self.args.out_prefix}_{model_stem}_{self.args.aug}"
            if self.args.tta:
                out_stem += "_tta"
            if not self.args.feature_extraction:  # Normal softmax or logit output
                df_pred = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

                # Classification
                if n_classes > 1:
                    if self.args.return_logits:
                        output = model.logits
                    else:
                        output = model.softmax

                    if self.args.tta:
                        # Calculates the mean across tta repetitions
                        output = dm.tta_process_output(output)
                    n_classes = output.shape[1]

                    # Handle out-of distribution prediction
                    if self.args.inverse_class_map == "same":
                        classes = class_map["inv"](list(range(n_classes)))
                    elif self.args.inverse_class_map == "none":
                        classes = range(output.shape[1])

                    df_prob = pd.DataFrame(data=output, columns=classes)
                    df = pd.concat((df_pred, df_prob), axis=1)

                # Regression
                else:
                    df = df_pred

                outname = out_stem + ".csv"
                df.to_csv(out_folder / outname, index=False)
                print(out_folder / outname)
            else:  # Outputs of feature extraction can vary depending on the pooling
                outname = f"{out_stem}_{self.args.feature_extraction}.p"
                with open(out_folder / outname, "wb") as f:
                    pickle.dump({"y_true": y_true, "features": y_pred}, f)
                print(out_folder / outname)

        return df

    def train_model(self):
        # initialize and get folder where the parameters are saved
        out_folder = self._create_out_folder()

        # get class mapping
        class_map, n_classes = self._load_class_map()

        # get data module
        dm = self._create_data_module(class_map)

        # create lr scheduler
        lr_scheduler = self._create_lr_scheduler()

        # get model
        model = self._create_model(n_classes, class_map, lr_scheduler)

        if (not self.args.resume) and self.args.ckpt_path:
            self._load_checkpoint(model)
            resume_ckpt = None
        else:
            resume_ckpt = self.args.ckpt_path

        callbacks = self._create_callbacks(out_folder)

        if not self.args.debug:
            logger = self._create_logger(model)
        else:
            logger = True

        trainer = self._create_trainer(callbacks, logger)

        if self.args.auto_lr:
            self._tune_lr(trainer, model, dm)

        if not self.args.debug:  # In debug because we can't access wandb.config
            self._save_config(out_folder, self.uid)

        self._perform_training(trainer, model, dm, resume_ckpt)

        dm.visualize_datasets(out_folder / f"aug-{self.args.aug}-{self.uid}")

        print(
            f"Best model: {callbacks[0].best_model_path} | score: {callbacks[0].best_model_score}"
        )
        return trainer

    def predict(self):
        # gpu_count = torch.cuda.device_count()

        out_folder = self._create_out_folder(training=False)

        ckpt = torch.load(
            self.args.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        class_map, n_classes = self._load_class_map()

        dm = self._create_data_module(class_map)

        model = self._create_model(None, class_map, ckpt, training=False)

        trainer = self._create_trainer(None, None, training=False)

        _ = self._predict(
            trainer, model, dm, class_map, n_classes, out_folder=out_folder
        )

        dm.visualize_datasets(out_folder)
