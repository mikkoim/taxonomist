import argparse
import uuid
from datetime import datetime
from pathlib import Path
import yaml
import sys

import taxonomist as src
import lightning.pytorch as pl
import wandb
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
import os


DESCRIPTION = """
The main training script.
"""

def main(args):
    os.environ['http_proxy'] = 'http://gate102.vyh.fi:81'
    os.environ['https_proxy'] = 'http://gate102.vyh.fi:81'
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser = src.add_dataset_args(parser)
    parser = src.add_dataloader_args(parser)
    parser = src.add_model_args(parser)
    parser = src.add_train_args(parser)
    parser = src.add_program_args(parser)

    args = parser.parse_args(args)
    #to dict: vars(args)

    user_arg_dict = vars(args)
    user_arg_dict["label_column"] = user_arg_dict.pop("label")
    user_arg_dict["timm_model_name"] = user_arg_dict.pop("model")
    user_arg_dict["class_map_name"] = user_arg_dict.pop("class_map")
    return src.LightningModelWrapper(src.LightningModelArguments(**user_arg_dict)).train_model()

if __name__ == "__main__":
    trainer = main(sys.argv[1:]) # returns pytorch lightning trainer that has been trained.
    if False:
        os.environ['http_proxy'] = 'http://gate102.vyh.fi:81'
        os.environ['https_proxy'] = 'http://gate102.vyh.fi:81'
        parser = argparse.ArgumentParser(description=DESCRIPTION)

        parser = src.add_dataset_args(parser)
        parser = src.add_dataloader_args(parser)
        parser = src.add_model_args(parser)
        parser = src.add_train_args(parser)
        parser = src.add_program_args(parser)

        args = parser.parse_args()

        # Run name parsing
        basename = f"{args.out_prefix}_{args.model}"
        args.basename = basename

        # It is possible to resume to an existing run that was cancelled/stopped if argument ckpt_path is provided that contains the weights of when the run was stopped/cancelled
        if not args.resume:
            uid = datetime.now().strftime("%y%m%d-%H%M") + f"-{str(uuid.uuid4())[:4]}"
        else:
            if not args.ckpt_path:
                raise ValueError("When resuming, a ckpt_path must be set")
            # Parse the uid from filename
            print(f"Using checkpoint from {args.ckpt_path}")
            ckpt_name = Path(args.ckpt_path).stem
            uid = ckpt_name.split("_")[-3]
            assert basename == "_".join(ckpt_name.split("_")[:-4])

        outname = f"{basename}_f{args.fold}_{uid}"

        out_folder = (
            Path(args.out_folder) / Path(args.dataset_name) / basename / f"f{args.fold}"
        )
        out_folder.mkdir(exist_ok=True, parents=True)

        # Class / label map loading
        if args.class_map != "none":
            class_map = src.load_class_map(args.class_map)
            n_classes = len(class_map["fwd_dict"])
        else:
            class_map = {"fwd": None, "inv": None, "fwd_dict": None, "inv_dict": None}
            n_classes = 1

        if args.deterministic:
            pl.seed_everything(seed=args.global_seed)

        # Data and model

        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html 
        # A datamodule encapsulates the five steps involved in data processing in PyTorch: 1) Download / tokenize / process. 2) Clean and (maybe) save to disk. 3) Load inside Dataset. 4) Apply transforms (rotate, tokenize, etc…). 5) Wrap inside a DataLoader.
        dm = src.LitDataModule(
            data_folder=args.data_folder,
            dataset_name=args.dataset_name,
            csv_path=args.csv_path,
            fold=args.fold,
            label=args.label,
            label_transform=class_map["fwd"],
            imsize=args.imsize,
            batch_size=args.batch_size,
            aug=args.aug,
            load_to_memory=args.load_to_memory,
            tta_n=args.tta_n
        )

        opt_args = {"name": args.opt}

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        # A LightningModule organizes your PyTorch code into 6 sections: 1) Initialization (__init__ and setup()). 2) Train Loop (training_step()) 3) Validation Loop (validation_step()) 4) Test Loop (test_step()) 5) Prediction Loop (predict_step()) 6) Optimizers and LR Schedulers (configure_optimizers())
        model = src.LitModule(
            model=args.model,
            freeze_base=args.freeze_base,
            pretrained=args.pretrained,
            criterion=args.criterion,
            opt=opt_args,
            n_classes=n_classes,
            lr=args.lr,
            label_transform=class_map["inv"],
        )

        # If using pretrained weights but not resuming a run
        if (not args.resume) and args.ckpt_path:
            ckpt = torch.load(
                args.ckpt_path,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            model.load_state_dict(ckpt["state_dict"])
            resume_ckpt = None

        # If resuming a run
        else:
            resume_ckpt = args.ckpt_path

        # Training callbacks
        checkpoint_callback_best = ModelCheckpoint(
            monitor="val/loss",
            dirpath=out_folder,
            filename=f"{outname}_" + "epoch{epoch:02d}_val-loss{val/loss:.2f}",
            auto_insert_metric_name=False,
        )
        checkpoint_callback_last = ModelCheckpoint(
            monitor="epoch",
            mode="max",
            dirpath=out_folder,
            filename=f"{outname}_" + "epoch{epoch:02d}_val-loss{val/loss:.2f}_last",
            auto_insert_metric_name=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [checkpoint_callback_best, checkpoint_callback_last, lr_monitor]

        if args.early_stopping:
            callbacks.append(
                EarlyStopping(monitor="val/loss", patience=args.early_stopping_patience)
            )

        if not args.debug:
            wandb_resume = True if args.resume else None
            print(wandb_resume)
            logger = WandbLogger(
                project=args.log_dir,
                name=outname,
                id=uid,
                resume=wandb_resume,
                allow_val_change=wandb_resume,
            )

            logger.watch(model)
            wandb.config.update(args, allow_val_change=True)
            # logger = TensorBoardLogger(args.log_dir,
            #                            name=basename,
            #                            version=uid)
            # logger.log_hyperparams(vars(args))
            # logger.log_graph(model)
        else:
            logger = True

        if args.smoke_test:
            limit_train_batches = 4
            limit_val_batches = 4
            limit_test_batches = 4
        else:
            limit_train_batches = 1.0
            limit_val_batches = 1.0
            limit_test_batches = 1.0

        # Training
        #  https://lightning.ai/docs/pytorch/stable/common/trainer.html
        # The Lightning Trainer does much more than just “training”. Under the hood, it handles all loop details for you, some examples include: 1) Automatically enabling/disabling grads 2) Running the training, validation and test dataloaders 3) Calling the Callbacks at the appropriate times 4) Putting batches and computations on the correct devices
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            logger=logger,
            log_every_n_steps=10,
            devices="auto",
            accelerator="auto",
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            callbacks=callbacks,
            precision=args.precision,
            deterministic=args.deterministic,
        )

        if args.auto_lr:
            tuner = Tuner(trainer)
            tuner.lr_find(model, dm)
            print(f"New lr: {model.hparams.lr}")
            wandb.config.update({"new_lr": model.hparams.lr}, allow_val_change=True)

        if not args.debug:  # In debug because we can't access wandb.config
            with open(out_folder / f"config_{uid}.yml", "w") as f:
                f.write(yaml.dump(vars(wandb.config)["_items"]))

        trainer.fit(model, dm, ckpt_path=resume_ckpt)
        trainer.test(model, datamodule=dm, ckpt_path="best")

        dm.visualize_datasets(out_folder / f"aug-{args.aug}-{uid}")

        print(
            f"Best model: {checkpoint_callback_best.best_model_path} | score: {checkpoint_callback_best.best_model_score}"
        )
