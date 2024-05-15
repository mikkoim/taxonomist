import argparse
import uuid
from datetime import datetime
from pathlib import Path
import yaml

import taxonomist as src
import lightning.pytorch as pl
import wandb
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner


DESCRIPTION = """
The main training script.
"""

if __name__ == "__main__":
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

    if not args.resume:
        uid = datetime.now().strftime("%y%m%d-%H%M") + f"-{str(uuid.uuid4())[:4]}"
    else:
        if not args.ckpt_path:
            raise ValueError("When resuming, a ckpt_path must be set")
        # Parse the uid from filename
        print(f"Using checkpoint from {args.ckpt_path}")
        ckpt_name = Path(args.ckpt_path).stem
        uid = ckpt_name.split("_")[-3]
        assert basename == "_".join(ckpt_name.split("_")[:-5])

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
        class_map_path=args.class_map
    )

    # Calculate class counts
    #class_counts = dm.calculate_and_get_class_counts()
    # Manually call setup method for 'fit' stage
    dm.setup(stage='fit')

    # Now you can directly access self.class_counts
    class_counts = dm.class_counts
    print(class_counts, args.fold, 22)


    opt_args = {"name": args.opt}

    model = src.LitModule(
        model=args.model,
        freeze_base=args.freeze_base,
        pretrained=args.pretrained,
        criterion=args.criterion,
        opt=opt_args,
        n_classes=n_classes,
        lr=args.lr,
        label_transform=class_map["inv"],
        class_counts=class_counts,  # Now passed here
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
        wandb.init()
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
