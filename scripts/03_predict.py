import argparse

import pandas as pd
import lightning.pytorch as pl
import taxonomist as src
import torch
from pathlib import Path
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = src.add_dataset_args(parser)
    parser = src.add_dataloader_args(parser)
    parser = src.add_model_args(parser)
    parser = src.add_train_args(parser)
    parser = src.add_program_args(parser)

    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()

    tag = f"{args.dataset_name}_{args.aug}"
    if args.tta:
        tag += "_tta"

    folder_type = "features" if args.feature_extraction else "predictions"

    if args.ckpt_path:
        model_stem = Path(args.ckpt_path).stem
        ckpt = torch.load(
            args.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        out_folder = Path(args.ckpt_path).parents[0] / folder_type / tag
    else:
        model_stem = args.model
        out_folder = (
            Path(args.out_folder)
            / args.dataset_name
            / model_stem
            / f"f{args.fold}"
            / folder_type
            / tag
        )

    out_folder.mkdir(exist_ok=True, parents=True)

    # Class / label map loading
    if args.class_map != "none":
        class_map = src.load_class_map(args.class_map)
        n_classes = len(class_map["fwd_dict"])
    else:
        class_map = {"fwd": None, "inv": None, "fwd_dict": None, "inv_dict": None}
        n_classes = 1

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
    )

    if args.feature_extraction:
        print("Loading a FeatureExtractionModule...")
        if args.ckpt_path:
            model = src.FeatureExtractionModule(
                feature_extraction_mode=args.feature_extraction,
                **ckpt["hyper_parameters"],
            )
        else:
            model = src.FeatureExtractionModule(
                feature_extraction_mode=args.feature_extraction,
                model=args.model,
                pretrained=True,
            )
    else:  # Normal prediction
        model = src.LitModule(**ckpt["hyper_parameters"])

    if args.ckpt_path:
        model.load_state_dict(ckpt["state_dict"])

    # Inverse class map loading
    if args.inverse_class_map == "same":
        model.label_transform = class_map["inv"]
    elif args.inverse_class_map == "none":
        model.label_transform = None
    else:
        raise ValueError("inverse_class_map must be 'same' or 'none")

    model.freeze()

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        fast_dev_run=2 if args.smoke_test else False,
        logger=False,
    )

    # Actual prediction
    if not args.tta:
        trainer.test(model, dm)
        y_true, y_pred = model.y_true, model.y_pred

    else:
        dm.setup()
        trainer.test(model, dataloaders=dm.tta_dataloader())
        y_true = dm.tta_process(model.y_true)
        y_pred = dm.tta_process(model.y_pred)

    dm.visualize_datasets(out_folder)

    out_stem = f"{args.out_prefix}_{model_stem}_{args.aug}"
    if args.tta:
        out_stem += "_tta"

    if not args.feature_extraction:  # Normal softmax or logit output
        df_pred = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        # Classification
        if n_classes > 1:
            if args.return_logits:
                output = model.logits
            else:
                output = model.softmax

            if args.tta:
                # Calculates the mean across tta repetitions
                output = dm.tta_process_output(output)
            n_classes = output.shape[1]

            # Handle out-of distribution prediction
            if args.inverse_class_map == "same":
                classes = class_map["inv"](list(range(n_classes)))
            elif args.inverse_class_map == "none":
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
        outname = f"{out_stem}_{args.feature_extraction}.p"
        with open(out_folder / outname, "wb") as f:
            pickle.dump({"y_true": y_true, "features": y_pred}, f)
        print(out_folder / outname)
