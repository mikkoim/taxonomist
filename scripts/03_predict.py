import argparse
import sys

import pandas as pd
import lightning.pytorch as pl
import taxonomist as src
import torch
from pathlib import Path


def main(args):
    parser = argparse.ArgumentParser()

    parser = src.add_dataset_args(parser)
    parser = src.add_dataloader_args(parser)
    parser = src.add_model_args(parser)
    parser = src.add_train_args(parser)
    parser = src.add_program_args(parser)

    args = parser.parse_args()

    user_arg_dict = vars(args)
    user_arg_dict["label_column"] = user_arg_dict.pop("label")
    user_arg_dict["timm_model_name"] = user_arg_dict.pop("model")
    user_arg_dict["class_map_name"] = user_arg_dict.pop("class_map")

    return src.LightningModelWrapper(src.LightningModelArguments(**user_arg_dict)).predict()

if __name__ == "__main__":
    trainer = main(sys.argv[1:])
    if False:
        parser = argparse.ArgumentParser()

        parser = src.add_dataset_args(parser)
        parser = src.add_dataloader_args(parser)
        parser = src.add_model_args(parser)
        parser = src.add_train_args(parser)
        parser = src.add_program_args(parser)

        args = parser.parse_args()

        user_arg_dict = vars(args)
        user_arg_dict["label_column"] = user_arg_dict.pop("label")
        user_arg_dict["timm_model_name"] = user_arg_dict.pop("model")
        user_arg_dict["class_map_name"] = user_arg_dict.pop("class_map")


        gpu_count = torch.cuda.device_count()

        tag = f"{args.aug}"
        if args.tta:
            tag += "_tta"
        out_folder = Path(args.ckpt_path).parents[0] / "predictions" / tag
        out_folder.mkdir(exist_ok=True, parents=True)

        ckpt = torch.load(
            args.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

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

        model = src.LitModule(**ckpt["hyper_parameters"])

        model.load_state_dict(ckpt["state_dict"])
        model.label_transform = class_map["inv"]
        model.freeze()

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            fast_dev_run=2 if args.smoke_test else False,
            logger=False,
        )

        if not args.tta:
            trainer.test(model, dm)
            y_true, y_pred = model.y_true, model.y_pred

        else:
            dm.setup()
            trainer.test(model, dataloaders=dm.tta_dataloader())
            y_true = dm.tta_process(model.y_true)
            y_pred = dm.tta_process(model.y_pred)

        dm.visualize_datasets(out_folder)

        df_pred = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        if n_classes > 1:
            softmax = model.softmax

            if args.tta:
                softmax = dm.tta_process_softmax(softmax)
            n_classes = softmax.shape[1]
            classes = class_map["inv"](list(range(n_classes)))
            df_prob = pd.DataFrame(data=softmax, columns=classes)
            df = pd.concat((df_pred, df_prob), axis=1)
        else:
            df = df_pred

        model_stem = Path(args.ckpt_path).stem
        out_stem = f"{args.out_prefix}_{model_stem}_{args.aug}"
        if args.tta:
            out_stem += "_tta"
        outname = out_stem + ".csv"
        df.to_csv(out_folder / outname, index=False)
