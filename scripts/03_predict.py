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