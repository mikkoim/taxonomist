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
    return src.TaxonomistModel(src.TaxonomistModelArguments(**user_arg_dict)).train_model()

if __name__ == "__main__":
    trainer = main(sys.argv[1:]) # returns pytorch lightning trainer that has been trained.
