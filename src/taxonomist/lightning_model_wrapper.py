from pathlib import Path
import uuid
from datetime import datetime
import yaml
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning.pytorch as pl
from albumentations.pytorch.transforms import ToTensorV2
import timm
from sklearn.metrics import accuracy_score, f1_score
import wandb


def cross_entropy(output, target):
    loss = F.cross_entropy(output, target.long())
    return loss


def choose_criterion(name):
    if name == "cross-entropy":
        return cross_entropy
    else:
        raise Exception(f"Invalid criterion name '{name}'")


class Model(nn.Module):
    """PyTorch module for an arbitary timm model, separating the base and projection head"""

    def __init__(
        self,
        model: str = "resnet18",
        freeze_base: bool = True,
        pretrained: bool = True,
        n_classes: int = 1,
    ):
        """Initializes the model

        Args:
            model (str): name of the model to use
            freeze_base (bool): if True, the base is frozen
            pretrained (bool): if True, use pretrained weights
            n_classes (int): output layer size
        """
        super().__init__()

        self.h_dim = (
            timm.create_model(model, pretrained=False).get_classifier().in_features
        )
        self.base_model = timm.create_model(model, num_classes=0, pretrained=pretrained)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.proj_head = nn.Sequential(nn.Linear(self.h_dim, n_classes))

    def forward(self, x):
        h = self.base_model(x)
        return self.proj_head(h)

    def base_forward(self, x):
        return self.base_model(x)

    def proj_forward(self, h):
        return self.proj_head(h)


class LitModule(pl.LightningModule):
    """PyTorch Lightning module for training an arbitary model"""

    def __init__(
        self,
        model: str,
        freeze_base: bool = False,
        pretrained: bool = True,
        n_classes: int = 1,
        criterion: str = "mse",
        opt: dict = {"name": "adam"},
        lr: float = 1e-4,
        label_transform=None,
    ):
        """Initialize the module
        Args:
            model (str): name of the ResNet model to use

            freeze_base (bool): whether to freeze the base model

            pretrained (bool): whether to use pretrained weights

            n_classes (int): number of outputs. Set 1 for regression

            criterion (str): loss function to use

            lr (float): learning rate

            label_transform: possible transform that is done for the output labels
        """
        super().__init__()
        self.save_hyperparameters(ignore=["label_transform"])
        self.example_input_array = torch.randn((1, 3, 224, 224))
        self.model = Model(
            model=model,
            freeze_base=freeze_base,
            pretrained=pretrained,
            n_classes=n_classes,
        )
        self.lr = lr
        self.label_transform = label_transform
        self.criterion = choose_criterion(criterion)
        self.opt_args = opt

        if criterion == "cross-entropy":
            self.is_classifier = True
        else:
            self.is_classifier = False

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def predict_func(self, output):
        """Processes the output for prediction"""
        if self.is_classifier:
            return output.argmax(dim=1)
        else:
            return output.flatten()

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def configure_optimizers(self):
        """Sets optimizers based on a dict passed as argument"""
        if self.opt_args["name"] == "adam":
            return torch.optim.Adam(self.model.parameters(), self.lr)
        elif self.opt_args["name"] == "adamw":
            return torch.optim.AdamW(self.model.parameters(), self.lr)
        else:
            raise Exception("Invalid optimizer")

    def common_step(self, batch, batch_idx):
        """
        Perform a common processing step during training, validation, or testing.
        """
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        return x, y, out, loss

    def common_epoch_end(self, outputs, name: str):
        """Combine outputs for calculating metrics at the end of an epoch."""
        y_true = torch.cat([x["y_true"] for x in outputs]).cpu().detach().numpy()
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().detach().numpy()

        if self.label_transform:
            y_true = self.label_transform(y_true)
            y_pred = self.label_transform(y_pred)

        if self.is_classifier:
            self.log(f"{name}/acc", accuracy_score(y_true, y_pred))
            self.log(
                f"{name}/f1",
                f1_score(y_true, y_pred, average="weighted", zero_division=0),
            )

        return y_true, y_pred

    # Training
    def training_step(self, batch, batch_idx):
        """
        Execute a training step and log relevant information.
        """
        _, y, out, loss = self.common_step(batch, batch_idx)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        outputs = {"loss": loss, "y_true": y, "y_pred": self.predict_func(out)}
        self.training_step_outputs.append(outputs)
        return loss

    def on_train_epoch_end(self):
        """
        Perform actions at the end of each training epoch.

        - Calls common_epoch_end method for additional processing.
        - Clears the list of training step outputs.
        """
        outputs = self.training_step_outputs
        _, _ = self.common_epoch_end(outputs, "train")
        self.training_step_outputs.clear()

    # Validation
    def validation_step(self, batch, batch_idx):
        """"
        Execute a validation step and log relevant information.
        """
        _, y, out, val_loss = self.common_step(batch, batch_idx)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        outputs = {"y_true": y, "y_pred": self.predict_func(out)}
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of each validation epoch.

        - Calls common_epoch_end method for additional processing.
        - Clears the list of validation step outputs.
        """
        outputs = self.validation_step_outputs
        _, _ = self.common_epoch_end(outputs, "val")
        self.validation_step_outputs.clear()

    # Testing
    def test_step(self, batch, batch_idx):
        """Execute a testing step and log relevant information."""
        _, y, out, test_loss = self.common_step(batch, batch_idx)
        self.log("test/loss", test_loss, on_step=True, on_epoch=True)
        outputs = {"y_true": y, "y_pred": self.predict_func(out), "out": out}
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        """
        Perform actions at the end of each testing epoch.

        - Extracts outputs from testing steps.
        - If the model is a classifier, computes softmax probabilities and stores them.
        - Calls common_epoch_end method for additional processing.
        """
        outputs = self.test_step_outputs
        if self.is_classifier:
            self.softmax = (
                torch.cat([x["out"] for x in outputs])
                .softmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
        self.y_true, self.y_pred = self.common_epoch_end(outputs, "test")


