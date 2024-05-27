import lightning.pytorch as pl
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score


def mse(output, target):
    output = output.flatten()
    loss = F.mse_loss(output, target)
    return loss


def mape(output, target):
    output = output.flatten()
    loss = torch.mean((torch.abs(output - target) / target.abs()))
    return loss


def l1(output, target):
    output = output.flatten()
    loss = F.l1_loss(output, target)
    return loss


def cross_entropy(output, target):
    loss = F.cross_entropy(output, target.long())
    return loss


def choose_criterion(name):
    if name == "cross-entropy":
        return cross_entropy
    if name == "mse":
        return mse
    if name == "mape":
        return mape
    if name == "l1":
        return l1
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
            timm.create_model(model, pretrained=False, num_classes=1)
            .get_classifier()
            .in_features
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
        lr_scheduler: dict = None,
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
        self.lr_scheduler = lr_scheduler
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
        self.batch_size = None

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
            optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        elif self.opt_args["name"] == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        else:
            raise Exception("Invalid optimizer")

        if self.lr_scheduler is not None:
            if self.lr_scheduler["name"] != "CosineAnnealingLR":
                raise Exception("Invalid learning rate schedule")
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.lr_scheduler["T_max"]
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def common_step(self, batch, batch_idx):
        """
        Perform a common processing step during training, validation, or testing.
        """
        x = batch["x"]
        y = batch["y"]
        fname = batch["fname"]
        out = self.model(x)
        loss = self.criterion(out, y)

        # Set the batch size for logging
        if self.batch_size is None:
            self.batch_size = len(x)
        return x, y, fname, out, loss

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
        _, y, _, out, loss = self.common_step(batch, batch_idx)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
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
        """
        Execute a validation step and log relevant information.
        """
        _, y, _, out, val_loss = self.common_step(batch, batch_idx)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
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
        _, y, fname, out, test_loss = self.common_step(batch, batch_idx)
        self.log("test/loss", test_loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
        outputs = {"y_true": y, "y_pred": self.predict_func(out), "fname": fname, "out": out}
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        """
        Perform actions at the end of each testing epoch.

        - Extracts outputs from testing steps.
        - If the model is a classifier, computes softmax probabilities and stores them.
        - Calls common_epoch_end method for additional processing.
        """
        outputs = self.test_step_outputs
        xss = [x["fname"] for x in outputs]
        self.fnames = np.array([x for xs in xss for x in xs])
        if self.is_classifier:
            logits = torch.cat([x["out"] for x in outputs])
            self.softmax = logits.softmax(dim=1).cpu().detach().numpy()
            self.logits = logits.cpu().detach().numpy()
        self.y_true, self.y_pred = self.common_epoch_end(outputs, "test")


class FeatureExtractionModule(pl.LightningModule):
    def __init__(
        self,
        feature_extraction_mode: str,
        model: str,
        freeze_base: bool = False,
        pretrained: bool = True,
        n_classes: int = 0,
        criterion: str = "cross-entropy",
        opt: dict = {"name": "adam"},
        lr: float = 1e-4,
        label_transform=None,
    ):
        """
        The feature exctraction module implements the same interface as the basic LitModule
        for passing LitModule parameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["label_transform"])
        self.example_input_array = torch.randn((1, 3, 224, 224))

        self.feature_extraction_mode = feature_extraction_mode
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

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if self.feature_extraction_mode == "unpooled":
            return self.model.base_model.forward_features(x)
        elif self.feature_extraction_mode == "pooled":
            return self.model.base_forward(x)
        else:
            raise Exception(
                f"Invalid feature extraction mode {self.feature_extraction_mode}"
            )

    def test_step(self, batch, batch_idx):
        bx, by = batch
        out = self.forward(bx)
        outputs = {
            "y_true": by.cpu().detach().numpy(),
            "out": out.cpu().detach().numpy(),
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.y_true = [x["y_true"] for x in outputs]
        self.y_pred = [x["out"] for x in outputs]
