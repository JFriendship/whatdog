import lightning as L
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class WhatdogResNet18(L.LightningModule):

    def __init__(
        self,
        num_classes: int = 120,
        learning_rate: float = 1e-3,
        weights: models.ResNet18_Weights | None = models.ResNet18_Weights.IMAGENET1K_V1,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["weights"])

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=weights)

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Update the final classification layer (updatable parameters)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.hparams.learning_rate)

        return optimizer
