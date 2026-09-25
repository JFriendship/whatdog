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

        self.learning_rate = learning_rate

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=weights)

        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Update the final classification layer (updatable parameters)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Training Evaluation Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Validation Evaluation Metrics
        self.val_acc_top_1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc_top_3 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.val_acc_top_5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Test Evaluation Metrics
        self.test_acc_top_1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc_top_3 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.test_acc_top_5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

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

        # preds = torch.argmax(logits, dim=1)
        self.val_acc_top_1(logits, y)
        self.val_acc_top_3(logits, y)
        self.val_acc_top_5(logits, y)
        self.val_recall(logits, y)
        self.val_f1(logits, y)

        # Shown in command line
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_top_1", self.val_acc_top_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_macro_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Only shown in metrics.csv
        self.log("val_acc_top_3", self.val_acc_top_3, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_acc_top_5", self.val_acc_top_5, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_macro_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=False)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # preds = torch.argmax(logits, dim=1)
        self.test_acc_top_1(logits, y)
        self.test_acc_top_3(logits, y)
        self.test_acc_top_5(logits, y)
        self.test_recall(logits, y)
        self.test_f1(logits, y)

        # Shown in command line
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc_top_1", self.test_acc_top_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_macro_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Only shown in metrics.csv
        self.log("test_acc_top_3", self.test_acc_top_3, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_acc_top_5", self.test_acc_top_5, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_macro_recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)

        return optimizer
