import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from training import WhatdogResNet18


@pytest.fixture
def model():
    """Provide a fresh, offline model with a small classification head."""
    return WhatdogResNet18(num_classes=10, learning_rate=1e-3, weights=None)


@pytest.fixture
def dummy_batch():
    """Provide a batch of images and deterministic class labels."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    return images, labels


@pytest.fixture
def fixed_logits():
    """Provide deterministic logits whose predictions are easy to verify."""
    return torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )


def test_model_initialization_and_freezing(model):
    """Verify that only the replacement classification head is trainable."""
    assert isinstance(model.model.fc, nn.Linear)
    assert model.model.fc.out_features == 10

    backbone_parameters = [
        parameter
        for name, parameter in model.model.named_parameters()
        if not name.startswith("fc.")
    ]
    assert backbone_parameters, "Expected the model to have backbone parameters."
    assert all(
        not parameter.requires_grad for parameter in backbone_parameters
    ), "All backbone parameters should be frozen."

    fc_parameters = list(model.model.fc.parameters())
    assert fc_parameters, "Expected the classification head to have parameters."
    assert all(
        parameter.requires_grad for parameter in fc_parameters
    ), "All FC layer parameters must require gradients."


def test_forward_pass_shape(model):
    """Verify the output shape for a standard image batch."""
    images = torch.randn(2, 3, 224, 224)

    model.eval()
    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (2, 10)


def test_training_step_calculates_loss_and_logs_metrics(
    model, dummy_batch, fixed_logits, mocker
):
    """Verify training loss, accuracy state, and logging configuration."""
    mocker.patch.object(model, "forward", return_value=fixed_logits)
    log_mock = mocker.patch.object(model, "log")

    loss = model.training_step(dummy_batch, batch_idx=0)

    expected_loss = F.cross_entropy(fixed_logits, dummy_batch[1])
    torch.testing.assert_close(loss, expected_loss)
    assert loss.requires_grad
    assert loss.ndim == 0
    assert model.train_acc.compute().item() == pytest.approx(0.75)

    log_calls = {call.args[0]: call for call in log_mock.call_args_list}
    assert log_calls.keys() == {"train_loss", "train_acc"}
    assert log_calls["train_loss"].args[1] is loss
    assert log_calls["train_loss"].kwargs == {
        "on_step": True,
        "on_epoch": True,
        "prog_bar": True,
    }
    assert log_calls["train_acc"].args[1] is model.train_acc
    assert log_calls["train_acc"].kwargs == {
        "on_step": False,
        "on_epoch": True,
        "prog_bar": True,
    }


@pytest.mark.parametrize(
    ("step_name", "loss_name", "accuracy_name", "metric_attribute"),
    [
        ("validation_step", "val_loss", "val_acc", "val_acc"),
        ("test_step", "test_loss", "test_acc", "test_acc"),
    ],
)
def test_evaluation_step_calculates_loss_and_logs_metrics(
    model,
    dummy_batch,
    fixed_logits,
    mocker,
    step_name,
    loss_name,
    accuracy_name,
    metric_attribute,
):
    """Verify validation and test loss, accuracy state, and logging."""
    mocker.patch.object(model, "forward", return_value=fixed_logits)
    log_mock = mocker.patch.object(model, "log")

    result = getattr(model, step_name)(dummy_batch, batch_idx=0)

    metric = getattr(model, metric_attribute)
    assert result is None
    assert metric.compute().item() == pytest.approx(0.75)

    log_calls = {call.args[0]: call for call in log_mock.call_args_list}
    assert log_calls.keys() == {loss_name, accuracy_name}
    loss_call = log_calls[loss_name]
    torch.testing.assert_close(
        loss_call.args[1], F.cross_entropy(fixed_logits, dummy_batch[1])
    )
    assert loss_call.kwargs == {"prog_bar": True}
    assert log_calls[accuracy_name].args[1] is metric
    assert log_calls[accuracy_name].kwargs == {"prog_bar": True}


def test_optimizer_configuration(model):
    """Verify that Adam updates exactly the classification-head parameters."""
    optimizer = model.configure_optimizers()

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    fc_parameter_ids = {id(parameter) for parameter in model.model.fc.parameters()}

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer_parameter_ids == fc_parameter_ids
    assert all(
        group["lr"] == pytest.approx(1e-3) for group in optimizer.param_groups
    )


def test_fast_dev_run():
    """Run one training, validation, and test batch through Lightning."""
    model = WhatdogResNet18(num_classes=10, weights=None)
    dataset = TensorDataset(
        torch.randn(4, 3, 32, 32),
        torch.randint(0, 10, (4,)),
    )
    loader = DataLoader(dataset, batch_size=2)
    trainer = Trainer(
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    test_results = trainer.test(model, dataloaders=loader)

    assert len(test_results) == 1
    assert {"test_loss", "test_acc"} <= test_results[0].keys()
