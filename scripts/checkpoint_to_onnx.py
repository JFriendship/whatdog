from pathlib import Path

import torch

from app.training.model import WhatdogResNet18


def save_checkpoint_to_onnx(checkpoint_path: Path, onnx_save_path: Path):
    model = WhatdogResNet18.load_from_checkpoint(
        checkpoint_path, weights=None, map_location="cpu"
    )
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_save_path,
        export_params=True,
        opset_version=25,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["predictions"],
        dynamic_shapes={"images": {0: "batch_size"}, "predictions": {0: "batch_size"}},
    )
