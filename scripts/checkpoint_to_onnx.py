import argparse
from pathlib import Path

import torch

from app.training.model import WhatdogResNet18


def save_checkpoint_to_onnx(checkpoint_path: Path, onnx_save_path: Path):
    model = WhatdogResNet18.load_from_checkpoint(
        checkpoint_path, weights=None, map_location="cpu"
    )
    model.eval()
    onnx_save_path.parent.mkdir(parents=True, exist_ok=True)
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
        dynamic_shapes={},
    )


def parse_args():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Whatdog Model Checkpoint to ONNX")

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--onnx-save-path",
        type=Path,
        default=PROJECT_ROOT / "onnx_models" / "model.onnx",
    )

    return parser.parse_args()


def main():
    print("===== SAVE MODEL CHECKPOINT TO ONNX =====")
    args = parse_args()

    save_checkpoint_to_onnx(args.checkpoint_path, args.onnx_save_path)
    print(f"Checkpoint successfully saved as ONNX file")


if __name__ == "__main__":
    main()
