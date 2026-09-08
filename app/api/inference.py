import json
from pathlib import Path
from PIL import Image
from torchvision import transforms

import torch
from app.training.model import WhatdogResNet18

class PyTorchInference():
    def __init__(self, checkpoint_path: Path, class_names_path: Path) -> None:
        self.class_names = json.loads(class_names_path.read_text(encoding="utf-8"))

        self.model = WhatdogResNet18.load_from_checkpoint(checkpoint_path, weights=None, map_location="cpu")
        self.model.eval()

        # Ensure model uses same number of classes as stored class names
        if self.model.hparams.num_classes != len(self.class_names):
            raise ValueError(
                f"The loaded model's num_classes ({self.model.hparams.num_classes}) does not match the checkpoint at {class_names_path} ({len(self.class_names)})"
            )

        self.evaluation_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(224),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        # Convert input image into input tensor and prep for inference

        # run image through the model and get top_k predictions

        # return results (match pydantic schemas)

        pass
