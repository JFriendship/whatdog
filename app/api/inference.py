from pathlib import Path
from PIL import Image

class PyTorchInference():
    def __init__(self, checkpoint_path: Path, class_names_path: Path):
        # Get class names


        # Load model and set to eval


        # Ensure model uses same number of classes as stored class names


        # store evaluation transforms

        
        pass

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        # Convert input image into input tensor and prep for inference

        
        # run image through the model and get top_k predictions


        # return results (match pydantic schemas)

        pass
