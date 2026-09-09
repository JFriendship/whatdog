import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path

from app.api.inference import PyTorchInference

@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = Path(os.environ["CHECKPOINT_PATH"])
    class_names_path = Path(os.environ["CLASS_NAMES_PATH"])

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    
    if not class_names_path.is_file():
        raise FileNotFoundError(f"class names file not found: {class_names_path}")
    
    print("LOADING WHATDOG MODEL")
    app.state.inference = PyTorchInference(checkpoint_path=checkpoint_path, class_names_path=class_names_path)
    print("FINISHED LOADING WHATDOG MODEL")

    yield

    print("TEARING DOWN WHATDOG MODEL")
    del app.state.inference
    print("WHATDOG MODEL SUCCESSFULLY TORN DOWN")

app = FastAPI(title="Whatdog API", version="2.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}