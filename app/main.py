from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.v1.routes import api_router
from ml.model_inference import get_hardcoded_dataset_labels
import onnxruntime as ort
import os
from pathlib import Path

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX Model...")
    # Use relative path from the app directory
    onnx_model_file_path = Path(__file__).parent / "ml" / "res18_SGD_model.onnx"
    if not onnx_model_file_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_file_path.absolute()}")
    app.state.onnx_session = ort.InferenceSession(str(onnx_model_file_path))
    print("Loading Index to Class Mapping...")
    app.state.idx_to_class = get_hardcoded_dataset_labels()

    yield

    print("Cleaning Up ONNX Model...")
    del app.state.onnx_session
    print("Cleaning Up Index to Class Mapping...")
    del app.state.idx_to_class

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")
