from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.v1.routes import api_router
from data_pipeline import load_imagefolder_dataset, clean_label_mapping
import onnxruntime as ort
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX Model...")
    onnx_model_file_path = "trained_models/res18_SGD_model.onnx"
    app.state.onnx_session = ort.InferenceSession(onnx_model_file_path)
    print("Loading Index to Class Mapping...")
    current_dir = os.getcwd()
    root_dir = current_dir + "/Images"
    dataset = load_imagefolder_dataset(root_dir=root_dir)
    app.state.idx_to_class = clean_label_mapping(dataset=dataset)

    yield

    print("Cleaning Up ONNX Model...")
    del app.state.onnx_session
    print("Cleaning Up Index to Class Mapping...")
    del app.state.idx_to_class

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")
