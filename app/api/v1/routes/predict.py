from fastapi import APIRouter, UploadFile, Request
from PIL import Image
import io
from schemas.prediction import PredictionResponse
from app.ml.data_pipeline import get_transformations
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile, request: Request):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    _, test_transform = get_transformations()
    
    session = request.app.state.onnx_session

    input_name = session.get_inputs()[0].name
    input_data = test_transform(image).unsqueeze(0).detach().cpu().numpy()

    outputs = session.run(None, {input_name: input_data})

    prediction_idx = np.argmax(outputs[0])
    idx_to_class = request.app.state.idx_to_class
    prediction_class = idx_to_class[prediction_idx]

    return {
        "label": prediction_class
    }
