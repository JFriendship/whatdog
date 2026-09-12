import io
from fastapi import APIRouter, Query, Request, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.api.schemas import PredictionResponse
from PIL import Image, UnidentifiedImageError

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile, top_k: int = Query(default=3, ge=1, le=10)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or a PNG",
        )
    
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
    
    predictions = await run_in_threadpool(
        request.app.state.inference.predict,
        image,
        top_k,
    )

    return {"predictions": predictions}