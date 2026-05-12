from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str