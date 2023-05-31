from io import BytesIO

from core import schemas
from core.predictor import predictor
from fastapi import APIRouter
from fastapi import UploadFile

router = APIRouter(prefix="/predictions")


@router.post("/")
def predict_emotion(audio: UploadFile) -> schemas.Prediction:
    prediction = predictor.predict(audio.file)
    return schemas.Prediction(prediction=prediction)
