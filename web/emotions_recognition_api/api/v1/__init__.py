from fastapi import APIRouter

from . import predictions

router = APIRouter(prefix="/v1")

router.include_router(predictions.router)
