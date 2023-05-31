from fastapi import APIRouter

from . import health
from . import v1

router = APIRouter()

router.include_router(health.router)
router.include_router(v1.router)
