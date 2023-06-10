from pydantic import BaseModel
from pydantic import Field


class Prediction(BaseModel):
    prediction: dict[str, float] = Field(..., description="Mapping of emotion to probability")
