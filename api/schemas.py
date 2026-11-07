from typing import Optional
from pydantic import BaseModel, Field


class GameInput(BaseModel):
    Rank: Optional[int] = Field(default=None, ge=1)
    Name: Optional[str] = Field(default="Unknown")
    Platform: str = Field(min_length=1)
    Year: float = Field(ge=1980, le=2050)
    Genre: str = Field(min_length=1)
    Publisher: Optional[str] = Field(default="Unknown")


class PredictionResponse(BaseModel):
    predicted_global_sales: float
    unit: str = "millions of units"


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    games: list[GameInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
