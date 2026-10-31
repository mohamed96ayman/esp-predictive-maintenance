"""FastAPI entry point. Inference never requires the experiment tracker."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from .model import RiskModel

Row = Annotated[list[FiniteFloat], Field(min_length=45, max_length=45)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    preprocessing_id: str
    feature_names: Annotated[list[str], Field(min_length=45, max_length=45)]
    samples: Annotated[list[Row], Field(min_length=504, max_length=504)]


class PredictionResponse(BaseModel):
    failure_risk: Annotated[float, Field(ge=0, le=1)]
    horizon_hours: int
    threshold: float
    alert: bool
    model_version: str


def create_app(model_dir=None):
    directory = Path(model_dir or os.environ.get("ESP_MODEL_DIR", "models"))

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.risk_model = RiskModel(directory)
        except Exception as exc:
            logging.getLogger(__name__).error("Cannot load model bundle: %s", exc)
            raise RuntimeError("Model bundle failed validation; check ESP_MODEL_DIR") from exc
        yield
        del app.state.risk_model

    application = FastAPI(title="SPE ESP Failure Risk", version="1.0.0", lifespan=lifespan,
                          description="168-hour failure-risk score for prepared BASIC windows.")

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_version": request.app.state.risk_model.manifest["model_version"]}

    @application.get("/model-info")
    def model_info(request: Request):
        return request.app.state.risk_model.info()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request):
        try:
            return request.app.state.risk_model.predict(
                body.samples, body.feature_names, body.preprocessing_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


app = create_app()
