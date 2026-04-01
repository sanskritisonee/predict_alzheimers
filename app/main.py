"""
FastAPI backend and simple UI for MRI upload and hippocampus volume prediction.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.inference_service import DEFAULT_MODEL_PATH, predict_from_file_bytes

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="HippoVolume AI",
    description="FastAPI backend and UI for hippocampus segmentation and volume estimation from brain MRI.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _validate_upload(file: UploadFile) -> None:
    name = (file.filename or "").lower()
    if not name.endswith(".nii") and not name.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Upload must be a .nii or .nii.gz MRI volume.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "model_ready": DEFAULT_MODEL_PATH.exists(),
            "model_path": str(DEFAULT_MODEL_PATH),
            "result": None,
            "error": None,
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_ready": DEFAULT_MODEL_PATH.exists(),
        "model_path": str(DEFAULT_MODEL_PATH),
    }


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    slice_index: int | None = Form(default=None),
) -> dict[str, object]:
    _validate_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return predict_from_file_bytes(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_scan.nii.gz",
            slice_index=slice_index,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    file: UploadFile = File(...),
    slice_index: int | None = Form(default=None),
) -> HTMLResponse:
    _validate_upload(file)
    file_bytes = await file.read()

    if not file_bytes:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "model_ready": DEFAULT_MODEL_PATH.exists(),
                "model_path": str(DEFAULT_MODEL_PATH),
                "result": None,
                "error": "The uploaded file was empty.",
            },
            status_code=400,
        )

    try:
        result = predict_from_file_bytes(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_scan.nii.gz",
            slice_index=slice_index,
        )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "model_ready": DEFAULT_MODEL_PATH.exists(),
                "model_path": str(DEFAULT_MODEL_PATH),
                "result": result,
                "error": None,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "model_ready": DEFAULT_MODEL_PATH.exists(),
                "model_path": str(DEFAULT_MODEL_PATH),
                "result": None,
                "error": str(exc),
            },
            status_code=500,
        )
