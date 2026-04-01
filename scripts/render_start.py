"""
Render startup helper for the FastAPI app.

If MODEL_PATH is missing and MODEL_URL is provided, downloads the model
before starting Uvicorn.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

import uvicorn


def ensure_model() -> Path:
    model_path = Path(os.environ.get("MODEL_PATH", "out/final_model/model.pth")).expanduser()
    model_url = os.environ.get("MODEL_URL")

    if model_path.exists():
        return model_path

    if not model_url:
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Set MODEL_PATH to an existing file "
            "or provide MODEL_URL so Render can download the model at startup."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(model_url) as response, model_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)

    return model_path


def main() -> None:
    model_path = ensure_model()
    os.environ.setdefault("MODEL_PATH", str(model_path))
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
