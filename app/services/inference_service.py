"""
Shared inference helpers for the FastAPI backend and UI.
"""

from __future__ import annotations

import base64
import io
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from inference.UNetInferenceAgent import UNetInferenceAgent

DEFAULT_MODEL_PATH = REPO_ROOT / "out" / "final_model" / "model.pth"
DEFAULT_PATCH_SIZE = 64


def _normalise_volume(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    max_val = np.max(arr)
    if max_val > 0:
        arr = arr / max_val
    return arr


def load_nifti(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    arr = nii.get_fdata()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {arr.shape}")
    return _normalise_volume(arr)


@lru_cache(maxsize=2)
def get_inference_agent(model_path: str, patch_size: int) -> UNetInferenceAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return UNetInferenceAgent(
        parameter_file_path=model_path,
        device=device,
        patch_size=patch_size,
    )


def compute_volume_summary(prediction: np.ndarray) -> dict[str, int]:
    anterior = int(np.sum(prediction == 1))
    posterior = int(np.sum(prediction == 2))
    total = anterior + posterior
    return {
        "anterior": anterior,
        "posterior": posterior,
        "total": total,
    }


def build_alzheimers_assessment(volumes: dict[str, int]) -> dict[str, str]:
    total = volumes["total"]
    return {
        "verdict": "Cannot determine",
        "status": "not_supported",
        "badge": "Not diagnostic",
        "headline": "Alzheimer's status cannot be predicted by this model.",
        "detail": (
            "This prototype segments the hippocampus and estimates volume only. "
            "A true Alzheimer's yes/no prediction would require a separately trained "
            "and clinically validated classification model with patient-level labels."
        ),
        "volume_context": (
            f"Measured hippocampal voxels in this scan: {total}. "
            "Use this as a structural measurement, not a diagnosis."
        ),
        "next_step": (
            "Treat the segmentation and volume report as supportive information for a clinician, "
            "not a standalone medical decision."
        ),
    }


def render_preview(
    image_vol: np.ndarray,
    pred_vol: np.ndarray,
    slice_index: int | None = None,
) -> tuple[str, int]:
    if slice_index is None:
        slice_index = int(image_vol.shape[0] // 2)

    if not (0 <= slice_index < image_vol.shape[0]):
        raise ValueError(f"slice_index must be in [0, {image_vol.shape[0] - 1}], got {slice_index}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_vol[slice_index, :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("MRI Slice")

    axes[1].imshow(pred_vol[slice_index, :, :].T, cmap="viridis", origin="lower", vmin=0, vmax=2)
    axes[1].set_title("Predicted Mask")

    axes[2].imshow(image_vol[slice_index, :, :].T, cmap="gray", origin="lower")
    axes[2].imshow(pred_vol[slice_index, :, :].T, cmap="autumn", origin="lower", alpha=0.35, vmin=0, vmax=2)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded, slice_index


def predict_from_file_bytes(
    file_bytes: bytes,
    filename: str,
    slice_index: int | None = None,
    model_path: Path | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> dict[str, object]:
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else (Path(filename).suffix or ".nii")
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = Path(tmp.name)

        image = load_nifti(temp_path)
        agent = get_inference_agent(str(model_path), patch_size)
        prediction = agent.single_volume_inference_unpadded(image)
        preview_base64, resolved_slice_index = render_preview(image, prediction, slice_index=slice_index)
        volumes = compute_volume_summary(prediction)

        return {
            "filename": filename,
            "shape": [int(x) for x in prediction.shape],
            "labels": [int(x) for x in np.unique(prediction).tolist()],
            "slice_index": int(resolved_slice_index),
            "volumes": volumes,
            "alzheimers_assessment": build_alzheimers_assessment(volumes),
            "preview_png_base64": preview_base64,
            "prototype_note": (
                "This output is a hippocampus segmentation and volume-estimation prototype "
                "for decision support. It is not a standalone Alzheimer's diagnosis."
            ),
        }
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
