"""
Utility script to run UNet inference on a single NIfTI (.nii/.nii.gz) MRI volume,
visualize slices, and optionally compare with a label file.
"""

import argparse
from pathlib import Path
import sys

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from inference.UNetInferenceAgent import UNetInferenceAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Run sample NIfTI inference with trained UNet")
    parser.add_argument("--image", required=True, type=str, help="Path to input MRI .nii/.nii.gz")
    parser.add_argument("--model", required=True, type=str, help="Path to trained model.pth")
    parser.add_argument("--label", type=str, default=None, help="Optional ground-truth label .nii/.nii.gz")
    parser.add_argument("--slice-index", type=int, default=None, help="Axial slice index. Defaults to middle slice.")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size used during training")
    parser.add_argument("--save-figure", type=str, default=None, help="Optional output PNG path")
    return parser.parse_args()


def load_nifti(path):
    nii = nib.load(str(path))
    arr = nii.get_fdata()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    max_val = np.max(arr)
    if max_val > 0:
        arr = arr / max_val
    return arr


def show_slices(image_vol, pred_vol, label_vol=None, slice_index=None, save_path=None):
    if slice_index is None:
        slice_index = image_vol.shape[0] // 2

    if not (0 <= slice_index < image_vol.shape[0]):
        raise ValueError(f"slice-index must be in [0, {image_vol.shape[0] - 1}], got {slice_index}")

    ncols = 3 if label_vol is None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(image_vol[slice_index, :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("MRI slice")

    axes[1].imshow(pred_vol[slice_index, :, :].T, cmap="viridis", origin="lower", vmin=0, vmax=2)
    axes[1].set_title("Predicted mask")

    overlay = image_vol[slice_index, :, :].T
    axes[2].imshow(overlay, cmap="gray", origin="lower")
    axes[2].imshow(pred_vol[slice_index, :, :].T, cmap="autumn", origin="lower", alpha=0.35, vmin=0, vmax=2)
    axes[2].set_title("Overlay")

    if label_vol is not None:
        axes[3].imshow(label_vol[slice_index, :, :].T, cmap="viridis", origin="lower", vmin=0, vmax=2)
        axes[3].set_title("Ground truth")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved visualization to: {out}")

    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    label_path = Path(args.label) if args.label else None

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if label_path and not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image = load_nifti(image_path)
    label = load_nifti(label_path) if label_path else None

    inference_agent = UNetInferenceAgent(
        parameter_file_path=str(model_path),
        device=device,
        patch_size=args.patch_size,
    )

    prediction = inference_agent.single_volume_inference_unpadded(image)

    unique_vals = np.unique(prediction)
    print(f"Prediction done. Shape: {prediction.shape}, labels: {unique_vals}")

    show_slices(
        image_vol=image,
        pred_vol=prediction,
        label_vol=label,
        slice_index=args.slice_index,
        save_path=args.save_figure,
    )


if __name__ == "__main__":
    main()
