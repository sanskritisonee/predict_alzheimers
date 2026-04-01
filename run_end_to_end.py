"""
One-command workflow for preprocessing, training, and optional sample inference.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


OUTLIERS = {"hippocampus_010.nii.gz", "hippocampus_281.nii.gz", "hippocampus_118.nii.gz"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the project end-to-end")
    parser.add_argument("--data-dir", type=str, default="data/TrainingSet",
                        help="Input dataset root containing images/ and labels/")
    parser.add_argument("--clean-dir", type=str, default="out/TrainingSet",
                        help="Output directory for cleaned dataset")
    parser.add_argument("--output-dir", type=str, default="out/predictions",
                        help="Training output directory")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--sample-image", type=str, default=None,
                        help="Optional .nii/.nii.gz image path for post-training inference test")
    parser.add_argument("--sample-label", type=str, default=None,
                        help="Optional .nii/.nii.gz label path for visualization")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    return parser.parse_args()


def clean_dataset(src_root: Path, dst_root: Path):
    src_images = src_root / "images"
    src_labels = src_root / "labels"

    if not src_images.is_dir() or not src_labels.is_dir():
        raise FileNotFoundError(
            "Dataset missing required folders:\n"
            f"  - {src_images}\n"
            f"  - {src_labels}"
        )

    dst_images = dst_root / "images"
    dst_labels = dst_root / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for f in src_labels.iterdir():
        if f.is_file() and f.name not in OUTLIERS:
            shutil.copy2(f, dst_labels / f.name)

    for f in src_images.iterdir():
        if f.is_file() and f.name not in OUTLIERS:
            shutil.copy2(f, dst_images / f.name)
            copied += 1

    print(f"Cleaned dataset is ready at: {dst_root.resolve()} ({copied} image files copied)")


def run_cmd(cmd):
    print("\n> " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def find_latest_model(output_dir: Path):
    model_files = sorted(output_dir.rglob("model.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return model_files[0] if model_files else None


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    data_dir = (repo_root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    clean_dir = (repo_root / args.clean_dir).resolve() if not Path(args.clean_dir).is_absolute() else Path(args.clean_dir)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    clean_dataset(data_dir, clean_dir)

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            str(repo_root / "model" / "run_ml_pipeline.py"),
            "--data-dir", str(clean_dir),
            "--output-dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patch-size", str(args.patch_size),
        ]
        run_cmd(train_cmd)

    if args.sample_image:
        model_path = find_latest_model(output_dir)
        if model_path is None:
            raise FileNotFoundError(
                f"No model.pth found in {output_dir}. Run training first or pass --skip-train only when model exists."
            )

        infer_cmd = [
            sys.executable,
            str(repo_root / "model" / "inference" / "predict_nii_sample.py"),
            "--image", str(Path(args.sample_image).resolve()),
            "--model", str(model_path),
            "--patch-size", str(args.patch_size),
            "--save-figure", str(repo_root / "out" / "sample_prediction.png"),
        ]

        if args.sample_label:
            infer_cmd.extend(["--label", str(Path(args.sample_label).resolve())])

        run_cmd(infer_cmd)

    print("\nWorkflow completed.")


if __name__ == "__main__":
    main()
