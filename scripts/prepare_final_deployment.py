"""
Prepare the final deployment model directory from an existing trained checkpoint.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Copy an existing trained model into out/final_model")
    parser.add_argument(
        "--source-model",
        type=str,
        default=None,
        help="Optional path to a specific model.pth. Defaults to the latest model under out/predictions.",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="out/final_model",
        help="Directory where final deployment artifacts will be stored.",
    )
    return parser.parse_args()


def find_latest_model(predictions_dir: Path) -> Path | None:
    model_files = sorted(predictions_dir.rglob("model.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return model_files[0] if model_files else None


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    source_model = Path(args.source_model).resolve() if args.source_model else None
    if source_model is None:
        source_model = find_latest_model(repo_root / "out" / "predictions")

    if source_model is None or not source_model.exists():
        raise FileNotFoundError("No trained model found. Pass --source-model or create a model.pth first.")

    source_run_dir = source_model.parent
    source_results = source_run_dir / "results.json"

    dest_dir = (repo_root / args.dest_dir).resolve() if not Path(args.dest_dir).is_absolute() else Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_model = dest_dir / "model.pth"
    shutil.copy2(source_model, dest_model)

    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": str(source_model),
        "destination_model": str(dest_model),
    }

    if source_results.exists():
        dest_results = dest_dir / "results.json"
        shutil.copy2(source_results, dest_results)
        manifest["source_results"] = str(source_results)
        manifest["destination_results"] = str(dest_results)

    manifest_path = dest_dir / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Prepared final deployment model at: {dest_model}")
    if source_results.exists():
        print(f"Copied metrics to: {dest_dir / 'results.json'}")
    print(f"Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
