# Predicting Hippocampal Volume from 3D Brain MRI

## Introduction

This project develops a deep learning pipeline for hippocampus segmentation and volume estimation from 3D brain MRI scans. Because hippocampal atrophy is strongly associated with Alzheimer's disease progression, automated hippocampal measurement can support clinicians in diagnosis and longitudinal monitoring.

The system uses the Hippocampus dataset from the [Medical Decathlon competition](http://medicaldecathlon.com), where each MRI volume is paired with a segmentation mask. A U-Net-based deep learning model is trained to segment hippocampal structures from MRI images and estimate anterior, posterior, and total hippocampal volumes.

The project also includes an end-to-end inference and deployment workflow. After training, the model can process new MRI studies, generate hippocampal predictions, and serve results through a FastAPI backend and browser-based upload UI.

This system is intended as a clinical decision-support prototype rather than a standalone Alzheimer's diagnostic tool. Its primary purpose is to assist radiologists and clinicians by automating hippocampal segmentation and volume quantification from MRI scans.

## The Dataset

I will be using the "Hippocampus" dataset from the [Medical Decathlon competition](http://medicaldecathlon.com). This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images here are T2 MRI scans of the full brain. 

## Key files

### June 2020

- [Exploratory Data Analysis of Hippocampus 3D brain MRI images](https://github.com/pranath/predict_alzheimers/blob/master/eda.ipynb)
- [Building & Training Model for Hippocampus volume prediction](https://github.com/pranath/predict_alzheimers/blob/master/model/experiments/UNetExperiment.py)
- [Using model for inference](https://github.com/pranath/predict_alzheimers/blob/master/model/inference/predict_nii_sample.py)
- `app/main.py`: FastAPI backend and upload UI

## Results

The trained model produces hippocampal segmentation masks and derived anterior, posterior, and total hippocampal volume measurements from MRI studies. The project also includes a FastAPI backend and browser UI for uploading `.nii` or `.nii.gz` brain MRI volumes and reviewing prediction overlays and volume measurements in a simple web workflow.

## Run End-to-End (Windows/Linux)

### 1) Create and activate virtual environment (Python 3.8+)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the full local training and notebook stack. `requirements.txt` is the slimmer runtime dependency set used for hosted deployment.

### 3) Dataset download and local structure

Download Medical Decathlon Task04 Hippocampus:

- Main site: http://medicaldecathlon.com/
- Mirror page: https://github.com/MIC-DKFZ/MedicalDecathlon

After extraction, place files in this structure:

```text
predict_alzheimers/
	data/
		TrainingSet/
			images/
				hippocampus_001.nii.gz
				...
			labels/
				hippocampus_001.nii.gz
				...
```

### 4) Folder structure and execution order

- `eda.ipynb`: EDA and preprocessing preparation (outlier review and clean dataset copy).
- `app/main.py`: FastAPI backend and browser UI.
- `app/services/inference_service.py`: shared prediction service used by the API and UI.
- `model/run_ml_pipeline.py`: training + validation + test metrics.
- `model/inference/predict_nii_sample.py`: single-sample `.nii` prediction and slice visualization.
- `run_end_to_end.py`: one-command preprocessing + training + optional sample inference.

Recommended order:

1. Run `eda.ipynb` (optional but recommended to inspect and understand data).
2. Preprocess/clean dataset (either from notebook or with `run_end_to_end.py`, which does it automatically).
3. Train with `model/run_ml_pipeline.py` (or via `run_end_to_end.py`).
4. Run inference on one sample with `model/inference/predict_nii_sample.py`.

### 5) Run commands

Train directly:

```bash
python model/run_ml_pipeline.py --data-dir data/TrainingSet --output-dir out/predictions --epochs 8
```

Test one sample prediction (`.nii`):

```bash
python model/inference/predict_nii_sample.py \
	--image data/TrainingSet/images/hippocampus_001.nii.gz \
	--label data/TrainingSet/labels/hippocampus_001.nii.gz \
	--model out/predictions/<timestamp>_Basic_unet/model.pth
```

### 6) Single command workflow

Use this from repo root:

```bash
python run_end_to_end.py --data-dir data/TrainingSet --sample-image data/TrainingSet/images/hippocampus_001.nii.gz --sample-label data/TrainingSet/labels/hippocampus_001.nii.gz
```

This command:

1. Creates a cleaned dataset copy in `out/TrainingSet` by copying data and excluding known outliers.
2. Trains/evaluates the model.
3. Runs sample prediction and saves visualization to `out/sample_prediction.png`.

### 7) Web App Deployment

Prepare the final model package:

```bash
python scripts/prepare_final_deployment.py
```

Run single-volume NIfTI inference:

```bash
python model/inference/predict_nii_sample.py --image data/TrainingSet/images/hippocampus_001.nii.gz --label data/TrainingSet/labels/hippocampus_001.nii.gz --model out/final_model/model.pth --save-figure out/final_prediction.png
```

Start the FastAPI backend and UI:

```bash
uvicorn app.main:app --reload
```

Then open:

```bash
http://127.0.0.1:8000
```

Available routes:

- `/`: upload UI
- `/health`: backend health check
- `/docs`: FastAPI Swagger UI
- `/api/predict`: API endpoint for file upload inference

The web app expects a trained model at `out/final_model/model.pth`.

### 7.1) Render deployment

For Render, use the lighter runtime dependencies in `requirements.txt` and the startup helper in `scripts/render_start.py`.

- Blueprint/config file: [render.yaml](./render.yaml)
- Render deployment notes: [RENDER.md](./RENDER.md)

Recommended environment variables on Render:

- `MODEL_PATH=/tmp/model.pth`
- `MODEL_URL=<direct download url for model.pth>`

The Render startup command is:

```bash
python scripts/render_start.py
```

On Render's free tier, the service has an ephemeral filesystem and no persistent disk support, so the model may be downloaded again on cold starts or redeploys.

### 8) GPU setup

This repo's training code uses PyTorch; it automatically uses CUDA if available.

- Verify PyTorch CUDA in Python:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

TensorFlow GPU check/config (optional, as requested):

```bash
python scripts/check_tf_gpu.py
```

### 9) VS Code + Jupyter smooth run

1. Install VS Code extensions: Python and Jupyter.
2. Select interpreter: Command Palette -> `Python: Select Interpreter` -> choose `.venv`.
3. Open `eda.ipynb` and select the same kernel (`.venv`).
4. Run notebook cells top-to-bottom.
5. Run training/inference from VS Code terminal using the commands above.

### 10) Common errors and fixes

- `FileNotFoundError` for `images/` or `labels/`: verify dataset folder is `data/TrainingSet/images` and `data/TrainingSet/labels`, or pass `--data-dir`.
- `No module named ...`: activate `.venv` and reinstall requirements.
- CUDA/GPU not found: install compatible CUDA drivers/toolkit, or run on CPU.
- TensorFlow GPU not found: verify CUDA/cuDNN compatibility for installed TensorFlow version.
