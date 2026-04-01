# Project Pipeline

## Required folders

- `data/TrainingSet/images`
- `data/TrainingSet/labels`
- `app/`
- `model/`
- `deployment/`
- `deployment/assets/`
- `scripts/`
- `out/final_model`

## What each part does

- `data/TrainingSet`: the NIfTI dataset used for training and sample inference.
- `app`: FastAPI backend and browser UI for MRI upload and prediction review.
- `model/run_ml_pipeline.py`: trains the UNet model and evaluates it.
- `model/inference/predict_nii_sample.py`: runs inference on one `.nii/.nii.gz` volume and saves a preview image.
- `scripts/prepare_final_deployment.py`: copies the chosen trained checkpoint into `out/final_model`.
- `deployment`: optional legacy DICOM/PACS prototype code.
- `out/final_model/model.pth`: the final trained model used for deployment.

## Training pipeline

1. Load NIfTI images and labels from `data/TrainingSet`.
2. Normalize and reshape each volume to the configured patch size.
3. Split the dataset into train, validation, and test subsets.
4. Train the 2D UNet on slice batches.
5. Save `model.pth` and evaluation metrics.
6. Promote the chosen checkpoint into `out/final_model`.

## Final deployment pipeline

1. Load the final model from `out/final_model/model.pth`.
2. Start the FastAPI app in `app/main.py`.
3. Upload a `.nii` or `.nii.gz` brain MRI through the browser UI or `/api/predict`.
4. Generate the final prediction preview and hippocampal volume summary.

## Main commands

Train:

```powershell
.\.venv\Scripts\python.exe model\run_ml_pipeline.py --data-dir data\TrainingSet --output-dir out\predictions --epochs 8
```

Prepare final model:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_final_deployment.py
```

Run NIfTI demo:

```powershell
.\.venv\Scripts\python.exe model\inference\predict_nii_sample.py --image data\TrainingSet\images\hippocampus_001.nii.gz --label data\TrainingSet\labels\hippocampus_001.nii.gz --model out\final_model\model.pth --save-figure out\final_prediction.png
```

Run FastAPI app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## Final project structure

- `.venv`
- `app`
- `data/TrainingSet`
- `deployment`
- `model`
- `out/final_model`
- `scripts`
- `README.md`
- `PIPELINE.md`
- `requirements.txt`
- `run_end_to_end.py`
