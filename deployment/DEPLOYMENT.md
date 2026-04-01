# Deployment Guide

## What this deployment does

This project deploys a hippocampus segmentation and volume-estimation model.
It does not make a standalone Alzheimer's diagnosis.

The deployment flow is:

1. Read a routed DICOM study or study folder.
2. Select the MRI series whose `SeriesDescription` matches the configured value.
3. Run inference with `out/final_model/model.pth`.
4. Create a DICOM Secondary Capture report with hippocampal volume measurements.
5. Optionally send that report to PACS/Orthanc using `storescu`.

## Required artifacts

- `out/final_model/model.pth`
- A routed DICOM study folder
- `deployment/config/pacs.example.json` as the configuration template
- Optional: `storescu` from DCMTK if you want to send results to PACS

## DICOM / PACS configuration

Copy `deployment/config/pacs.example.json` and edit these values:

- `routing_dir`: folder containing routed studies or one study folder
- `model_path`: final model checkpoint
- `report_path`: output report DICOM path
- `series_description`: series selector, usually `HippoCrop`
- `skip_send`: `true` for local testing, `false` to send to PACS
- `send_host`: PACS/Orthanc host
- `send_port`: PACS/Orthanc DIMSE port
- `send_aet`: called AE title
- `storescu_bin`: `storescu` executable path if not on PATH
- `cleanup_study`: delete processed study only if you truly want that behavior

## Local report generation

Create a report DICOM without sending it:

```powershell
.\.venv\Scripts\python.exe deployment\inference_dcm.py "C:\path\to\routed_dicoms" --model-path out\final_model\model.pth --report-path out\reports\final_report.dcm --skip-send
```

Or use a config file:

```powershell
.\.venv\Scripts\python.exe deployment\inference_dcm.py --config deployment\config\pacs.example.json
```

## PACS / Orthanc send

Send the generated report to a PACS or Orthanc endpoint:

```powershell
.\.venv\Scripts\python.exe deployment\inference_dcm.py "C:\path\to\routed_dicoms" --model-path out\final_model\model.pth --report-path out\reports\final_report.dcm --send-host 10.0.0.25 --send-port 4242 --send-aet HIPPOAI
```

## Common options

- `--series-description HippoCrop`
- `--skip-send`
- `--send-host <hostname>`
- `--send-port <port>`
- `--send-aet <called-ae-title>`
- `--storescu-bin <path-to-storescu>`
- `--cleanup-study`
- `--config <path-to-json-config>`

## For broader deployment

For a real multi-site deployment, package the inference service in a container and place it behind:

- Orthanc + DICOMweb + OHIF for a self-hosted setup
- or a managed cloud DICOM service plus a containerized worker

In both cases, the model should be treated as a clinical decision-support prototype.
