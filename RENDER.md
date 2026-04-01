# Render Deployment

This project can be deployed on Render as a Python web service, including on Render's free web service tier.

## What changed

- `requirements.txt` now contains only runtime dependencies for the FastAPI app.
- `requirements-dev.txt` contains the full local training and notebook stack.
- `scripts/render_start.py` starts the app and can download the model from `MODEL_URL` if `MODEL_PATH` does not exist yet.
- `render.yaml` defines a free-tier-friendly Render web service.

## Recommended setup

1. Create a new Render web service from this repo, or deploy the included `render.yaml` Blueprint.
2. Use the default start command from `render.yaml`:

```text
python scripts/render_start.py
```

3. Keep `MODEL_PATH=/tmp/model.pth`.
4. Set `MODEL_URL` to a direct download URL for your `model.pth`.

## Environment variables

- `MODEL_PATH`: filesystem path to the trained model. On free Render, use a temporary path such as `/tmp/model.pth`.
- `MODEL_URL`: direct download URL used when the model is missing at `MODEL_PATH`.
- `PYTHON_VERSION`: pinned to `3.11.11` in `render.yaml`.

## Notes

- Render's free web services are available, but they have important limits: they can spin down when idle, local filesystem changes are lost on redeploy/restart, and free services do not support persistent disks.
- Because of that, this setup downloads the model from `MODEL_URL` into `/tmp/model.pth` when the service starts and the file is missing.
- Expect slower cold starts on the free tier because the service may need to spin up and download the model again.
- The app exposes `/`, `/health`, and `/api/predict`.
