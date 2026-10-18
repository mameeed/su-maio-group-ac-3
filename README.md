# Diabetes Triage ML Service

MAIO Assignment 3, Group AC.

A small ML service for a virtual diabetes clinic. It predicts a disease progression score from patient features, so nurses can sort patients by risk (higher = worse). It uses the scikit-learn diabetes dataset as stand-in data.

## Run with Docker

```bash
docker pull ghcr.io/mameeed/su-maio-group-ac-3:v0.2
docker run -it -p 80:80 ghcr.io/mameeed/su-maio-group-ac-3:v0.2
```

For the baseline, use `:v0.1` instead. Both use port 80, so stop one before starting the other.

## API

### `GET /health`

```bash
curl http://localhost:80/health
```

```json
{"status": "ok", "model_version": "v0.2"}
```

### `POST /predict`

All 10 features are required and must be numbers.

```bash
curl -X POST http://localhost:80/predict -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

```json
{"prediction": 202.11}
```

### Error response

A missing or non-numeric feature returns HTTP **422** with a JSON body. Each item in `detail` names the field (`loc`) and the problem (`msg`). Example with `s6` missing:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}
    }
  ]
}
```

## Run locally (without Docker)

Needs Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py
ruff check .
python -m pytest -v
uvicorn app:app --port 8000
```

`train.py` saves the model to `model/model.joblib` and the metrics to `model/metrics.json`.

## Reproducibility

- Fixed seed (`random_state=42`) for the train/test split
- All package versions are pinned in `requirements.txt` / `requirements-dev.txt`
- Python version and Docker base image are pinned (`python:3.12.7-slim`)
- The model is trained inside the Docker build, so the image contains everything it needs

Running `python train.py` again gives the same RMSE every time. To see the comparison of the models we tried, run `python compare_models.py`.

## CI/CD (GitHub Actions)

- **CI** (`.github/workflows/ci.yml`) runs on every push to main and every PR: lint (ruff), a training run, unit tests (pytest), and an upload of the model and metrics as an artifact.
- **Release** (`.github/workflows/release.yml`) runs when a `v*` tag is pushed. It builds the Docker image and smoke-tests the container (`/health`, `/predict`, and a 422 on bad input). Then it pushes the image to GHCR (amd64 + arm64) and creates a GitHub Release with `metrics.json` and the changelog.

## Versions

See [CHANGELOG.md](CHANGELOG.md) for what changed between v0.1 and v0.2, with metrics.
