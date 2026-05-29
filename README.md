# Neural Morphing Demo

Minimal deployment branch for the Neural Morphing web demo.

This branch intentionally keeps only the files needed to build and deploy the Gradio reference demo:

- `python_project_idea.py` - demo application and DAC morphing prototype.
- `requirements.txt` - Python runtime dependencies for the demo container.
- `Dockerfile` - container image definition.
- `cloudbuild.yaml` - optional Cloud Build pipeline for Cloud Run.

The full plugin, paper experiments, evaluation tools, and generated artifacts remain on `develop`.

## Run Locally

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8080 python python_project_idea.py
```

Then open `http://localhost:8080`.

## Build The Container

```bash
docker build -t neural-morphing-demo .
docker run --rm -p 8080:8080 neural-morphing-demo
```

## Deploy With Google Cloud Build And Cloud Run

Create the Artifact Registry repository once if it does not already exist:

```bash
gcloud artifacts repositories create cloud-run-source-deploy \
  --repository-format=docker \
  --location=europe-west1
```

Submit a one-off build from this branch:

```bash
gcloud builds submit --project neural-morphing --config cloudbuild.yaml .
```

The checked-in Cloud Build config deploys the `neural-morphing` Cloud Run service with 4 CPU, 8 GiB memory, and a 900 second request timeout. Those resources are required for the DAC model; the default Cloud Run 512 MiB limit is not enough.

For private-repository deployment, create a Cloud Build trigger for this `demo` branch and point it at `cloudbuild.yaml`. The repository can stay private; reviewers only need the Cloud Run URL.
