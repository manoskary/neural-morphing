# Neural Morphing Demo

Hybrid demo branch for local plugin testing and hosted Gradio review.

This branch contains:

- `Source/`, `Resources/`, `JUCE/`, `CMakeLists.txt` - VST3 and Standalone build tree.
- `bridge/` - optional Python bridge runtime used by the plugin.
- `python_project_idea.py`, `assets/`, `examples/` - Gradio demo app with curated examples.
- `Dockerfile`, `cloudbuild.yaml` - Cloud Run deployment files.

Paper, evaluation, result, table, and figure artifacts are intentionally not part of this branch.

## Run The Gradio Demo

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8080 python python_project_idea.py
```

Then open `http://localhost:8080`.

The default demo settings are:

- Threshold: `0.99`
- Playback Dry/Wet: `0.7`
- Match Mode: `beam`
- Swap Mode: `full_layer_gated`

## Build The Local Plugin

Configure and build the Python-bridge plugin:

```powershell
cmake -S . -B build-bridge -G "Visual Studio 17 2022" -A x64 -DNEURAL_MORPHING_ENABLE_ONNX=OFF -DNM_WITH_PYBRIDGE=ON
cmake --build build-bridge --config Release --target NeuralMorphing_All
```

Build outputs are written under:

```text
build-bridge/NeuralMorphing_artefacts/Release/VST3/
build-bridge/NeuralMorphing_artefacts/Release/Standalone/
```

Attempt the native ONNX build only when ONNX Runtime is installed and discoverable by CMake:

```powershell
cmake -S . -B build-onnx -G "Visual Studio 17 2022" -A x64 -DNEURAL_MORPHING_ENABLE_ONNX=ON -DNM_WITH_PYBRIDGE=OFF
cmake --build build-onnx --config Release --target NeuralMorphing_All
```

## Python Checks

```powershell
.\.venv\Scripts\python.exe -m py_compile python_project_idea.py
.\.venv\Scripts\python.exe -m pytest bridge
```

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
