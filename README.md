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
- Continuity: `0.10`
- Envelope: `0.05`
- Playback Dry/Wet: `0.9`
- Match Mode: `beam`
- Swap Mode: `palette_only`

In the plugin, `Palette Bias` chooses similar versus adventurous palette grains, `Temperature` controls deterministic variation, `Continuity` controls temporal coherence, and `RVQ Focus` moves matching between coarse and fine DAC layers. `Grain Size` and `Grain Step` repool cached palette embeddings without re-encoding the palette sounds.

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

Optional Standalone smoke-test preload:

```powershell
$env:NEURAL_MORPHING_DEMO_PALETTE_FILES="C:\path\palette_a.wav;C:\path\palette_b.wav"
$env:NEURAL_MORPHING_DEMO_SOURCE_FILE="C:\path\source.wav"
$env:NEURAL_MORPHING_DEMO_STATUS_FILE="C:\path\status.txt"
& ".\build-bridge\NeuralMorphing_artefacts\Release\Standalone\Neural Morphing.exe"
```

When the bridge and palette are ready, the status line should show `wet=ready` and `tok=NN%`.
For individual smoke overrides, set `NEURAL_MORPHING_DEMO_TEMPERATURE`, `NEURAL_MORPHING_DEMO_THRESHOLD`, `NEURAL_MORPHING_DEMO_CONTINUITY`, `NEURAL_MORPHING_DEMO_RVQ_FOCUS`, `NEURAL_MORPHING_DEMO_PALETTE_BIAS`, `NEURAL_MORPHING_DEMO_GRAIN_SIZE`, `NEURAL_MORPHING_DEMO_GRAIN_STEP`, `NEURAL_MORPHING_DEMO_ENVELOPE`, or `NEURAL_MORPHING_DEMO_DRY_WET`. `sig` fingerprints the decoded palette wet signal; `out` fingerprints the final volume mix.

Run the deterministic control sweep with the bridge already listening on port 8000:

```powershell
.\tools\sweep_plugin_controls.ps1
```

Use `Render HQ WAV` in the Standalone to export the loaded source through the current palette and morph settings. In a DAW, use `Arm HQ Render` before the host freeze/bounce/export; the VST cannot render the whole host track by itself.

Attempt the native ONNX build only when ONNX Runtime is installed and discoverable by CMake:

```powershell
cmake -S . -B build-onnx -G "Visual Studio 17 2022" -A x64 -DNEURAL_MORPHING_ENABLE_ONNX=ON -DNM_WITH_PYBRIDGE=OFF -DONNXRUNTIME_ROOT="C:\path\to\onnxruntime"
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
