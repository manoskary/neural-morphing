<p align="center">
  <img src="assets/neural_morphing_title.png" alt="Neural Morphing" width="1000">
</p>

Neural audio morphing effect for VST3 hosts, Standalone use, and hosted Gradio review.

This repository contains:

- `Source/`, `Resources/`, `JUCE/`, `CMakeLists.txt` - VST3 and Standalone build tree.
- `bridge/` - optional Python bridge runtime used by the plugin.
- `python_project_idea.py`, `assets/`, `examples/` - Gradio demo app with curated examples.
- `Dockerfile`, `cloudbuild.yaml` - Cloud Run deployment files.

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

## Install The VST3 And Standalone

There is no packaged installer yet. The steps below build the plugin from source, copy the VST3 bundle to the standard user location, and run the Standalone directly from the build directory.

### Common Prerequisites

- Git and CMake 3.22 or newer.
- Python 3.10 through 3.12 for the recommended Python bridge backend.
- A C++17 compiler supported by CMake.
- A VST3-compatible DAW for using the plugin version.

### 1. Clone The Repository

Clone with the JUCE submodule:

```bash
git clone --branch demo --recurse-submodules https://github.com/manoskary/neural-morphing.git
cd neural-morphing
```

For an existing checkout, initialize JUCE with:

```bash
git submodule update --init --recursive
```

### 2. Install And Start The Python Bridge

The bridge is the simplest backend to set up. It is a separate process: starting the Standalone or loading the VST3 does **not** start it automatically.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\bridge\requirements.txt
.\.venv\Scripts\python.exe .\bridge\server.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r bridge/requirements.txt
.venv/bin/python bridge/server.py
```

Keep that terminal open while using Neural Morphing. The bridge listens on `http://localhost:8000`, supports CPU and CUDA, and downloads the DAC model from Hugging Face on first use. To force a device, set `BRIDGE_DEVICE` to `cpu` or `cuda` before starting it. macOS currently uses the CPU backend because the bridge does not yet select Apple Metal/MPS.

### 3. Build And Install

The commands below build both the Standalone and VST3. Copy the complete `.vst3` bundle, not only the library inside it. The install locations follow the [official VST3 platform paths](https://steinbergmedia.github.io/vst3_dev_portal/pages/Technical%2BDocumentation/Locations%2BFormat/Plugin%2BLocations.html).

#### Windows

Install Visual Studio 2022 with the **Desktop development with C++** workload, then run:

```powershell
cmake -S . -B build-bridge -G "Visual Studio 17 2022" -A x64 -DNEURAL_MORPHING_ENABLE_ONNX=OFF -DNM_WITH_PYBRIDGE=ON
cmake --build build-bridge --config Release --target NeuralMorphing_Standalone NeuralMorphing_VST3
```

The build produces:

```text
build-bridge/NeuralMorphing_artefacts/Release/VST3/Neural Morphing.vst3/
build-bridge/NeuralMorphing_artefacts/Release/Standalone/Neural Morphing.exe
```

```powershell
$vst3Directory = "$env:LOCALAPPDATA\Programs\Common\VST3"
New-Item -ItemType Directory -Force $vst3Directory | Out-Null
Copy-Item ".\build-bridge\NeuralMorphing_artefacts\Release\VST3\Neural Morphing.vst3" $vst3Directory -Recurse -Force
& ".\build-bridge\NeuralMorphing_artefacts\Release\Standalone\Neural Morphing.exe"
```

For a system-wide VST3 installation, use `C:\Program Files\Common Files\VST3` from an Administrator PowerShell terminal.

#### macOS

Install the Xcode command-line tools and configure an Xcode build:

```bash
xcode-select --install
cmake -S . -B build-bridge -G Xcode -DNEURAL_MORPHING_ENABLE_ONNX=OFF -DNM_WITH_PYBRIDGE=ON
cmake --build build-bridge --config Release --target NeuralMorphing_Standalone NeuralMorphing_VST3
```

The build produces:

```text
build-bridge/NeuralMorphing_artefacts/Release/VST3/Neural Morphing.vst3/
build-bridge/NeuralMorphing_artefacts/Release/Standalone/Neural Morphing.app/
```

Install the VST3 for the current user and launch the Standalone:

```bash
mkdir -p "$HOME/Library/Audio/Plug-Ins/VST3"
cp -R "build-bridge/NeuralMorphing_artefacts/Release/VST3/Neural Morphing.vst3" "$HOME/Library/Audio/Plug-Ins/VST3/"
open "build-bridge/NeuralMorphing_artefacts/Release/Standalone/Neural Morphing.app"
```

The system-wide VST3 location is `/Library/Audio/Plug-Ins/VST3` and requires administrator access.

#### Linux

On Ubuntu or Debian, install the compiler and JUCE dependencies:

```bash
sudo apt update
sudo apt install build-essential cmake git pkg-config \
  libasound2-dev libjack-jackd2-dev libcurl4-openssl-dev \
  libfreetype-dev libfontconfig1-dev \
  libx11-dev libxcomposite-dev libxcursor-dev libxext-dev \
  libxinerama-dev libxrandr-dev libxrender-dev
```

Package names differ on other distributions. Configure a Release build with the system compiler:

```bash
cmake -S . -B build-bridge -DCMAKE_BUILD_TYPE=Release \
  -DNEURAL_MORPHING_ENABLE_ONNX=OFF -DNM_WITH_PYBRIDGE=ON
cmake --build build-bridge --target NeuralMorphing_Standalone NeuralMorphing_VST3 --parallel
```

The build produces:

```text
build-bridge/NeuralMorphing_artefacts/Release/VST3/Neural Morphing.vst3/
build-bridge/NeuralMorphing_artefacts/Release/Standalone/Neural Morphing
```

Install the VST3 for the current user and launch the Standalone:

```bash
mkdir -p "$HOME/.vst3"
cp -R "build-bridge/NeuralMorphing_artefacts/Release/VST3/Neural Morphing.vst3" "$HOME/.vst3/"
"build-bridge/NeuralMorphing_artefacts/Release/Standalone/Neural Morphing"
```

System-wide VST3 locations are `/usr/lib/vst3` and `/usr/local/lib/vst3` and require root access.

### 4. First Test

Restart the DAW or rescan its plugins, then insert **Neural Morphing** as an audio effect. The DAW track supplies the source audio; the Standalone uses the source sound loaded from its interface.

Add palette sounds and select **Bridge Only**, **DAC**, and **Palette Only** for the most direct first test. If the bridge is connected, the status line reports `backend=python_bridge` and eventually shows wet/token activity.

### Optional Native ONNX Backend

The native backend does not need the Python bridge while running, but it requires an ONNX Runtime C++ SDK for the target operating system and exported DAC model files.

On Windows, configure it with the Visual Studio generator:

```powershell
cmake -S . -B build-onnx -G "Visual Studio 17 2022" -A x64 -DNEURAL_MORPHING_ENABLE_ONNX=ON -DNM_WITH_PYBRIDGE=OFF -DONNXRUNTIME_ROOT="C:\path\to\onnxruntime" -DNEURAL_MORPHING_DAC_MODEL_DIR="C:\path\to\exported-dac-model"
cmake --build build-onnx --config Release --target NeuralMorphing_Standalone NeuralMorphing_VST3
& ".\build-onnx\NeuralMorphing_artefacts\Release\Standalone\Neural Morphing.exe"
```

On macOS or Linux, use the same compiler setup as the bridge build:

```bash
cmake -S . -B build-onnx -DCMAKE_BUILD_TYPE=Release \
  -DNEURAL_MORPHING_ENABLE_ONNX=ON -DNM_WITH_PYBRIDGE=OFF \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DNEURAL_MORPHING_DAC_MODEL_DIR=/path/to/exported-dac-model
cmake --build build-onnx --target NeuralMorphing_Standalone NeuralMorphing_VST3 --parallel
```

The build bundles the DAC files beside the Standalone and inside the VST3. `NEURAL_MORPHING_MODEL_DIR` can still override that location. Keep the ONNX Runtime shared library installed or discoverable from its SDK `lib` directory when launching on macOS or Linux; Windows builds copy `onnxruntime.dll` automatically.

The model directory must contain `encoder.onnx`, `decoder.onnx`, `embeddings.bin`, and `metadata.json`. `tools/export_dac.py` creates this layout.

## Developer Validation

Optional Standalone smoke-test preload:

```powershell
$env:NEURAL_MORPHING_DEMO_PALETTE_FILES="C:\path\palette_a.wav;C:\path\palette_b.wav"
$env:NEURAL_MORPHING_DEMO_SOURCE_FILE="C:\path\source.wav"
$env:NEURAL_MORPHING_DEMO_STATUS_FILE="C:\path\status.txt"
& ".\build-bridge\NeuralMorphing_artefacts\Release\Standalone\Neural Morphing.exe"
```

When the bridge and palette are ready, the status line should show wet activity and `tok=NN%`.
For individual smoke overrides, set `NEURAL_MORPHING_DEMO_TEMPERATURE`, `NEURAL_MORPHING_DEMO_THRESHOLD`, `NEURAL_MORPHING_DEMO_CONTINUITY`, `NEURAL_MORPHING_DEMO_RVQ_FOCUS`, `NEURAL_MORPHING_DEMO_PALETTE_BIAS`, `NEURAL_MORPHING_DEMO_GRAIN_SIZE`, `NEURAL_MORPHING_DEMO_GRAIN_STEP`, `NEURAL_MORPHING_DEMO_ENVELOPE`, or `NEURAL_MORPHING_DEMO_DRY_WET`. `sig` fingerprints the decoded palette wet signal; `out` fingerprints the final volume mix.

Run the deterministic control sweep with the bridge already listening on port 8000:

```powershell
.\tools\sweep_plugin_controls.ps1
```

Use `Render HQ WAV` in the Standalone to export the loaded source through the current palette and morph settings. In a DAW, use `Arm HQ Render` before the host freeze/bounce/export; the VST cannot render the whole host track by itself.

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
