# neural-morphing
A JUCE-based VST3/AU insert effect that morphs incoming audio into a palette of target sounds using latent representations from the Descript Audio Codec (DAC).

## Project Layout

- `Source/` – JUCE plugin sources (processor, editor, workers, backend interface).
- `tools/export_dac.py` – utility to export DAC encoder/decoder artefacts for the native backend.
- `python_project_idea.py` – original Python prototype used to explore the morphing pipeline.
- `requirements.txt` – Python dependencies for the export tool and prototype notebooks.

## Prerequisites

- **JUCE** checked out as a submodule (`git submodule update --init --recursive`).
- **CMake ≥ 3.22** and a C++17 compiler.
- Platform SDKs required by JUCE (on Linux: `libxrandr-dev libxinerama-dev libxcursor-dev libxrender-dev libfreetype6-dev libfontconfig1-dev`).
- Optional: ONNX Runtime development package if you plan to build the native DAC backend (`onnxruntime-dev` on Ubuntu, or use the prebuilt SDK).

### Python environment

Create a virtual environment and install the Python requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This provides the tooling needed for DAC export (`tools/export_dac.py`) and for experimenting with the original Python morphing prototype.

## Building the Plugin

```bash
cmake -S . -B build -DNEURAL_MORPHING_ENABLE_ONNX=ON
cmake --build build --config Release
```

If ONNX Runtime is not installed, either install it or pass `-DNEURAL_MORPHING_ENABLE_ONNX=OFF` to build against the lightweight stub backend. The generated VST3/AU artefacts will be placed in `build/NeuralMorphing_artefacts/`.

After a successful build you can install the plugin:

- macOS: `cmake --install build --config Release`
- Windows: `cmake --install build --config Release`
- Linux: copy the produced VST3 bundle manually to your plugin directory.

## DAC Export Workflow

1. Install the Hugging Face `transformers`, `torch`, and DAC dependencies (`pip install -r requirements.txt`).
2. Export the encoder/decoder and embeddings:
   ```bash
   python tools/export_dac.py --output /path/to/dac_export --model descript/dac_44khz
   ```
3. Point the plugin at the export directory by setting the environment variable `NEURAL_MORPHING_MODEL_DIR=/path/to/dac_export` before launching your DAW.
4. Ensure the CMake build is configured with ONNX Runtime available (`-DNEURAL_MORPHING_ENABLE_ONNX=ON`).

If the environment variable is absent or ONNX Runtime is unavailable, the plugin automatically falls back to the stub backend for development.

## TODO

- [x] Scaffold JUCE VST3/AU project with parameter set, workers, and lock-free FIFOs.
- [x] Provide DAC export tool and ONNX Runtime backend stub integration.
- [ ] Replace linear `PaletteIndex` with HNSWlib ANN implementation and metadata catalogue.
- [ ] Implement real latent segment matching in `MatchWorker` (temperature, threshold, stride, envelope follow).
- [ ] Integrate onset detector refinements (spectral flux, look-ahead) tuned for drum/percussive sources.
- [ ] Add persistent plugin state for palette file lists and backend settings.
- [ ] Build production-ready UI (file browser, progress meters, error messaging, parameter grouping).
- [ ] Add automated tests/CI scripts for export tooling and backend loading.
