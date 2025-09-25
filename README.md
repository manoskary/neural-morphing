# neural-morphing
A VST plugin for granular sound shaping with neural audio codecs.

## DAC Export Workflow

1. Install the Hugging Face `transformers`, `torch`, and DAC dependencies in a Python environment.
2. Run `python tools/export_dac.py --output /path/to/dac_export` to dump ONNX/TorchScript encoder/decoder pairs plus embedding tables.
3. Point the plugin at those artefacts by setting the environment variable `NEURAL_MORPHING_MODEL_DIR=/path/to/dac_export` before launching your DAW.
4. Configure CMake with ONNX Runtime available to build the native backend (set `-DNEURAL_MORPHING_ENABLE_ONNX=ON`).

If the environment variable is absent or ONNX Runtime is unavailable, the plugin automatically falls back to the stub backend for development.
