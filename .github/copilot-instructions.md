# Copilot Instructions for neural-morphing

## Project Overview

This is a JUCE-based VST3/AU audio plugin that performs real-time neural audio morphing using latent representations from the Descript Audio Codec (DAC). The plugin allows users to morph incoming audio into a palette of target sounds.

## Architecture & Key Components

### Core Plugin Structure (C++)
- **Source/PluginProcessor.{h,cpp}** - Main audio processor implementing JUCE AudioProcessor interface
- **Source/PluginEditor.{h,cpp}** - GUI components and user interface
- **Source/ModelBackend.{h,cpp}** - Abstract interface for neural model backends
- **Source/ModelBackendOnnx.{h,cpp}** - ONNX Runtime implementation for DAC inference
- **Source/Workers.{h,cpp}** - Background worker threads for audio processing
- **Source/OnsetDetector.{h,cpp}** - Real-time onset detection for audio segmentation
- **Source/PaletteIndex.{h,cpp}** - Palette management and nearest neighbor search
- **Source/LockFreeRing.h** - Lock-free circular buffer for real-time audio

### Python Tools
- **tools/export_dac.py** - Export DAC models to ONNX format for plugin use
- **python_project_idea.py** - Original Python prototype exploring morphing algorithms
- **requirements.txt** - Python dependencies for tools and prototyping

## Technology Stack

### C++ Development
- **Language**: C++17 standard
- **Framework**: JUCE 8.x for audio plugin development
- **Build System**: CMake ≥ 3.22
- **Neural Inference**: ONNX Runtime (optional, falls back to stub backend)
- **Plugin Formats**: VST3, AU (Audio Units)

### Python Environment
- **ML Libraries**: torch, transformers, onnx, onnxruntime
- **Audio Processing**: librosa, soundfile, scipy
- **Model Hub**: huggingface_hub for DAC model access

## Build Configuration

### CMake Options
- `NEURAL_MORPHING_ENABLE_ONNX=ON/OFF` - Enable/disable ONNX Runtime backend
- Release builds recommended for performance-critical audio processing

### Environment Variables
- `NEURAL_MORPHING_MODEL_DIR` - Path to exported DAC model artifacts

## Development Workflow

### Building the Plugin
```bash
cmake -S . -B build -DNEURAL_MORPHING_ENABLE_ONNX=ON
cmake --build build --config Release
```

### Python Development
1. Create virtual environment: `python -m venv .venv`
2. Install dependencies: `pip install -r requirements.txt`
3. Export DAC models: `python tools/export_dac.py --output /path --model descript/dac_44khz`

## Coding Conventions

### C++ Style
- Follow JUCE conventions for class naming and structure
- Use RAII and smart pointers for memory management
- Implement real-time safe audio processing (no allocations in audio thread)
- Lock-free data structures for inter-thread communication
- Extensive use of JUCE framework classes and patterns

### File Organization
- Header files contain class declarations with inline simple methods
- Implementation files contain complex logic and audio processing algorithms
- Separate concerns: UI, audio processing, backend abstraction, workers

### Real-Time Audio Constraints
- No dynamic memory allocation in `processBlock()` and audio callbacks
- Use lock-free data structures for producer-consumer patterns
- Minimize computational complexity in audio thread
- Delegate heavy processing to background worker threads

## Key Development Areas

### Audio Processing Pipeline
1. Onset detection for audio segmentation
2. Neural encoding using DAC encoder
3. Latent space matching against palette
4. Neural decoding to generate morphed audio
5. Real-time buffering and streaming

### Backend Abstraction
- Stub backend for development without ONNX Runtime
- ONNX backend for production neural inference
- Extensible architecture for additional ML frameworks

### Current TODO Items
- Replace linear palette search with HNSWlib ANN implementation
- Implement sophisticated latent matching (temperature, threshold, envelope follow)
- Add production UI with file browsers and progress indicators
- Implement persistent plugin state and settings
- Add automated testing and CI pipeline

## Dependencies & Platform Support

### Required
- JUCE framework (included as submodule)
- CMake ≥ 3.22, C++17 compiler
- Platform-specific JUCE dependencies (X11 libs on Linux, etc.)

### Optional
- ONNX Runtime for neural inference backend
- Python environment for model export and prototyping

### Supported Platforms
- macOS (AU/VST3 with automatic installation)
- Windows (VST3 with automatic installation)
- Linux (VST3 with manual installation)

## Testing & Debugging

### Development Backend
- Stub backend allows development without ML dependencies
- Automatic fallback when ONNX Runtime unavailable
- Environment variable controls model loading

### Audio Testing
- Test with various sample rates and buffer sizes
- Validate real-time performance and latency
- Check thread safety and lock-free operation
- Verify proper JUCE plugin lifecycle handling

When contributing to this project, prioritize real-time audio performance, follow JUCE best practices, and maintain the clean separation between audio processing and neural inference backends.