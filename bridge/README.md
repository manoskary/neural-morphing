# Neural Morphing Python Bridge

This directory contains the Python FastAPI server that acts as a bridge between the Neural Morphing JUCE plugin and the Python DAC (Descript Audio Codec) model.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   python server.py
   ```

The server will start on `http://localhost:8000` by default.

## Environment Variables

- `DAC_MODEL_NAME`: DAC model to use (default: "descript/dac_44khz")
- `HOST`: Server host (default: "localhost")
- `PORT`: Server port (default: "8000")

## Endpoints

- `GET /health`: Health check and server info
- `POST /encode`: Encode audio file to tokens
- `POST /tokens_to_vectors`: Convert tokens to embedding vectors
- `POST /decode`: Decode tokens back to audio

## Plugin Configuration

In the Neural Morphing plugin:
1. Select "Python Bridge" as the backend
2. Make sure the server URL matches the running server (default: http://localhost:8000)
3. The plugin will automatically connect to the server when processing audio

## Development

The server automatically reloads when you modify the code if you run it with:
```bash
uvicorn server:app --reload
```