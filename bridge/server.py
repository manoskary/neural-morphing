#!/usr/bin/env python3
"""
Neural Morphing Python Bridge Server
===================================
FastAPI server that exposes DAC model endpoints for the Neural Morphing plugin.
"""

import os
import base64
import logging
import tempfile
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DacModel, AutoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class EncodeRequest(BaseModel):
    path: str

class EncodeResponse(BaseModel):
    B: int
    T: int
    codebooks: int
    tokens: List[int]
    sample_rate: int

class TokensToVectorsRequest(BaseModel):
    B: int
    T: int
    tokens: List[int]
    frame_index: int = 0

class TokensToVectorsResponse(BaseModel):
    D: int
    vector: List[float]

class DecodeRequest(BaseModel):
    B: int
    T: int
    tokens: List[int]

class DecodeResponse(BaseModel):
    wav_b64: str

class HealthResponse(BaseModel):
    ok: bool
    sample_rate: int
    codebook_count: int
    embedding_dim: int

# Global DAC model instance
dac_model = None
dac_processor = None
sample_rate = 44100
codebook_count = 0
embedding_dim = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Neural Morphing Bridge", version="1.0.0")

def initialize_dac_model():
    """Initialize the DAC model"""
    global dac_model, dac_processor, sample_rate, codebook_count, embedding_dim

    try:
        model_name = os.getenv("DAC_MODEL_NAME", "descript/dac_44khz")
        logger.info("Loading DAC model '%s' on device %s", model_name, device)

        dac_model = DacModel.from_pretrained(model_name)
        dac_model.to(device)
        dac_model.eval()

        dac_processor = AutoProcessor.from_pretrained(model_name)

        sample_rate = getattr(dac_processor, "sampling_rate", sample_rate)
        codebook_count = getattr(dac_model.config, "n_codebooks", 1)

        # Infer embedding dimension from decoder input (quantized representation width)
        embedding_dim = getattr(getattr(dac_model, "decoder", None), "conv1", None)
        if embedding_dim is not None:
            embedding_dim = getattr(embedding_dim, "in_channels", 0)
        if not embedding_dim:
            # Fallback: run zero codes through quantizer to infer dim
            with torch.no_grad():
                dummy_codes = torch.zeros((1, codebook_count, 1), dtype=torch.long, device=device)
                quantized_rep, _, _ = dac_model.quantizer.from_codes(dummy_codes)
                embedding_dim = int(quantized_rep.shape[1])

        logger.info(
            "DAC model ready (sample_rate=%s, codebooks=%s, embedding_dim=%s)",
            sample_rate,
            codebook_count,
            embedding_dim,
        )

    except Exception as e:
        logger.error(f"Failed to load DAC model: {e}")
        raise

def encode_audio(audio_path: str) -> tuple:
    """Encode audio file using DAC"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        if len(audio) == 0:
            raise ValueError("Audio file is empty or could not be loaded")
            
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Process with DAC
        inputs = dac_processor(
            raw_audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            encoder_outputs = dac_model.encode(inputs["input_values"])
            audio_codes = encoder_outputs.audio_codes

        # Convert to list of integers
        tokens = audio_codes.detach().cpu().reshape(-1).tolist()

        # Get dimensions
        B, num_codebooks, T = audio_codes.shape

        return B, T, tokens, num_codebooks

    except Exception as e:
        logger.error(f"Failed to encode audio {audio_path}: {e}")
        raise

def tokens_to_vector(tokens: List[int], B: int, T: int, frame_index: int) -> List[float]:
    """Convert tokens to embedding vector for a specific frame"""
    try:
        if frame_index < 0 or frame_index >= T:
            raise ValueError(f"Frame index {frame_index} out of range [0, {T})")
            
        # Reshape tokens back to codebook format
        num_codebooks = len(tokens) // (B * T) if T > 0 else 1
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        tokens_tensor = tokens_tensor.reshape(B, num_codebooks, T)

        with torch.no_grad():
            quantized_representation, _, _ = dac_model.quantizer.from_codes(tokens_tensor)

        frame_vector = quantized_representation[:, :, frame_index]
        embedding = frame_vector.squeeze(0).detach().cpu().tolist()

        return embedding
        
    except Exception as e:
        logger.error(f"Failed to convert tokens to vector: {e}")
        raise

def decode_tokens(tokens: List[int], B: int, T: int) -> bytes:
    """Decode tokens back to audio using DAC"""
    try:
        # Reshape tokens back to tensor
        num_codebooks = len(tokens) // (B * T) if T > 0 else 1
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        tokens_tensor = tokens_tensor.reshape(B, num_codebooks, T)

        with torch.no_grad():
            decoded = dac_model.decode(audio_codes=tokens_tensor)

        audio_values = decoded.audio_values if hasattr(decoded, "audio_values") else decoded
        audio_np = audio_values.detach().cpu().numpy()
        audio_np = np.squeeze(audio_np)

        # Ensure shape (samples,) for mono signals
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_np, sample_rate)

            with open(tmp_file.name, "rb") as f:
                wav_bytes = f.read()

        os.unlink(tmp_file.name)

        return wav_bytes
        
    except Exception as e:
        logger.error(f"Failed to decode tokens: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the DAC model on startup"""
    initialize_dac_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        ok=True,
        sample_rate=sample_rate,
        codebook_count=codebook_count or 1,
        embedding_dim=embedding_dim or (codebook_count or 1)
    )

@app.post("/encode", response_model=EncodeResponse)
async def encode_endpoint(request: EncodeRequest):
    """Encode audio file to tokens"""
    try:
        if not os.path.exists(request.path):
            raise HTTPException(status_code=404, detail=f"Audio file not found: {request.path}")
            
        B, T, tokens, num_codebooks = encode_audio(request.path)

        return EncodeResponse(
            B=B,
            T=T,
            codebooks=num_codebooks,
            tokens=tokens,
            sample_rate=sample_rate
        )
        
    except Exception as e:
        logger.error(f"Encode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokens_to_vectors", response_model=TokensToVectorsResponse)
async def tokens_to_vectors_endpoint(request: TokensToVectorsRequest):
    """Convert tokens to embedding vector for a specific frame"""
    try:
        vector = tokens_to_vector(request.tokens, request.B, request.T, request.frame_index)
        
        return TokensToVectorsResponse(
            D=len(vector),
            vector=vector
        )
        
    except Exception as e:
        logger.error(f"Tokens to vectors error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decode", response_model=DecodeResponse)
async def decode_endpoint(request: DecodeRequest):
    """Decode tokens back to audio"""
    try:
        wav_bytes = decode_tokens(request.tokens, request.B, request.T)
        wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        return DecodeResponse(wav_b64=wav_b64)
        
    except Exception as e:
        logger.error(f"Decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "localhost")
    
    logger.info(f"Starting Neural Morphing Bridge Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
