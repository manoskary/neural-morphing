#!/usr/bin/env python3
"""
Neural Morphing Python Bridge Server
===================================
FastAPI server that exposes DAC model endpoints for the Neural Morphing plugin.
"""

import os
import base64
import logging
from pathlib import Path
import tempfile
import soundfile as sf
from typing import List, Dict, Any

import torch
import numpy as np
import librosa
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

app = FastAPI(title="Neural Morphing Bridge", version="1.0.0")

def initialize_dac_model():
    """Initialize the DAC model"""
    global dac_model, dac_processor, sample_rate
    
    try:
        model_name = os.getenv("DAC_MODEL_NAME", "descript/dac_44khz")
        logger.info(f"Loading DAC model: {model_name}")
        
        dac_model = DacModel.from_pretrained(model_name, device_map="auto")
        dac_processor = AutoProcessor.from_pretrained(model_name, device="auto")
        
        sample_rate = dac_processor.sampling_rate
        logger.info(f"DAC model loaded successfully. Sample rate: {sample_rate} Hz")
        
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
        inputs = {k: v.to(dac_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            encoder_outputs = dac_model.encode(inputs["input_values"])
            audio_codes = encoder_outputs.audio_codes
            
        # Convert to list of integers
        tokens = audio_codes.cpu().numpy().flatten().astype(int).tolist()
        
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
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).reshape(B, num_codebooks, T)
        
        # Extract tokens for the specific frame
        frame_tokens = tokens_tensor[:, :, frame_index]  # Shape: [B, num_codebooks]
        
        # Get embeddings (simplified approach - just use token values as features)
        # In a real implementation, you might want to use learned embeddings
        embedding = frame_tokens.float().flatten().tolist()
        
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to convert tokens to vector: {e}")
        raise

def decode_tokens(tokens: List[int], B: int, T: int) -> bytes:
    """Decode tokens back to audio using DAC"""
    try:
        # Reshape tokens back to tensor
        num_codebooks = len(tokens) // (B * T) if T > 0 else 1
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).reshape(B, num_codebooks, T)
        tokens_tensor = tokens_tensor.to(dac_model.device)
        
        with torch.no_grad():
            # Decode using DAC
            audio_values = dac_model.decode(tokens_tensor)
            
        # Get the actual audio data
        if hasattr(audio_values, 'audio_values'):
            audio_data = audio_values.audio_values
        else:
            audio_data = audio_values
            
        # Convert to numpy and ensure it's the right shape
        audio_np = audio_data.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
            
        # Save to temporary WAV file and return as bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_np, sample_rate)
            
            # Read back as bytes
            with open(tmp_file.name, 'rb') as f:
                wav_bytes = f.read()
                
            # Clean up
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
        codebook_count=1,  # DAC typically uses multiple codebooks, but we'll simplify
        embedding_dim=128  # This would depend on your specific embedding approach
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