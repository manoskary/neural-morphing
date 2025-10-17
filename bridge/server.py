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

ENCODE_WINDOW_SECONDS = max(float(os.getenv("BRIDGE_ENCODE_WINDOW_SECONDS", "8.0")), 0.5)
DECODE_CHUNK_FRAMES = max(int(os.getenv("BRIDGE_DECODE_CHUNK_FRAMES", "1024")), 1)

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


def _stream_audio_windows(path: str, target_sr: int, window_seconds: float):
    """Yield normalized mono chunks from disk to reduce peak memory usage."""
    try:
        with sf.SoundFile(path) as source:
            src_sr = source.samplerate
            window_frames = max(int(window_seconds * src_sr), 1)

            while True:
                frames = source.read(window_frames, dtype="float32", always_2d=True)
                if frames.size == 0:
                    break

                mono = librosa.to_mono(frames.T)

                if src_sr != target_sr:
                    mono = librosa.resample(mono, orig_sr=src_sr, target_sr=target_sr)

                mono = librosa.util.normalize(mono.astype(np.float32, copy=False))
                if mono.size == 0:
                    continue

                yield mono
    except Exception as exc:
        logger.error("Audio streaming failed for '%s': %s", path, exc)
        raise

def encode_audio(audio_path: str) -> tuple:
    """Encode audio file using DAC"""
    try:
        chunk_codes: List[torch.Tensor] = []
        batch_size = None
        num_codebooks = None

        for chunk_idx, chunk in enumerate(_stream_audio_windows(audio_path, sample_rate, ENCODE_WINDOW_SECONDS)):
            if chunk.size == 0:
                continue

            inputs = dac_processor(
                raw_audio=chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                encoder_outputs = dac_model.encode(inputs["input_values"])

            audio_codes = encoder_outputs.audio_codes.detach().cpu().contiguous()

            if batch_size is None:
                batch_size = int(audio_codes.shape[0])
            elif batch_size != audio_codes.shape[0]:
                raise RuntimeError("Inconsistent batch size encountered during encoding")

            if num_codebooks is None:
                num_codebooks = int(audio_codes.shape[1])
            elif num_codebooks != audio_codes.shape[1]:
                raise RuntimeError("Inconsistent codebook count during encoding")

            if num_codebooks is None:
                num_codebooks = int(audio_codes.shape[1])
            elif num_codebooks != audio_codes.shape[1]:
                raise RuntimeError("Inconsistent codebook count during encoding")

            if audio_codes.shape[-1] == 0:
                continue

            chunk_codes.append(audio_codes)
            logger.info(
                "encode_audio: chunk %s frames=%s tokens=%s",
                chunk_idx,
                audio_codes.shape[-1],
                audio_codes.numel(),
            )

        if not chunk_codes or batch_size is None or num_codebooks is None:
            raise ValueError("Audio file is empty or could not be processed")

        full_codes = torch.cat(chunk_codes, dim=-1)

        total_frames = int(full_codes.shape[2])
        tokens = full_codes.reshape(-1).tolist()

        return batch_size, total_frames, tokens, num_codebooks

    except Exception as e:
        logger.error(f"Failed to encode audio {audio_path}: {e}")
        raise

def tokens_to_vector(tokens: List[int], B: int, T: int, frame_index: int) -> List[float]:
    """Convert tokens to embedding vector for a specific frame"""
    try:
        if frame_index < 0 or frame_index >= T:
            raise ValueError(f"Frame index {frame_index} out of range [0, {T})")

        if B <= 0 or T <= 0:
            raise ValueError("Batch size and frame count must be positive")

        if len(tokens) % (B * T) != 0:
            raise ValueError("Token count is inconsistent with provided batch/frame sizes")

        num_codebooks = len(tokens) // (B * T)

        frame_tokens = torch.empty((B, num_codebooks, 1), dtype=torch.long)

        for b in range(B):
            batch_offset = b * num_codebooks * T
            for c in range(num_codebooks):
                idx = batch_offset + c * T + frame_index
                frame_tokens[b, c, 0] = tokens[idx]

        frame_tokens = frame_tokens.to(device)

        with torch.no_grad():
            quantized_representation, _, _ = dac_model.quantizer.from_codes(frame_tokens)

        frame_vector = quantized_representation[:, :, 0]
        embedding = frame_vector.squeeze(0).detach().cpu().tolist()

        return embedding
        
    except Exception as e:
        logger.error(f"Failed to convert tokens to vector: {e}")
        raise

def decode_tokens(tokens: List[int], B: int, T: int) -> bytes:
    """Decode tokens back to audio using DAC"""
    try:
        if B <= 0 or T <= 0:
            raise ValueError("Batch size and frame count must be positive")

        if len(tokens) % (B * T) != 0:
            raise ValueError("Token count is inconsistent with provided batch/frame sizes")

        num_codebooks = len(tokens) // (B * T)

        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        tokens_tensor = tokens_tensor.reshape(B, num_codebooks, T)

        decoded_audio = None

        for chunk_idx, start in enumerate(range(0, T, DECODE_CHUNK_FRAMES)):
            end = min(start + DECODE_CHUNK_FRAMES, T)
            chunk_codes = tokens_tensor[:, :, start:end].to(device)

            with torch.no_grad():
                decoded_chunk = dac_model.decode(audio_codes=chunk_codes)

            audio_values = decoded_chunk.audio_values if hasattr(decoded_chunk, "audio_values") else decoded_chunk
            audio_chunk = audio_values.detach().cpu()

            logger.info(
                "decode_tokens: chunk %s frames=%s",
                chunk_idx,
                end - start,
            )

            if decoded_audio is None:
                decoded_audio = audio_chunk
            else:
                decoded_audio = torch.cat((decoded_audio, audio_chunk), dim=-1)

        if decoded_audio is None:
            raise ValueError("No audio could be reconstructed from tokens")

        audio_np = decoded_audio.numpy()
        audio_np = np.squeeze(audio_np)

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

        logger.info("/encode start path='%s'", request.path)
        B, T, tokens, num_codebooks = encode_audio(request.path)
        logger.info("/encode done path='%s' frames=%s codebooks=%s tokens=%s", request.path, T, num_codebooks, len(tokens))

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
        logger.info(
            "/tokens_to_vectors start B=%s T=%s frame_index=%s token_count=%s",
            request.B,
            request.T,
            request.frame_index,
            len(request.tokens),
        )
        vector = tokens_to_vector(request.tokens, request.B, request.T, request.frame_index)
        logger.info("/tokens_to_vectors done frame_index=%s vector_len=%s", request.frame_index, len(vector))

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
        logger.info(
            "/decode start B=%s T=%s token_count=%s",
            request.B,
            request.T,
            len(request.tokens),
        )
        wav_bytes = decode_tokens(request.tokens, request.B, request.T)
        logger.info("/decode done wav_bytes=%s", len(wav_bytes))
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
