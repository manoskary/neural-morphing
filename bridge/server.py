#!/usr/bin/env python3
"""
Neural Morphing Python Bridge Server
===================================
FastAPI server exposing multiple codec backends (DAC, SpectroStream) for the
Neural Morphing JUCE plugin.
"""

import base64
import io
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, DacModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN_LAYOUT_CODEBOOK_MAJOR = "codebook_major"
TOKEN_LAYOUT_FRAME_MAJOR = "frame_major"
PCM_LAYOUT_INTERLEAVED = "interleaved"
PCM_DTYPE_FLOAT32 = "float32le"

ENCODE_WINDOW_SECONDS = max(float(os.getenv("BRIDGE_ENCODE_WINDOW_SECONDS", "8.0")), 0.5)
DECODE_CHUNK_FRAMES = max(int(os.getenv("BRIDGE_DECODE_CHUNK_FRAMES", "1024")), 1)
BRIDGE_CODEC_DEFAULT = os.getenv("NEURAL_MORPHING_BRIDGE_CODEC", "dac").strip().lower()
BRIDGE_WARM_START = os.getenv("BRIDGE_WARM_START", "0").strip() in ("1", "true", "TRUE", "yes", "on")
BRIDGE_ENABLE_STREAM_SESSIONS = os.getenv("BRIDGE_ENABLE_STREAM_SESSIONS", "0").strip() in ("1", "true", "TRUE", "yes", "on")
FORCE_CPU_BACKENDS = False


class EncodeRequest(BaseModel):
    path: str


class EncodeResponse(BaseModel):
    B: int
    T: int
    codebooks: int
    tokens: List[int]
    sample_rate: int
    token_layout: str = TOKEN_LAYOUT_CODEBOOK_MAJOR


class EncodePCMRequest(BaseModel):
    sample_rate: int
    channels: int
    num_samples: int
    dtype: str = PCM_DTYPE_FLOAT32
    pcm_layout: str = PCM_LAYOUT_INTERLEAVED
    pcm_b64: str


class EncodePCMResponse(EncodeResponse):
    pass


class TokensToVectorsRequest(BaseModel):
    B: int
    T: int
    tokens: List[int]
    frame_index: int = 0
    codebooks: Optional[int] = None
    token_layout: str = TOKEN_LAYOUT_CODEBOOK_MAJOR


class TokensToVectorsResponse(BaseModel):
    D: int
    vector: List[float]


class TokensToVectorsBatchRequest(BaseModel):
    B: int
    T: int
    tokens: List[int]
    start_frame: int = 0
    frame_count: int = 0
    codebooks: Optional[int] = None
    token_layout: str = TOKEN_LAYOUT_CODEBOOK_MAJOR


class TokensToVectorsBatchResponse(BaseModel):
    D: int
    count: int
    vectors: List[List[float]]


class DecodeRequest(BaseModel):
    B: int
    T: int
    tokens: List[int]
    codebooks: Optional[int] = None
    token_layout: str = TOKEN_LAYOUT_CODEBOOK_MAJOR


class DecodeResponse(BaseModel):
    wav_b64: str


class DecodePCMRequest(DecodeRequest):
    pass


class DecodePCMResponse(BaseModel):
    sample_rate: int
    channels: int
    num_samples: int
    dtype: str = PCM_DTYPE_FLOAT32
    pcm_layout: str = PCM_LAYOUT_INTERLEAVED
    pcm_b64: str


class HealthResponse(BaseModel):
    ok: bool
    sample_rate: int
    codebook_count: int
    embedding_dim: int


class CapabilitiesResponse(BaseModel):
    ok: bool
    supported_codecs: List[str]
    active_codec: str
    sample_rate: int
    codebook_count: int
    embedding_dim: int
    required_input_channels: int
    frame_rate_hz: float
    token_layout: str
    supports_pcm_endpoints: bool


class CodecRequest(BaseModel):
    codec: str


class CodecResponse(CapabilitiesResponse):
    pass


class SessionOpenResponse(BaseModel):
    ok: bool
    session_id: str


class SessionPushRequest(EncodePCMRequest):
    session_id: str


class SessionPushResponse(BaseModel):
    ok: bool
    accepted_samples: int


class SessionPullRequest(BaseModel):
    session_id: str


class SessionPullResponse(BaseModel):
    ok: bool
    sample_rate: int
    channels: int
    num_samples: int
    dtype: str = PCM_DTYPE_FLOAT32
    pcm_layout: str = PCM_LAYOUT_INTERLEAVED
    pcm_b64: str


class SessionCloseRequest(BaseModel):
    session_id: str


class SessionCloseResponse(BaseModel):
    ok: bool


@dataclass
class CodecMetadata:
    sample_rate: int
    codebook_count: int
    embedding_dim: int
    required_input_channels: int
    frame_rate_hz: float


@dataclass
class TokenBlock:
    B: int
    T: int
    codebooks: int
    tokens: List[int]
    token_layout: str = TOKEN_LAYOUT_CODEBOOK_MAJOR


def _frame_major_to_codebook_major(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim != 2:
        raise ValueError(f"Expected [T, K] tokens, got shape={tokens.shape}")
    return tokens.T.reshape(-1)


def _codebook_major_to_frame_major(tokens: Sequence[int], codebooks: int, frames: int) -> np.ndarray:
    if codebooks <= 0 or frames <= 0:
        raise ValueError("codebooks and frames must be positive")
    arr = np.asarray(tokens, dtype=np.int32)
    expected = codebooks * frames
    if arr.size != expected:
        raise ValueError(f"Expected {expected} tokens, got {arr.size}")
    return arr.reshape(codebooks, frames).T


def _normalize_token_layout(layout: Optional[str]) -> str:
    if isinstance(layout, str) and layout.strip().lower() == TOKEN_LAYOUT_FRAME_MAJOR:
        return TOKEN_LAYOUT_FRAME_MAJOR
    return TOKEN_LAYOUT_CODEBOOK_MAJOR


def _decode_pcm_payload(
    pcm_b64: str,
    *,
    channels: int,
    num_samples: int,
    dtype: str,
    pcm_layout: str,
) -> np.ndarray:
    if dtype != PCM_DTYPE_FLOAT32:
        raise ValueError(f"Unsupported PCM dtype: {dtype}")
    if pcm_layout != PCM_LAYOUT_INTERLEAVED:
        raise ValueError(f"Unsupported PCM layout: {pcm_layout}")
    if channels <= 0:
        raise ValueError("channels must be positive")

    payload = base64.b64decode(pcm_b64.encode("utf-8"))
    audio = np.frombuffer(payload, dtype="<f4")
    if num_samples <= 0:
        if audio.size % channels != 0:
            raise ValueError("PCM payload size is inconsistent with channel count")
        num_samples = audio.size // channels

    expected = num_samples * channels
    if audio.size < expected:
        raise ValueError(f"PCM payload too short: expected {expected} float32 values, got {audio.size}")

    return np.ascontiguousarray(audio[:expected].reshape(num_samples, channels), dtype=np.float32)


def _encode_pcm_payload(samples: np.ndarray) -> tuple[str, int, int]:
    if samples.ndim != 2:
        raise ValueError(f"Expected [N, C] samples, got shape={samples.shape}")
    samples = np.ascontiguousarray(samples.astype(np.float32, copy=False))
    interleaved = samples.reshape(-1)
    pcm_b64 = base64.b64encode(interleaved.tobytes(order="C")).decode("utf-8")
    return pcm_b64, samples.shape[0], samples.shape[1]


class CodecAdapter:
    codec_id: str = "base"

    def ensure_loaded(self) -> None:
        raise NotImplementedError

    def metadata(self) -> CodecMetadata:
        raise NotImplementedError

    def encode_samples(self, samples: np.ndarray, sample_rate: int) -> TokenBlock:
        raise NotImplementedError

    def tokens_to_vector_row(self, block: TokenBlock, frame_index: int) -> np.ndarray:
        raise NotImplementedError

    def tokens_to_vector_rows(self, block: TokenBlock, start_frame: int, frame_count: int) -> np.ndarray:
        if frame_count <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: List[np.ndarray] = []
        for frame_index in range(start_frame, start_frame + frame_count):
            vectors.append(np.asarray(self.tokens_to_vector_row(block, frame_index), dtype=np.float32))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(vectors, axis=0).astype(np.float32, copy=False)

    def decode_tokens(self, block: TokenBlock) -> np.ndarray:
        raise NotImplementedError

    def warm_start(self) -> None:
        # Optional no-op.
        return


class DacAdapter(CodecAdapter):
    codec_id = "dac"

    def __init__(self, device: torch.device):
        self._device = device
        self._model_name = os.getenv("DAC_MODEL_NAME", "descript/dac_44khz")
        self._model: Optional[DacModel] = None
        self._processor = None
        self._metadata: Optional[CodecMetadata] = None

    def ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None and self._metadata is not None:
            return

        logger.info("Loading DAC model '%s' on device %s", self._model_name, self._device)
        model = DacModel.from_pretrained(self._model_name)
        model.to(self._device)
        model.eval()
        processor = AutoProcessor.from_pretrained(self._model_name)

        sample_rate = int(getattr(processor, "sampling_rate", 44100))
        codebook_count = int(getattr(model.config, "n_codebooks", 1))

        embedding_dim = getattr(getattr(model, "decoder", None), "conv1", None)
        if embedding_dim is not None:
            embedding_dim = getattr(embedding_dim, "in_channels", 0)
        if not embedding_dim:
            with torch.no_grad():
                dummy_codes = torch.zeros((1, codebook_count, 1), dtype=torch.long, device=self._device)
                quantized_rep, _, _ = model.quantizer.from_codes(dummy_codes)
                embedding_dim = int(quantized_rep.shape[1])

        with torch.no_grad():
            one_second = np.zeros(sample_rate, dtype=np.float32)
            inputs = processor(raw_audio=one_second, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            frame_rate = float(model.encode(inputs["input_values"]).audio_codes.shape[-1])

        self._model = model
        self._processor = processor
        self._metadata = CodecMetadata(
            sample_rate=sample_rate,
            codebook_count=codebook_count,
            embedding_dim=int(embedding_dim),
            required_input_channels=1,
            frame_rate_hz=frame_rate,
        )
        logger.info(
            "DAC adapter ready (sample_rate=%s, codebooks=%s, embedding_dim=%s, frame_rate=%.2f)",
            self._metadata.sample_rate,
            self._metadata.codebook_count,
            self._metadata.embedding_dim,
            self._metadata.frame_rate_hz,
        )

    def metadata(self) -> CodecMetadata:
        self.ensure_loaded()
        assert self._metadata is not None
        return self._metadata

    def _prepare_mono(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        md = self.metadata()
        if samples.ndim != 2:
            raise ValueError(f"Expected [N, C] samples, got shape={samples.shape}")
        mono = np.mean(samples, axis=1)
        if sample_rate != md.sample_rate:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=md.sample_rate)
        mono = librosa.util.normalize(np.asarray(mono, dtype=np.float32))
        return mono

    @torch.inference_mode()
    def encode_samples(self, samples: np.ndarray, sample_rate: int) -> TokenBlock:
        self.ensure_loaded()
        assert self._model is not None and self._processor is not None
        mono = self._prepare_mono(samples, sample_rate)
        inputs = self._processor(raw_audio=mono, sampling_rate=self.metadata().sample_rate, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        audio_codes = self._model.encode(inputs["input_values"]).audio_codes.detach().cpu().contiguous()

        B = int(audio_codes.shape[0])
        K = int(audio_codes.shape[1])
        T = int(audio_codes.shape[2])
        tokens = audio_codes.reshape(-1).to(torch.int32).tolist()
        return TokenBlock(B=B, T=T, codebooks=K, tokens=tokens)

    @torch.inference_mode()
    def tokens_to_vector_row(self, block: TokenBlock, frame_index: int) -> np.ndarray:
        vectors = self.tokens_to_vector_rows(block, frame_index, 1)
        if vectors.shape[0] != 1:
            raise ValueError("Failed to extract DAC vector row")
        return vectors[0]

    @torch.inference_mode()
    def tokens_to_vector_rows(self, block: TokenBlock, start_frame: int, frame_count: int) -> np.ndarray:
        self.ensure_loaded()
        assert self._model is not None

        if start_frame < 0 or start_frame >= block.T:
            raise ValueError(f"start_frame {start_frame} out of range [0, {block.T})")
        if frame_count <= 0:
            return np.zeros((0, self.metadata().embedding_dim), dtype=np.float32)

        end_frame = min(block.T, start_frame + frame_count)
        if end_frame <= start_frame:
            return np.zeros((0, self.metadata().embedding_dim), dtype=np.float32)

        if block.B <= 0 or block.T <= 0 or block.codebooks <= 0:
            raise ValueError("Invalid token block dimensions")
        if len(block.tokens) != block.B * block.codebooks * block.T:
            raise ValueError("Token count is inconsistent with provided block dimensions")

        tokens_tensor = torch.as_tensor(block.tokens, dtype=torch.long).view(block.B, block.codebooks, block.T)
        frame_tokens = tokens_tensor[:, :, start_frame:end_frame].to(self._device)
        quantized_representation, _, _ = self._model.quantizer.from_codes(frame_tokens)
        vectors = np.asarray(quantized_representation.squeeze(0).transpose(0, 1).detach().cpu(), dtype=np.float32)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    @torch.inference_mode()
    def decode_tokens(self, block: TokenBlock) -> np.ndarray:
        self.ensure_loaded()
        assert self._model is not None

        if block.B <= 0 or block.T <= 0 or block.codebooks <= 0:
            raise ValueError("Invalid token block dimensions")
        if len(block.tokens) != block.B * block.codebooks * block.T:
            raise ValueError("Token count is inconsistent with provided block dimensions")

        tokens_tensor = torch.as_tensor(block.tokens, dtype=torch.long).view(block.B, block.codebooks, block.T)
        decoded_chunks: List[torch.Tensor] = []
        for start in range(0, block.T, DECODE_CHUNK_FRAMES):
            end = min(start + DECODE_CHUNK_FRAMES, block.T)
            chunk_codes = tokens_tensor[:, :, start:end].to(self._device)
            decoded = self._model.decode(audio_codes=chunk_codes)
            audio_values = decoded.audio_values if hasattr(decoded, "audio_values") else decoded
            decoded_chunks.append(audio_values.detach().cpu())

        if not decoded_chunks:
            raise ValueError("No audio could be reconstructed from tokens")

        audio_tensor = torch.cat(decoded_chunks, dim=-1)
        audio_np = np.asarray(audio_tensor.numpy(), dtype=np.float32)
        if audio_np.ndim == 3:
            if audio_np.shape[0] != block.B:
                raise ValueError("Decoded batch size mismatch")
            # [B, C, T] -> [T, C]
            samples = np.swapaxes(audio_np[0], 0, 1)
        elif audio_np.ndim == 2:
            if audio_np.shape[0] != block.B:
                raise ValueError("Decoded batch size mismatch")
            # Some DAC variants return [B, T] for mono.
            samples = audio_np[0][:, np.newaxis]
        elif audio_np.ndim == 1 and block.B == 1:
            samples = audio_np[:, np.newaxis]
        else:
            raise ValueError(f"Unexpected decoded tensor shape: {audio_np.shape}")

        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        return np.ascontiguousarray(samples, dtype=np.float32)


class SpectroStreamAdapter(CodecAdapter):
    codec_id = "spectrostream"

    def __init__(self):
        self._codec = None
        self._audio_mod = None
        self._metadata: Optional[CodecMetadata] = None
        self._rvq_codebooks: Optional[np.ndarray] = None

    def ensure_loaded(self) -> None:
        if self._codec is not None and self._audio_mod is not None and self._metadata is not None:
            return

        if FORCE_CPU_BACKENDS:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
            os.environ.setdefault("JAX_PLATFORMS", "cpu")

        try:
            from magenta_rt import audio as mrt_audio
            from magenta_rt import spectrostream
        except Exception as exc:
            raise RuntimeError(
                "SpectroStream dependencies are unavailable. Install magenta_rt and its runtime dependencies."
            ) from exc

        codec = spectrostream.SpectroStream()
        metadata = CodecMetadata(
            sample_rate=int(codec.sample_rate),
            codebook_count=int(codec.config.rvq_depth),
            embedding_dim=int(codec.config.embedding_dim),
            required_input_channels=int(codec.num_channels),
            frame_rate_hz=float(codec.frame_rate),
        )
        self._codec = codec
        self._audio_mod = mrt_audio
        self._metadata = metadata
        self._rvq_codebooks = np.asarray(codec.rvq_codebooks, dtype=np.float32)
        logger.info(
            "SpectroStream adapter ready (sample_rate=%s, channels=%s, codebooks=%s, embedding_dim=%s, frame_rate=%.2f)",
            metadata.sample_rate,
            metadata.required_input_channels,
            metadata.codebook_count,
            metadata.embedding_dim,
            metadata.frame_rate_hz,
        )

    def metadata(self) -> CodecMetadata:
        self.ensure_loaded()
        assert self._metadata is not None
        return self._metadata

    def _prepare_samples(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        md = self.metadata()
        if samples.ndim != 2:
            raise ValueError(f"Expected [N, C] samples, got shape={samples.shape}")

        if samples.shape[1] == 1 and md.required_input_channels == 2:
            samples = np.repeat(samples, 2, axis=1)
        elif samples.shape[1] >= md.required_input_channels:
            samples = samples[:, : md.required_input_channels]
        else:
            reps = [samples[:, min(i, samples.shape[1] - 1)] for i in range(md.required_input_channels)]
            samples = np.stack(reps, axis=1)

        if sample_rate != md.sample_rate:
            channels = []
            for ch in range(samples.shape[1]):
                channels.append(librosa.resample(samples[:, ch], orig_sr=sample_rate, target_sr=md.sample_rate))
            min_len = min(len(ch) for ch in channels)
            samples = np.stack([ch[:min_len] for ch in channels], axis=1)

        return np.ascontiguousarray(samples.astype(np.float32, copy=False))

    def warm_start(self) -> None:
        self.ensure_loaded()
        assert self._audio_mod is not None
        silence = np.zeros((self.metadata().sample_rate, self.metadata().required_input_channels), dtype=np.float32)
        waveform = self._audio_mod.Waveform(silence, self.metadata().sample_rate)
        self.decode_tokens(self.encode_samples(waveform.samples, waveform.sample_rate))

    def encode_samples(self, samples: np.ndarray, sample_rate: int) -> TokenBlock:
        self.ensure_loaded()
        assert self._codec is not None and self._audio_mod is not None
        prepared = self._prepare_samples(samples, sample_rate)
        waveform = self._audio_mod.Waveform(prepared, self.metadata().sample_rate)
        tokens_frame_major = np.asarray(self._codec.encode(waveform), dtype=np.int32)
        if tokens_frame_major.ndim != 2:
            raise ValueError(f"Unexpected SpectroStream token shape: {tokens_frame_major.shape}")

        T, K = int(tokens_frame_major.shape[0]), int(tokens_frame_major.shape[1])
        tokens = _frame_major_to_codebook_major(tokens_frame_major).astype(np.int32).tolist()
        return TokenBlock(B=1, T=T, codebooks=K, tokens=tokens)

    def tokens_to_vector_row(self, block: TokenBlock, frame_index: int) -> np.ndarray:
        vectors = self.tokens_to_vector_rows(block, frame_index, 1)
        if vectors.shape[0] != 1:
            raise ValueError("Failed to extract SpectroStream vector row")
        return vectors[0]

    def tokens_to_vector_rows(self, block: TokenBlock, start_frame: int, frame_count: int) -> np.ndarray:
        self.ensure_loaded()
        assert self._codec is not None and self._rvq_codebooks is not None

        if block.B != 1:
            raise ValueError("SpectroStream adapter currently supports batch size B=1")
        if start_frame < 0 or start_frame >= block.T:
            raise ValueError(f"start_frame {start_frame} out of range [0, {block.T})")
        if frame_count <= 0:
            return np.zeros((0, self.metadata().embedding_dim), dtype=np.float32)

        end_frame = min(block.T, start_frame + frame_count)
        if end_frame <= start_frame:
            return np.zeros((0, self.metadata().embedding_dim), dtype=np.float32)

        frame_major = _codebook_major_to_frame_major(block.tokens, block.codebooks, block.T)
        codebooks = self._rvq_codebooks
        depth = min(frame_major.shape[1], codebooks.shape[0])
        if depth <= 0:
            return np.zeros((end_frame - start_frame, self.metadata().embedding_dim), dtype=np.float32)

        token_ids = np.asarray(frame_major[start_frame:end_frame, :depth], dtype=np.int64)
        if np.any(token_ids < 0) or np.any(token_ids >= codebooks.shape[1]):
            raise ValueError("SpectroStream token id out of range for RVQ codebooks")

        vectors = codebooks[np.arange(depth)[np.newaxis, :], token_ids].sum(axis=1)
        return np.ascontiguousarray(vectors.astype(np.float32, copy=False))

    def decode_tokens(self, block: TokenBlock) -> np.ndarray:
        self.ensure_loaded()
        assert self._codec is not None

        if block.B != 1:
            raise ValueError("SpectroStream adapter currently supports batch size B=1")

        frame_major = _codebook_major_to_frame_major(block.tokens, block.codebooks, block.T)
        waveform = self._codec.decode(frame_major.astype(np.int32))
        samples = np.asarray(waveform.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        return np.ascontiguousarray(samples, dtype=np.float32)


def _detect_supported_codecs() -> List[str]:
    codecs = ["dac"]
    try:
        from magenta_rt import spectrostream  # pylint: disable=unused-import,g-import-not-at-top
        codecs.append("spectrostream")
    except Exception:
        pass
    return codecs


class BridgeRuntime:
    def __init__(self, device: torch.device):
        self.device = device
        self.supported_codecs = _detect_supported_codecs()
        self._adapters: Dict[str, CodecAdapter] = {}
        self._active_codec = BRIDGE_CODEC_DEFAULT if BRIDGE_CODEC_DEFAULT in self.supported_codecs else "dac"
        self._sessions: Dict[str, dict] = {}

    @property
    def active_codec(self) -> str:
        return self._active_codec

    def _build_adapter(self, codec_id: str) -> CodecAdapter:
        if codec_id == "dac":
            return DacAdapter(self.device)
        if codec_id == "spectrostream":
            return SpectroStreamAdapter()
        raise ValueError(f"Unsupported codec '{codec_id}'")

    def get_adapter(self, codec_id: Optional[str] = None) -> CodecAdapter:
        codec = (codec_id or self._active_codec).strip().lower()
        if codec not in self.supported_codecs:
            raise ValueError(f"Codec '{codec}' is not supported by this bridge")
        if codec not in self._adapters:
            self._adapters[codec] = self._build_adapter(codec)
        adapter = self._adapters[codec]
        adapter.ensure_loaded()
        return adapter

    def set_codec(self, codec_id: str) -> CodecAdapter:
        codec = codec_id.strip().lower()
        if codec not in self.supported_codecs:
            raise ValueError(f"Unsupported codec '{codec}'. Supported: {self.supported_codecs}")
        adapter = self.get_adapter(codec)
        self._active_codec = codec
        return adapter

    def capabilities(self) -> CapabilitiesResponse:
        adapter = self.get_adapter(self._active_codec)
        md = adapter.metadata()
        return CapabilitiesResponse(
            ok=True,
            supported_codecs=list(self.supported_codecs),
            active_codec=self._active_codec,
            sample_rate=md.sample_rate,
            codebook_count=md.codebook_count,
            embedding_dim=md.embedding_dim,
            required_input_channels=md.required_input_channels,
            frame_rate_hz=md.frame_rate_hz,
            token_layout=TOKEN_LAYOUT_CODEBOOK_MAJOR,
            supports_pcm_endpoints=True,
        )

    def open_session(self) -> str:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = {"chunks": [], "sample_rate": 0, "channels": 0}
        return session_id

    def push_session_chunk(self, session_id: str, samples: np.ndarray, sample_rate: int) -> int:
        if session_id not in self._sessions:
            raise KeyError("Unknown session id")
        state = self._sessions[session_id]
        state["chunks"].append(np.ascontiguousarray(samples.astype(np.float32, copy=False)))
        state["sample_rate"] = int(sample_rate)
        state["channels"] = int(samples.shape[1])
        return int(samples.shape[0])

    def pull_session_audio(self, session_id: str) -> tuple[np.ndarray, int]:
        if session_id not in self._sessions:
            raise KeyError("Unknown session id")
        state = self._sessions[session_id]
        if not state["chunks"]:
            return np.zeros((0, 1), dtype=np.float32), 0
        merged = np.concatenate(state["chunks"], axis=0)
        state["chunks"].clear()
        return merged, int(state["sample_rate"])

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _stream_audio_windows(path: str, window_seconds: float):
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    with sf.SoundFile(str(path_obj)) as source:
        src_sr = source.samplerate
        window_frames = max(int(window_seconds * src_sr), 1)
        while True:
            frames = source.read(window_frames, dtype="float32", always_2d=True)
            if frames.size == 0:
                break
            yield np.ascontiguousarray(frames, dtype=np.float32), int(src_sr)


def _encode_path_with_adapter(adapter: CodecAdapter, audio_path: str) -> TokenBlock:
    chunk_tokens: List[np.ndarray] = []
    chunk_codebooks = None
    batch_size = None
    for chunk_idx, (chunk, src_sr) in enumerate(_stream_audio_windows(audio_path, ENCODE_WINDOW_SECONDS)):
        block = adapter.encode_samples(chunk, src_sr)
        if block.T <= 0:
            continue
        if batch_size is None:
            batch_size = block.B
        elif batch_size != block.B:
            raise RuntimeError("Inconsistent batch size encountered during encoding")

        if chunk_codebooks is None:
            chunk_codebooks = block.codebooks
        elif chunk_codebooks != block.codebooks:
            raise RuntimeError("Inconsistent codebook count encountered during encoding")

        codebook_major = np.asarray(block.tokens, dtype=np.int32).reshape(block.codebooks, block.T)
        chunk_tokens.append(codebook_major)
        logger.info("encode_path: chunk=%s frames=%s codebooks=%s", chunk_idx, block.T, block.codebooks)

    if not chunk_tokens or chunk_codebooks is None or batch_size is None:
        raise ValueError("Audio file is empty or could not be processed")

    joined = np.concatenate(chunk_tokens, axis=1)
    return TokenBlock(
        B=batch_size,
        T=int(joined.shape[1]),
        codebooks=int(joined.shape[0]),
        tokens=joined.reshape(-1).astype(np.int32).tolist(),
    )


def _token_block_from_request(
    B: int,
    T: int,
    tokens: Sequence[int],
    *,
    codebooks: Optional[int],
    token_layout: Optional[str],
    adapter: CodecAdapter,
) -> TokenBlock:
    if B <= 0 or T <= 0:
        raise ValueError("Batch size and frame count must be positive")

    codebooks_final = int(codebooks) if codebooks is not None else 0
    if codebooks_final <= 0:
        if len(tokens) % (B * T) != 0:
            raise ValueError("Token count is inconsistent with provided batch/frame sizes")
        codebooks_final = len(tokens) // (B * T)

    if codebooks_final <= 0:
        raise ValueError("Token payload appears to be empty")

    expected = B * codebooks_final * T
    if len(tokens) != expected:
        raise ValueError(f"Expected {expected} tokens but got {len(tokens)}")

    layout = _normalize_token_layout(token_layout)
    tokens_list = list(int(t) for t in tokens)
    if layout == TOKEN_LAYOUT_FRAME_MAJOR:
        if B != 1:
            raise ValueError("Frame-major layout is only supported for B=1 requests")
        frame_major = np.asarray(tokens_list, dtype=np.int32).reshape(T, codebooks_final)
        tokens_list = _frame_major_to_codebook_major(frame_major).astype(np.int32).tolist()

    return TokenBlock(B=B, T=T, codebooks=codebooks_final, tokens=tokens_list)


def _can_use_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception as exc:
        logger.warning("CUDA reported available but is unusable (%s); falling back to CPU.", exc)
        return False


def _select_torch_device() -> torch.device:
    preference = os.getenv("BRIDGE_DEVICE", "auto").strip().lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if _can_use_cuda():
            return torch.device("cuda")
        raise RuntimeError("BRIDGE_DEVICE=cuda requested but CUDA is unavailable/unusable")
    return torch.device("cuda" if _can_use_cuda() else "cpu")


device = _select_torch_device()
logger.info("Bridge compute device: %s", device)
FORCE_CPU_BACKENDS = device.type != "cuda"
runtime = BridgeRuntime(device=device)

app = FastAPI(title="Neural Morphing Bridge", version="2.0.0")


@app.on_event("startup")
async def startup_event():
    if BRIDGE_WARM_START:
        logger.info("Bridge warm start enabled for codec '%s'", runtime.active_codec)
        adapter = runtime.get_adapter(runtime.active_codec)
        adapter.warm_start()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        adapter = runtime.get_adapter(runtime.active_codec)
        md = adapter.metadata()
        return HealthResponse(
            ok=True,
            sample_rate=md.sample_rate,
            codebook_count=md.codebook_count,
            embedding_dim=md.embedding_dim,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities_endpoint():
    try:
        return runtime.capabilities()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/codec", response_model=CodecResponse)
async def codec_endpoint(request: CodecRequest):
    try:
        runtime.set_codec(request.codec)
        caps = runtime.capabilities()
        return CodecResponse(**caps.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/encode", response_model=EncodeResponse)
async def encode_endpoint(request: EncodeRequest):
    try:
        audio_path = Path(request.path).expanduser()
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {request.path}")

        adapter = runtime.get_adapter(runtime.active_codec)
        block = _encode_path_with_adapter(adapter, str(audio_path))
        logger.info(
            "/encode done codec=%s path=%s frames=%s codebooks=%s tokens=%s",
            runtime.active_codec,
            audio_path,
            block.T,
            block.codebooks,
            len(block.tokens),
        )
        return EncodeResponse(
            B=block.B,
            T=block.T,
            codebooks=block.codebooks,
            tokens=block.tokens,
            sample_rate=adapter.metadata().sample_rate,
            token_layout=TOKEN_LAYOUT_CODEBOOK_MAJOR,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Encode error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/encode_pcm", response_model=EncodePCMResponse)
async def encode_pcm_endpoint(request: EncodePCMRequest):
    try:
        samples = _decode_pcm_payload(
            request.pcm_b64,
            channels=request.channels,
            num_samples=request.num_samples,
            dtype=request.dtype,
            pcm_layout=request.pcm_layout,
        )
        adapter = runtime.get_adapter(runtime.active_codec)
        block = adapter.encode_samples(samples, request.sample_rate)
        return EncodePCMResponse(
            B=block.B,
            T=block.T,
            codebooks=block.codebooks,
            tokens=block.tokens,
            sample_rate=adapter.metadata().sample_rate,
            token_layout=TOKEN_LAYOUT_CODEBOOK_MAJOR,
        )
    except Exception as exc:
        logger.exception("Encode PCM error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/tokens_to_vectors", response_model=TokensToVectorsResponse)
async def tokens_to_vectors_endpoint(request: TokensToVectorsRequest):
    try:
        adapter = runtime.get_adapter(runtime.active_codec)
        block = _token_block_from_request(
            request.B,
            request.T,
            request.tokens,
            codebooks=request.codebooks,
            token_layout=request.token_layout,
            adapter=adapter,
        )
        vector = adapter.tokens_to_vector_row(block, request.frame_index)
        return TokensToVectorsResponse(D=int(vector.shape[0]), vector=vector.astype(np.float32).tolist())
    except Exception as exc:
        logger.exception("Tokens to vectors error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/tokens_to_vectors_batch", response_model=TokensToVectorsBatchResponse)
async def tokens_to_vectors_batch_endpoint(request: TokensToVectorsBatchRequest):
    try:
        adapter = runtime.get_adapter(runtime.active_codec)
        block = _token_block_from_request(
            request.B,
            request.T,
            request.tokens,
            codebooks=request.codebooks,
            token_layout=request.token_layout,
            adapter=adapter,
        )

        if request.start_frame < 0 or request.start_frame >= block.T:
            raise ValueError(f"start_frame {request.start_frame} out of range [0, {block.T})")

        requested_count = request.frame_count if request.frame_count > 0 else (block.T - request.start_frame)
        end_frame = min(block.T, request.start_frame + requested_count)
        matrix = adapter.tokens_to_vector_rows(block, request.start_frame, end_frame - request.start_frame)
        if matrix.ndim != 2:
            raise ValueError(f"Expected vector matrix [N, D], got shape={matrix.shape}")
        vectors = matrix.astype(np.float32).tolist()
        expected_dim = int(matrix.shape[1]) if matrix.shape[0] > 0 else 0

        return TokensToVectorsBatchResponse(
            D=expected_dim or 0,
            count=len(vectors),
            vectors=vectors,
        )
    except Exception as exc:
        logger.exception("Tokens to vectors batch error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/decode", response_model=DecodeResponse)
async def decode_endpoint(request: DecodeRequest):
    try:
        adapter = runtime.get_adapter(runtime.active_codec)
        block = _token_block_from_request(
            request.B,
            request.T,
            request.tokens,
            codebooks=request.codebooks,
            token_layout=request.token_layout,
            adapter=adapter,
        )
        samples = adapter.decode_tokens(block)

        with io.BytesIO() as buffer:
            sf.write(buffer, samples, adapter.metadata().sample_rate, format="WAV")
            buffer.seek(0)
            wav_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return DecodeResponse(wav_b64=wav_b64)
    except Exception as exc:
        logger.exception("Decode error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/decode_pcm", response_model=DecodePCMResponse)
async def decode_pcm_endpoint(request: DecodePCMRequest):
    try:
        adapter = runtime.get_adapter(runtime.active_codec)
        block = _token_block_from_request(
            request.B,
            request.T,
            request.tokens,
            codebooks=request.codebooks,
            token_layout=request.token_layout,
            adapter=adapter,
        )
        samples = adapter.decode_tokens(block)
        pcm_b64, num_samples, channels = _encode_pcm_payload(samples)
        return DecodePCMResponse(
            sample_rate=adapter.metadata().sample_rate,
            channels=channels,
            num_samples=num_samples,
            dtype=PCM_DTYPE_FLOAT32,
            pcm_layout=PCM_LAYOUT_INTERLEAVED,
            pcm_b64=pcm_b64,
        )
    except Exception as exc:
        logger.exception("Decode PCM error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/session/open", response_model=SessionOpenResponse)
async def session_open_endpoint():
    if not BRIDGE_ENABLE_STREAM_SESSIONS:
        raise HTTPException(status_code=404, detail="Streaming sessions are disabled")
    session_id = runtime.open_session()
    return SessionOpenResponse(ok=True, session_id=session_id)


@app.post("/session/push", response_model=SessionPushResponse)
async def session_push_endpoint(request: SessionPushRequest):
    if not BRIDGE_ENABLE_STREAM_SESSIONS:
        raise HTTPException(status_code=404, detail="Streaming sessions are disabled")
    try:
        samples = _decode_pcm_payload(
            request.pcm_b64,
            channels=request.channels,
            num_samples=request.num_samples,
            dtype=request.dtype,
            pcm_layout=request.pcm_layout,
        )
        accepted = runtime.push_session_chunk(request.session_id, samples, request.sample_rate)
        return SessionPushResponse(ok=True, accepted_samples=accepted)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/session/pull", response_model=SessionPullResponse)
async def session_pull_endpoint(request: SessionPullRequest):
    if not BRIDGE_ENABLE_STREAM_SESSIONS:
        raise HTTPException(status_code=404, detail="Streaming sessions are disabled")
    try:
        samples, sample_rate = runtime.pull_session_audio(request.session_id)
        pcm_b64, num_samples, channels = _encode_pcm_payload(samples)
        return SessionPullResponse(
            ok=True,
            sample_rate=sample_rate,
            channels=channels,
            num_samples=num_samples,
            dtype=PCM_DTYPE_FLOAT32,
            pcm_layout=PCM_LAYOUT_INTERLEAVED,
            pcm_b64=pcm_b64,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/session/close", response_model=SessionCloseResponse)
async def session_close_endpoint(request: SessionCloseRequest):
    if not BRIDGE_ENABLE_STREAM_SESSIONS:
        raise HTTPException(status_code=404, detail="Streaming sessions are disabled")
    runtime.close_session(request.session_id)
    return SessionCloseResponse(ok=True)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "localhost")

    logger.info("Starting Neural Morphing Bridge Server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
