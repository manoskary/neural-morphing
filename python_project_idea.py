from __future__ import annotations

"""
RVQ-Aware Latent Granular Morphing with DAC
===========================================
Builds a palette of DAC code grains, matches target grains with
RVQ-aware descriptors, enforces temporal continuity, and applies
Top-K mixing for finer layers.
"""

import contextlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
EXAMPLES_DIR = APP_DIR / "examples"
HERO_IMAGE = ASSETS_DIR / "neural_morphing_title.png"
HERO_IMAGE_URL = "/gradio_api/file=assets/neural_morphing_title.png"

DEMO_EXAMPLE_PACKS = [
    {
        "name": "Bass + Percussion -> Rhythmic Loop",
        "sources": ("demo_bass_motif.wav", "demo_percussion_texture.wav"),
        "target": "demo_rhythmic_loop.wav",
    },
    {
        "name": "Synth + Loop -> Bass Motif",
        "sources": ("demo_synth_pulse.wav", "demo_rhythmic_loop.wav"),
        "target": "demo_bass_motif.wav",
    },
]

if HERO_IMAGE.exists():
    gr.set_static_paths([ASSETS_DIR])

APP_CSS = f"""
#neural-morphing-app {{
    min-height: 100vh;
    background:
        radial-gradient(circle at 12% 8%, rgba(255, 112, 67, 0.26), transparent 28rem),
        radial-gradient(circle at 86% 16%, rgba(236, 64, 122, 0.26), transparent 30rem),
        linear-gradient(180deg, #07101f 0%, #0b2233 48%, #071522 100%);
    color: #f7fbff;
}}

#neural-morphing-app .gradio-container {{
    max-width: 1220px !important;
    margin: 0 auto !important;
    padding: 22px !important;
    background: transparent !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

#neural-morphing-app .nm-hero {{
    position: relative;
    min-height: 0;
    aspect-ratio: 3200 / 711;
    margin-bottom: 18px;
    overflow: hidden;
    border: 1px solid rgba(61, 241, 235, 0.55);
    border-radius: 8px;
    background: rgba(2, 8, 18, 0.42);
    box-shadow:
        0 0 0 1px rgba(255, 64, 129, 0.20),
        0 24px 70px rgba(0, 0, 0, 0.42),
        inset 0 -80px 120px rgba(3, 18, 29, 0.28);
}}

#neural-morphing-app .nm-hero-image {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center top;
}}

#neural-morphing-app .nm-hero::after {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 6px 6px;
    mix-blend-mode: screen;
    opacity: 0.28;
}}

#neural-morphing-app .nm-hero-chrome {{
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}}

#neural-morphing-app .nm-chip {{
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 10px;
    border: 1px solid rgba(63, 241, 238, 0.62);
    border-radius: 999px;
    background: rgba(4, 18, 30, 0.70);
    color: #d9ffff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
    box-shadow: 0 0 18px rgba(19, 235, 226, 0.20);
    backdrop-filter: blur(8px);
}}

#neural-morphing-app .nm-chip-hot {{
    border-color: rgba(255, 78, 152, 0.70);
    color: #ffe6f1;
    box-shadow: 0 0 18px rgba(255, 78, 152, 0.26);
}}

#neural-morphing-app .nm-main-grid {{
    gap: 18px !important;
    align-items: stretch;
}}

#neural-morphing-app .nm-panel {{
    padding: 16px !important;
    border: 1px solid rgba(61, 241, 235, 0.24);
    border-radius: 8px;
    background:
        linear-gradient(180deg, rgba(11, 29, 47, 0.88), rgba(6, 16, 29, 0.92)),
        radial-gradient(circle at 95% 0%, rgba(255, 74, 149, 0.16), transparent 18rem);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
}}

#neural-morphing-app .nm-panel h3 {{
    margin: 0 0 12px !important;
    color: #efffff;
    font-size: 15px;
    line-height: 1.2;
    font-weight: 850;
    letter-spacing: 0;
    text-transform: uppercase;
}}

#neural-morphing-app .block,
#neural-morphing-app .form,
#neural-morphing-app .wrap,
#neural-morphing-app .gr-box {{
    border-color: rgba(67, 241, 238, 0.18) !important;
    border-radius: 8px !important;
    background: rgba(4, 13, 24, 0.46) !important;
}}

#neural-morphing-app label,
#neural-morphing-app .label-wrap,
#neural-morphing-app .svelte-1gfkn6j {{
    color: #caeff5 !important;
}}

#neural-morphing-app input,
#neural-morphing-app textarea,
#neural-morphing-app select {{
    color: #f5ffff !important;
}}

.nm-primary,
.nm-secondary,
#neural-morphing-app .nm-primary button,
#neural-morphing-app .nm-secondary button {{
    min-height: 44px;
    border: 0 !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 850 !important;
    letter-spacing: 0;
    text-transform: uppercase;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.26);
}}

.nm-primary,
#neural-morphing-app .nm-primary button {{
    background: linear-gradient(90deg, #ff6a2a 0%, #ff2f91 55%, #8f43ff 100%) !important;
}}

.nm-secondary,
#neural-morphing-app .nm-secondary button {{
    background: linear-gradient(90deg, #05c9d6 0%, #1c7fff 100%) !important;
}}

.nm-primary:hover,
.nm-secondary:hover,
#neural-morphing-app .nm-primary button:hover,
#neural-morphing-app .nm-secondary button:hover {{
    filter: brightness(1.08);
    transform: translateY(-1px);
}}

#neural-morphing-app .nm-audio-grid {{
    gap: 10px !important;
}}

#neural-morphing-app .nm-example-row {{
    align-items: end;
    gap: 10px !important;
}}

#neural-morphing-app audio {{
    filter: saturate(1.18);
}}

@media (max-width: 760px) {{
    #neural-morphing-app .gradio-container {{
        padding: 12px !important;
    }}

    #neural-morphing-app .nm-hero {{
        min-height: 0;
        background-position: center top;
    }}

    #neural-morphing-app .nm-panel {{
        padding: 12px !important;
    }}
}}

body {{
    background:
        radial-gradient(circle at 12% 8%, rgba(255, 112, 67, 0.26), transparent 28rem),
        radial-gradient(circle at 86% 16%, rgba(236, 64, 122, 0.26), transparent 30rem),
        linear-gradient(180deg, #07101f 0%, #0b2233 48%, #071522 100%) !important;
}}

.gradio-container {{
    max-width: 1220px !important;
    margin: 0 auto !important;
    padding: 22px !important;
    background: transparent !important;
}}

.nm-hero {{
    position: relative;
    min-height: 0;
    aspect-ratio: 3200 / 711;
    margin-bottom: 18px;
    overflow: hidden;
    border: 1px solid rgba(61, 241, 235, 0.55);
    border-radius: 8px;
    background: rgba(2, 8, 18, 0.42);
    box-shadow:
        0 0 0 1px rgba(255, 64, 129, 0.20),
        0 24px 70px rgba(0, 0, 0, 0.42),
        inset 0 -80px 120px rgba(3, 18, 29, 0.28);
}}

.nm-hero-image {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center top;
}}

.nm-hero::after {{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 6px 6px;
    mix-blend-mode: screen;
    opacity: 0.28;
}}

.nm-hero-chrome {{
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}}

.nm-chip {{
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 10px;
    border: 1px solid rgba(63, 241, 238, 0.62);
    border-radius: 999px;
    background: rgba(4, 18, 30, 0.70);
    color: #d9ffff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
    box-shadow: 0 0 18px rgba(19, 235, 226, 0.20);
    backdrop-filter: blur(8px);
}}

.nm-chip-hot {{
    border-color: rgba(255, 78, 152, 0.70);
    color: #ffe6f1;
    box-shadow: 0 0 18px rgba(255, 78, 152, 0.26);
}}

.nm-main-grid {{
    gap: 18px !important;
    align-items: stretch;
}}

.nm-panel {{
    padding: 16px !important;
    border: 1px solid rgba(61, 241, 235, 0.24);
    border-radius: 8px;
    background:
        linear-gradient(180deg, rgba(11, 29, 47, 0.88), rgba(6, 16, 29, 0.92)),
        radial-gradient(circle at 95% 0%, rgba(255, 74, 149, 0.16), transparent 18rem);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
}}

.nm-panel h3 {{
    margin: 0 0 12px !important;
    color: #efffff;
    font-size: 15px;
    line-height: 1.2;
    font-weight: 850;
    letter-spacing: 0;
    text-transform: uppercase;
}}

.nm-primary,
.nm-secondary,
.nm-primary button,
.nm-secondary button {{
    min-height: 44px;
    border: 0 !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 850 !important;
    letter-spacing: 0;
    text-transform: uppercase;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.26);
}}

.nm-primary,
.nm-primary button {{
    background: linear-gradient(90deg, #ff6a2a 0%, #ff2f91 55%, #8f43ff 100%) !important;
}}

.nm-secondary,
.nm-secondary button {{
    background: linear-gradient(90deg, #05c9d6 0%, #1c7fff 100%) !important;
}}

.nm-primary:hover,
.nm-secondary:hover,
.nm-primary button:hover,
.nm-secondary button:hover {{
    filter: brightness(1.08);
    transform: translateY(-1px);
}}

@media (max-width: 760px) {{
    .gradio-container {{
        padding: 12px !important;
    }}

    .nm-hero {{
        min-height: 0;
    }}

    .nm-panel {{
        padding: 12px !important;
    }}
}}
"""

HERO_HTML = f"""
<section class="nm-hero" aria-label="Neural Morphing">
  <img class="nm-hero-image" src="{HERO_IMAGE_URL}" alt="Neural Morphing" />
</section>
"""

librosa = None
np = None
sf = None
torch = None
F = None
tqdm = None
AutoProcessor = None
DacModel = None


def _ensure_runtime_dependencies() -> None:
    """Import ML/audio dependencies only after the web server is ready to launch."""
    global AutoProcessor, DacModel, F, librosa, np, sf, torch, tqdm

    if torch is not None:
        return

    import librosa as _librosa
    import numpy as _np
    import soundfile as _sf
    import torch as _torch
    import torch.nn.functional as _F
    from tqdm import tqdm as _tqdm
    from transformers import AutoProcessor as _AutoProcessor
    from transformers import DacModel as _DacModel

    librosa = _librosa
    np = _np
    sf = _sf
    torch = _torch
    F = _F
    tqdm = _tqdm
    AutoProcessor = _AutoProcessor
    DacModel = _DacModel


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _available_demo_examples():
    examples = []
    for pack in DEMO_EXAMPLE_PACKS:
        sources = [EXAMPLES_DIR / filename for filename in pack["sources"]]
        target = EXAMPLES_DIR / pack["target"]
        if all(path.exists() for path in sources) and target.exists():
            examples.append((pack["name"], [str(path) for path in sources], str(target)))
    return examples


def _demo_example_names():
    return [name for name, _, _ in _available_demo_examples()]


class LatentGranularSynthesis:
    MATCH_MODES = {"greedy", "greedy_smooth", "beam", "viterbi"}
    SWAP_ALIASES = {
        "full_layer": "full_layer_gated",
        "rvq_group": "rvq_group_current",
    }
    SWAP_MODES = {
        "identity",
        "palette_only",
        "coarse_gated",
        "coarse_forced",
        "middle_only",
        "fine_only",
        "middle_fine",
        "rvq_group_current",
        "full_layer_gated",
        "full_layer_forced",
    }

    DAC_DEFAULTS = {
        "temperature": 0.47,
        "threshold": 0.99,
        "continuity": 0.93,
        "rvq_focus": 0.30,
        "unit": 7,
        "stride": 2,
        "top_k": 7,
    }

    SPECTROSTREAM_DEFAULTS = {
        "temperature": 0.4315336855083648,
        "threshold": 0.24313963041725395,
        "continuity": 0.7887727172362835,
        "rvq_focus": 0.3460889655971231,
        "unit": 2,
        "stride": 2,
        "top_k": 8,
    }

    @staticmethod
    def _select_device(device):
        if device is not None:
            return torch.device(device)

        if not torch.cuda.is_available():
            return torch.device("cpu")

        # Some systems expose CUDA but have an unsupported GPU architecture for the installed torch build.
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"CUDA reported available but is unusable ({exc}); falling back to CPU.")
            return torch.device("cpu")

    def __init__(
        self,
        model_name="descript/dac_44khz",
        device=None,
        chunk_duration_s=8.0,
        match_batch=2048,
    ):
        """Initialize multi-codec morphing stack (DAC + optional SpectroStream)."""
        _ensure_runtime_dependencies()

        self.device = self._select_device(device)
        self.compute_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.chunk_duration_s = max(chunk_duration_s, 1.0)
        self.match_batch = max(int(match_batch), 1)
        # Keep chunk loudness by default to preserve temporal envelope fidelity.
        self.normalize_input_chunks = _env_flag("NEURAL_MORPHING_NORMALIZE_INPUT_CHUNKS", "0")
        # Keep a tiny fixed headroom to avoid hard clipping in int16 exports.
        self.output_peak_target = float(
            np.clip(float(os.getenv("NEURAL_MORPHING_OUTPUT_PEAK_TARGET", "0.995")), 0.80, 0.999)
        )
        self.model_name = model_name

        self.model = None
        self.processor = None
        self.sample_rate = 44100
        self.required_input_channels = 1
        self.codec_id = "dac"
        self.supported_codecs = ["dac"]

        self.spectro_codec = None
        self.spectro_audio_mod = None
        self.spectro_codebooks = None

        try:
            from magenta_rt import audio as _mrt_audio  # noqa: F401
            from magenta_rt import spectrostream as _mrt_spectrostream  # noqa: F401
            self.supported_codecs.append("spectrostream")
        except Exception:
            pass

        self._load_dac()

        self.unit = 1
        self.stride = 1
        self.temperature = 0.47
        self.threshold = 0.99
        self.continuity = 0.93
        self.rvq_focus = 0.30
        self.top_k = 1
        self.candidate_count = 96
        self.beam_width = 12
        self.match_mode = "beam"
        self.swap_mode = "palette_only"
        self._apply_codec_defaults("dac")
        self.last_timings = {"encode_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0}
        self.last_sequence_diagnostics = {}

        self.files = None
        self.last_aug = False
        self.pitch_aug = [-5, -2, 2, 5]
        self.vol_aug = [0.3, 0.7]

        self.codebook_embeddings = None
        self.rvq_groups = None
        self.palette_codes = None
        self.palette_desc_groups = None
        self.palette_desc_full = None
        self.palette_file_ids = None
        self.palette_frame_indices = None
        self.prev_best_index = None

        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Compute device: {self.device}")
        print(f"Supported codecs: {', '.join(self.supported_codecs)}")

    def _codec_defaults(self, codec_id: str | None = None) -> dict:
        codec = (codec_id or self.codec_id or "dac").strip().lower()
        if codec == "spectrostream":
            return dict(self.SPECTROSTREAM_DEFAULTS)
        return dict(self.DAC_DEFAULTS)

    def _apply_codec_defaults(self, codec_id: str | None = None) -> None:
        defaults = self._codec_defaults(codec_id)
        self.temperature = float(defaults["temperature"])
        self.threshold = float(defaults["threshold"])
        self.continuity = float(defaults["continuity"])
        self.rvq_focus = float(defaults["rvq_focus"])
        self.unit = max(1, int(defaults["unit"]))
        self.stride = max(1, int(defaults["stride"]))
        self.top_k = max(1, int(defaults["top_k"]))

    def runtime_params(self) -> dict:
        return {
            "temperature": float(self.temperature),
            "threshold": float(self.threshold),
            "continuity": float(self.continuity),
            "rvq_focus": float(self.rvq_focus),
            "unit": int(self.unit),
            "stride": int(self.stride),
            "top_k": int(self.top_k),
        }

    def _load_dac(self):
        if self.model is not None and self.processor is not None:
            self.codec_id = "dac"
            self.sample_rate = int(self.processor.sampling_rate)
            self.required_input_channels = 1
            self.codebook_embeddings = None
            print("Using codec: dac")
            return

        self.model = DacModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.codec_id = "dac"
        self.sample_rate = int(self.processor.sampling_rate)
        self.required_input_channels = 1
        self.codebook_embeddings = None
        print(f"Using DAC model: {self.model_name}")
        print("Using codec: dac")

    def _load_spectrostream(self):
        if self.spectro_codec is not None and self.spectro_audio_mod is not None:
            self.codec_id = "spectrostream"
            self.sample_rate = int(self.spectro_codec.sample_rate)
            self.required_input_channels = int(self.spectro_codec.num_channels)
            if self.spectro_codebooks is not None:
                self.codebook_embeddings = [
                    torch.from_numpy(self.spectro_codebooks[q]).to(torch.float32)
                    for q in range(self.spectro_codebooks.shape[0])
                ]
            print("Using codec: spectrostream")
            return

        # SpectroStream currently relies on TensorFlow/JAX internals; on low-VRAM GPUs
        # this frequently fails with libdevice/JIT/OOM errors. Keep CPU as robust default.
        if not _env_flag("NEURAL_MORPHING_SPECTROSTREAM_USE_GPU", "0"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        from magenta_rt import audio as mrt_audio
        from magenta_rt import spectrostream

        self.spectro_audio_mod = mrt_audio
        self.spectro_codec = spectrostream.SpectroStream()
        self.spectro_codebooks = np.asarray(self.spectro_codec.rvq_codebooks, dtype=np.float32)
        self.codec_id = "spectrostream"
        self.sample_rate = int(self.spectro_codec.sample_rate)
        self.required_input_channels = int(self.spectro_codec.num_channels)
        self.codebook_embeddings = [
            torch.from_numpy(self.spectro_codebooks[q]).to(torch.float32)
            for q in range(self.spectro_codebooks.shape[0])
        ]
        print("Using codec: spectrostream")

    def set_codec(self, codec_id: str):
        codec = (codec_id or "dac").strip().lower()
        if codec not in self.supported_codecs:
            raise ValueError(f"Unsupported codec '{codec}'. Supported: {self.supported_codecs}")

        if codec == "dac":
            self._load_dac()
        elif codec == "spectrostream":
            self._load_spectrostream()
        else:
            raise ValueError(f"Unsupported codec '{codec}'")

        # Palette/index tensors are codec-specific and must be rebuilt after switching.
        self.files = None
        self.prev_best_index = None
        self.rvq_groups = None
        self.palette_codes = None
        self.palette_desc_groups = None
        self.palette_desc_full = None
        self.palette_file_ids = None
        self.palette_frame_indices = None

        self._apply_codec_defaults(codec)
        defaults = self.runtime_params()
        return (
            f"Codec switched to '{self.codec_id}'. Rebuild the source palette. "
            f"Defaults applied: temp={defaults['temperature']:.4f}, thr={defaults['threshold']:.4f}, "
            f"cont={defaults['continuity']:.4f}, rvq={defaults['rvq_focus']:.4f}, "
            f"unit={defaults['unit']}, stride={defaults['stride']}, top_k={defaults['top_k']}."
        )

    @staticmethod
    def _resolve_file_entry(entry):
        """Best-effort resolve of Gradio file payloads into filesystem paths."""
        if entry is None:
            return None

        if isinstance(entry, (str, os.PathLike)):
            return Path(entry).expanduser()

        if isinstance(entry, dict):
            candidate = entry.get("path") or entry.get("name")
            if candidate:
                return Path(candidate).expanduser()

        for attr in ("name", "path"):
            candidate = getattr(entry, attr, None)
            if candidate:
                return Path(candidate).expanduser()

        return None

    def _materialize_files(self, files):
        if files is None:
            return []

        candidates = files if isinstance(files, (list, tuple)) else [files]

        resolved = []
        for entry in candidates:
            path = self._resolve_file_entry(entry)
            if path is None:
                continue

            if not path.exists():
                print(f"Skipping missing file: {path}")
                continue

            resolved.append(str(path))

        return resolved

    def encode(self, audio_array):
        """Encode audio using the active codec."""
        if audio_array is None:
            return None, None

        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.detach().cpu().numpy()

        audio_array = np.asarray(audio_array, dtype=np.float32)

        if audio_array.size == 0:
            return None, None

        if self.codec_id == "dac":
            if audio_array.ndim > 1:
                audio_array = librosa.to_mono(audio_array)

            inputs = self.processor(
                raw_audio=audio_array,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                with self._autocast_context():
                    encoder_outputs = self.model.encode(inputs["input_values"])

            audio_codes = encoder_outputs.audio_codes.detach().to("cpu")
            quantized_representation = encoder_outputs.quantized_representation.detach().to("cpu", dtype=torch.float32)
            return audio_codes, quantized_representation

        if self.codec_id == "spectrostream":
            if self.spectro_codec is None or self.spectro_audio_mod is None:
                self._load_spectrostream()

            if audio_array.ndim == 1:
                samples = audio_array[:, np.newaxis]
            else:
                samples = np.asarray(audio_array, dtype=np.float32)
                if samples.shape[0] < samples.shape[1]:
                    samples = samples.T

            if samples.shape[1] == 1 and self.required_input_channels == 2:
                samples = np.repeat(samples, 2, axis=1)
            elif samples.shape[1] > self.required_input_channels:
                samples = samples[:, : self.required_input_channels]
            elif samples.shape[1] < self.required_input_channels:
                reps = [samples[:, min(i, samples.shape[1] - 1)] for i in range(self.required_input_channels)]
                samples = np.stack(reps, axis=1)

            waveform = self.spectro_audio_mod.Waveform(np.ascontiguousarray(samples, dtype=np.float32), self.sample_rate)
            tokens_frame_major = np.asarray(self.spectro_codec.encode(waveform), dtype=np.int32)
            if tokens_frame_major.ndim != 2:
                return None, None

            # [T, K] -> [1, K, T] to match DAC-style downstream pipeline.
            tokens = torch.from_numpy(np.ascontiguousarray(tokens_frame_major.T[np.newaxis, :, :])).to(torch.int64)
            return tokens, None

        raise RuntimeError(f"Unsupported active codec: {self.codec_id}")

    def decode(self, audio_codes=None, quantized_representation=None):
        """Decode tokens/latents with the active codec."""
        if audio_codes is None and quantized_representation is None:
            return None

        if self.codec_id == "dac":
            target_dtype = self.compute_dtype if self.device.type == "cuda" else torch.float32

            with torch.no_grad():
                with self._autocast_context():
                    if audio_codes is not None:
                        tokens = audio_codes.to(self.device, dtype=torch.int64)
                        return self.model.decode(audio_codes=tokens)

                    quantized_representation = quantized_representation.to(self.device, dtype=target_dtype)
                    return self.model.decode(quantized_representation)

        if self.codec_id == "spectrostream":
            if self.spectro_codec is None:
                self._load_spectrostream()
            if audio_codes is None:
                return None

            tokens = audio_codes.detach().cpu().numpy()
            if tokens.ndim == 3:
                tokens = tokens[0]
            if tokens.ndim != 2:
                return None

            # [K, T] -> [T, K] expected by SpectroStream.
            frame_major = np.asarray(tokens.T, dtype=np.int32)
            waveform = self.spectro_codec.decode(frame_major)
            samples = np.asarray(waveform.samples, dtype=np.float32)
            if samples.ndim == 1:
                channels_first = samples[np.newaxis, :]
            else:
                channels_first = np.ascontiguousarray(samples.T, dtype=np.float32)

            # Return a DAC-like payload for downstream compatibility ([B, C, T]).
            audio_values = torch.from_numpy(np.ascontiguousarray(channels_first[np.newaxis, :, :], dtype=np.float32))
            return SimpleNamespace(audio_values=audio_values)

        return None

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return contextlib.nullcontext()

    def _init_rvq_groups(self, num_codebooks):
        if num_codebooks <= 1:
            self.rvq_groups = [list(range(num_codebooks)), [], []]
            return

        q0 = max(1, num_codebooks // 3)
        q1 = max(q0 + 1, (2 * num_codebooks) // 3)
        q1 = min(q1, num_codebooks)

        self.rvq_groups = [
            list(range(0, q0)),
            list(range(q0, q1)),
            list(range(q1, num_codebooks)),
        ]

    def _load_codebook_embeddings(self):
        quantizer = getattr(self.model, "quantizer", None)
        if quantizer is None:
            return None

        if hasattr(quantizer, "codebooks"):
            embeddings = []
            for book in list(quantizer.codebooks):
                weight = book.weight if hasattr(book, "weight") else book
                embeddings.append(weight.detach().cpu().to(torch.float32))
            return embeddings

        if hasattr(quantizer, "quantizers"):
            embeddings = []
            for book in quantizer.quantizers:
                table = getattr(book, "codebook", None) or getattr(book, "embedding", None) or getattr(book, "embeddings", None)
                if table is None:
                    return None
                weight = table.weight if hasattr(table, "weight") else table
                embeddings.append(weight.detach().cpu().to(torch.float32))
            return embeddings

        if hasattr(quantizer, "embeddings"):
            weight = quantizer.embeddings
            weight = weight.weight if hasattr(weight, "weight") else weight
            if isinstance(weight, torch.Tensor) and weight.ndim == 3:
                return [weight[i].detach().cpu().to(torch.float32) for i in range(weight.shape[0])]

        return None

    def _ensure_rvq_setup(self, audio_codes):
        if audio_codes is None:
            return

        if self.rvq_groups is None:
            self._init_rvq_groups(audio_codes.shape[1])

        if self.codebook_embeddings is None:
            self.codebook_embeddings = self._load_codebook_embeddings()

    def _group_weights(self):
        focus = float(np.clip(self.rvq_focus, 0.0, 1.0))
        coarse = 1.0 - focus
        fine = focus
        mid = 0.5 * (coarse + fine)

        weights = np.array([coarse, mid, fine], dtype=np.float32)
        weight_sum = weights.sum()
        if weight_sum > 0.0:
            weights /= weight_sum
        return weights

    def _normalize(self, vec):
        if vec.numel() == 0:
            return vec
        return torch.nan_to_num(F.normalize(vec, dim=0), nan=0.0, posinf=0.0, neginf=0.0)

    def _descriptors_for_codes(self, codes):
        if self.codebook_embeddings is None:
            token_means = codes.to(torch.float32).mean(dim=1)
            full_desc = self._normalize(token_means.flatten())
            return full_desc, [full_desc, torch.empty(0), torch.empty(0)]

        codebook_vecs = []
        for q, emb in enumerate(self.codebook_embeddings):
            tokens = codes[q].to(torch.long)
            vec = emb[tokens].mean(dim=0)
            codebook_vecs.append(vec)

        full_desc = self._normalize(torch.cat(codebook_vecs, dim=0))

        group_descs = []
        for group in self.rvq_groups:
            if not group:
                group_descs.append(torch.empty(0))
                continue
            stacked = torch.cat([codebook_vecs[q] for q in group], dim=0)
            group_descs.append(self._normalize(stacked))

        return full_desc, group_descs

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return chunk
        chunk = np.asarray(chunk, dtype=np.float32)
        if not self.normalize_input_chunks:
            return chunk
        peak = float(np.max(np.abs(chunk)))
        if peak > 1.0e-6:
            chunk = chunk / peak
        return chunk

    def _prepare_output_audio(self, prepared: np.ndarray) -> np.ndarray:
        prepared = np.asarray(prepared, dtype=np.float32)
        prepared = np.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
        if prepared.size == 0:
            return prepared
        peak = float(np.max(np.abs(prepared)))
        if peak > self.output_peak_target and peak > 1.0e-6:
            prepared = prepared * (self.output_peak_target / peak)
        return np.clip(prepared, -self.output_peak_target, self.output_peak_target)

    def _stream_audio(self, path):
        """Yield normalized mono chunks from disk to keep memory usage low."""
        if not path:
            return

        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Stream skipped, file not found: {path}")
            return

        try:
            with sf.SoundFile(str(path_obj)) as source:
                source_sr = source.samplerate
                block_frames = max(int(self.chunk_duration_s * source_sr), source_sr)
                while True:
                    frames = source.read(block_frames, dtype="float32", always_2d=True)
                    if frames.size == 0:
                        break

                    chunk = np.asarray(frames, dtype=np.float32)
                    if self.required_input_channels <= 1:
                        mono = librosa.to_mono(chunk.T)
                        if source_sr != self.sample_rate:
                            mono = librosa.resample(mono, orig_sr=source_sr, target_sr=self.sample_rate)
                        mono = self._normalize_chunk(mono)
                        if mono.size == 0:
                            continue
                        yield mono
                        continue

                    if source_sr != self.sample_rate:
                        channels = []
                        for ch in range(chunk.shape[1]):
                            channels.append(librosa.resample(chunk[:, ch], orig_sr=source_sr, target_sr=self.sample_rate))
                        min_len = min(len(ch) for ch in channels)
                        chunk = np.stack([ch[:min_len] for ch in channels], axis=1)

                    if chunk.shape[1] == 1 and self.required_input_channels == 2:
                        chunk = np.repeat(chunk, 2, axis=1)
                    elif chunk.shape[1] > self.required_input_channels:
                        chunk = chunk[:, : self.required_input_channels]
                    elif chunk.shape[1] < self.required_input_channels:
                        reps = [chunk[:, min(i, chunk.shape[1] - 1)] for i in range(self.required_input_channels)]
                        chunk = np.stack(reps, axis=1)

                    chunk = self._normalize_chunk(chunk)
                    if chunk.size == 0:
                        continue
                    yield chunk
        except Exception as exc:
            print(f"Failed streaming {path}: {exc}")

    def _ingest_audio_segment(self, segment):
        if segment is None:
            return None

        segment = np.asarray(segment, dtype=np.float32)
        if self.required_input_channels <= 1:
            if segment.ndim > 1:
                if segment.shape[0] < segment.shape[1]:
                    segment = segment.T
                segment = librosa.to_mono(segment.T)
        else:
            if segment.ndim == 1:
                segment = np.repeat(segment[:, np.newaxis], self.required_input_channels, axis=1)
            elif segment.ndim == 2:
                if segment.shape[0] < segment.shape[1]:
                    segment = segment.T
                if segment.shape[1] == 1 and self.required_input_channels == 2:
                    segment = np.repeat(segment, 2, axis=1)
                elif segment.shape[1] > self.required_input_channels:
                    segment = segment[:, : self.required_input_channels]
                elif segment.shape[1] < self.required_input_channels:
                    reps = [segment[:, min(i, segment.shape[1] - 1)] for i in range(self.required_input_channels)]
                    segment = np.stack(reps, axis=1)
            else:
                return None

        if segment.size == 0:
            return None

        audio_codes, _ = self.encode(segment)
        if audio_codes is None or audio_codes.shape[-1] < self.unit:
            return None

        return audio_codes

    def _add_palette_from_codes(self, audio_codes, file_id):
        if audio_codes is None or audio_codes.shape[-1] < self.unit:
            return 0

        self._ensure_rvq_setup(audio_codes)

        codes = audio_codes.squeeze(0)
        segments = codes.unfold(-1, self.unit, self.stride)
        if segments.numel() == 0:
            return 0

        segments = segments.permute(1, 0, 2).contiguous()
        added = 0

        for idx in range(segments.shape[0]):
            grain_codes = segments[idx]
            full_desc, group_descs = self._descriptors_for_codes(grain_codes)

            self._palette_codes_list.append(grain_codes.to(torch.int16))
            self._palette_full_desc_list.append(full_desc.to(torch.float32))
            for g, desc in enumerate(group_descs):
                self._palette_group_desc_lists[g].append(desc.to(torch.float32))
            self._palette_file_ids_list.append(file_id)
            self._palette_frame_indices_list.append(idx * self.stride)
            added += 1

        return added

    def _prepare_palette_tensors(self):
        if not self._palette_codes_list:
            self.palette_codes = None
            self.palette_desc_full = None
            self.palette_desc_groups = None
            self.palette_file_ids = None
            self.palette_frame_indices = None
            return

        self.palette_codes = torch.stack(self._palette_codes_list, dim=0).to(torch.int16)
        self.palette_desc_full = self._normalize(torch.stack(self._palette_full_desc_list, dim=0))

        group_descs = []
        for group_list in self._palette_group_desc_lists:
            if group_list:
                group_descs.append(self._normalize(torch.stack(group_list, dim=0)))
            else:
                group_descs.append(torch.empty((self.palette_codes.shape[0], 0)))
        self.palette_desc_groups = group_descs

        self.palette_file_ids = np.asarray(self._palette_file_ids_list, dtype=np.int32)
        self.palette_frame_indices = np.asarray(self._palette_frame_indices_list, dtype=np.int64)

    def _group_distances(self, target_group_descs):
        distances = []
        if self.palette_desc_groups is None:
            return distances

        for group_idx, target_desc in enumerate(target_group_descs):
            if target_desc.numel() == 0:
                distances.append(torch.zeros(self.palette_desc_groups[group_idx].shape[0]))
                continue
            target_desc = self._normalize(target_desc.to(torch.float32))
            palette_desc = self.palette_desc_groups[group_idx]
            dot = torch.matmul(palette_desc, target_desc)
            distances.append(1.0 - dot)
        return distances

    def set_temperature(self, temperature, threshold):
        self.temperature = float(temperature)
        self.threshold = float(threshold)

    def set_unit(self, unit, stride):
        self.unit = max(int(unit), 1)
        self.stride = max(int(stride), 1)

    def set_matching(self, continuity, rvq_focus):
        self.continuity = float(continuity)
        self.rvq_focus = float(rvq_focus)

    def set_topk(self, top_k):
        self.top_k = max(int(top_k), 1)

    def set_ablation(self, match_mode: str, swap_mode: str):
        match_mode = (match_mode or "beam").strip().lower()
        swap_mode = (swap_mode or "full_layer").strip().lower()
        swap_mode = self.SWAP_ALIASES.get(swap_mode, swap_mode)

        if match_mode not in self.MATCH_MODES:
            raise ValueError(f"Unsupported match_mode: {match_mode}")
        if swap_mode not in self.SWAP_MODES:
            raise ValueError(f"Unsupported swap_mode: {swap_mode}")

        self.match_mode = match_mode
        self.swap_mode = swap_mode

    def build_dataset(self, files, aug_checkbox: bool):
        aug_checkbox = True
        resolved_files = self._materialize_files(files)
        if not resolved_files:
            return "Please upload at least one audio file before building the palette."

        self.files = resolved_files
        self.last_aug = bool(aug_checkbox)
        self.prev_best_index = None
        self._palette_codes_list = []
        self._palette_full_desc_list = []
        self._palette_group_desc_lists = [[], [], []]
        self._palette_file_ids_list = []
        self._palette_frame_indices_list = []

        n_files = 0
        total_grains = 0

        for file_id, path in enumerate(resolved_files):
            file_segments = []
            path_str = str(path)

            try:
                if aug_checkbox:
                    y, _ = librosa.load(path_str, sr=self.sample_rate, mono=True)
                    y = librosa.util.normalize(y.astype(np.float32, copy=False))
                    base_codes = self._ingest_audio_segment(y)
                    if base_codes is not None:
                        file_segments.append(base_codes)

                    for vol in self.vol_aug:
                        codes = self._ingest_audio_segment(np.clip(y * vol, -1.0, 1.0))
                        if codes is not None:
                            file_segments.append(codes)

                    for pitch in self.pitch_aug:
                        y_pitch = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=pitch)
                        y_pitch = librosa.util.normalize(y_pitch.astype(np.float32, copy=False))
                        codes = self._ingest_audio_segment(y_pitch)
                        if codes is not None:
                            file_segments.append(codes)
                else:
                    for chunk in self._stream_audio(path_str):
                        codes = self._ingest_audio_segment(chunk)
                        if codes is not None:
                            file_segments.append(codes)

                if file_segments:
                    file_codes = torch.cat(file_segments, dim=-1)
                    total_grains += self._add_palette_from_codes(file_codes, file_id)
                    n_files += 1
            except Exception as exc:
                print(f"Error processing {path}: {exc}")

        if not self._palette_codes_list:
            self._prepare_palette_tensors()
            return "No audio processed. Please verify the input files."

        self._prepare_palette_tensors()
        return f"Done! {n_files} files processed. Codebook grains: {total_grains}."

    def _meta_penalty(self, prev_idx, idx):
        if self.palette_file_ids is None or self.palette_frame_indices is None:
            return 0.0

        if self.palette_file_ids[prev_idx] != self.palette_file_ids[idx]:
            return 1.0

        frame_delta = abs(int(self.palette_frame_indices[idx]) - int(self.palette_frame_indices[prev_idx]))
        if frame_delta <= self.stride:
            return 0.0

        return min(1.0, frame_delta / float(self.unit * 4))

    def _transition_cost(self, prev_idx, idx):
        latent = 1.0 - float(torch.dot(self.palette_desc_full[prev_idx], self.palette_desc_full[idx]))
        return latent + self._meta_penalty(prev_idx, idx)

    def _sequence_diagnostics(self, grains, path, select_ms):
        if not grains or not path:
            return {
                "objective_j": float("nan"),
                "emission_cost": float("nan"),
                "transition_cost": float("nan"),
                "weighted_transition_cost": float("nan"),
                "index_jitter": float("nan"),
                "file_switch_rate": float("nan"),
                "adjacent_step_rate": float("nan"),
                "runtime_ms": float(select_ms),
            }

        selected = []
        for grain_idx, cand_idx in enumerate(path):
            candidates = grains[grain_idx]["candidates"]
            selected.append(candidates[int(cand_idx)])

        ann = [int(c["ann_index"]) for c in selected]
        emissions = [float(c["emission"]) for c in selected]
        transition_values = []
        file_switches = []
        adjacent_steps = []
        for prev_ann, next_ann in zip(ann[:-1], ann[1:]):
            transition_values.append(float(self._transition_cost(prev_ann, next_ann)))
            if self.palette_file_ids is not None and self.palette_frame_indices is not None:
                same_file = bool(self.palette_file_ids[prev_ann] == self.palette_file_ids[next_ann])
                file_switches.append(float(not same_file))
                frame_delta = abs(int(self.palette_frame_indices[next_ann]) - int(self.palette_frame_indices[prev_ann]))
                adjacent_steps.append(float(same_file and frame_delta <= self.stride))

        emission_cost = float(np.sum(emissions))
        transition_cost = float(np.sum(transition_values)) if transition_values else 0.0
        weighted_transition_cost = float(self.continuity * transition_cost)
        jitter = float(np.mean(np.abs(np.diff(np.asarray(ann, dtype=np.float64))))) if len(ann) > 1 else 0.0
        return {
            "objective_j": emission_cost + weighted_transition_cost,
            "emission_cost": emission_cost,
            "transition_cost": transition_cost,
            "weighted_transition_cost": weighted_transition_cost,
            "index_jitter": jitter,
            "file_switch_rate": float(np.mean(file_switches)) if file_switches else float("nan"),
            "adjacent_step_rate": float(np.mean(adjacent_steps)) if adjacent_steps else float("nan"),
            "runtime_ms": float(select_ms),
            "steps": int(len(path)),
            "candidate_count_max": int(max(len(g["candidates"]) for g in grains)),
        }

    def _select_path_greedy(self, grains, use_transition: bool):
        path = []
        prev_best = self.prev_best_index
        for grain in grains:
            best_idx = 0
            best_score = float("inf")
            for cand_idx, candidate in enumerate(grain["candidates"]):
                score = candidate["emission"]
                if use_transition and self.continuity > 0.0 and prev_best is not None:
                    score += self.continuity * self._transition_cost(prev_best, candidate["ann_index"])
                if score < best_score:
                    best_score = score
                    best_idx = cand_idx
            path.append(best_idx)
            prev_best = grain["candidates"][best_idx]["ann_index"]
        return path

    def _select_path_beam(self, grains):
        beam_width = max(1, min(self.beam_width, max(len(grain["candidates"]) for grain in grains)))
        history = []

        first = grains[0]
        beam = []
        for cand_idx, candidate in enumerate(first["candidates"]):
            score = candidate["emission"]
            if self.prev_best_index is not None:
                score += self.continuity * self._transition_cost(self.prev_best_index, candidate["ann_index"])
            beam.append({"score": score, "candidate_idx": cand_idx, "back": -1})

        beam = sorted(beam, key=lambda x: x["score"])[:beam_width]
        history.append(beam)

        for grain_idx in range(1, len(grains)):
            grain = grains[grain_idx]
            new_beam = []

            for prev_idx, prev_state in enumerate(history[-1]):
                prev_candidate = grains[grain_idx - 1]["candidates"][prev_state["candidate_idx"]]
                prev_ann = prev_candidate["ann_index"]
                for cand_idx, candidate in enumerate(grain["candidates"]):
                    score = prev_state["score"] + candidate["emission"]
                    if self.continuity > 0.0:
                        score += self.continuity * self._transition_cost(prev_ann, candidate["ann_index"])
                    new_beam.append({"score": score, "candidate_idx": cand_idx, "back": prev_idx})

            new_beam = sorted(new_beam, key=lambda x: x["score"])[:beam_width]
            history.append(new_beam)

        best_idx = min(range(len(history[-1])), key=lambda i: history[-1][i]["score"])
        path = [0] * len(grains)
        for grain_idx in range(len(grains) - 1, -1, -1):
            state = history[grain_idx][best_idx]
            path[grain_idx] = state["candidate_idx"]
            best_idx = state["back"]
        return path

    def _select_path_viterbi(self, grains):
        first = grains[0]
        prev_scores = np.asarray([float(c["emission"]) for c in first["candidates"]], dtype=np.float64)
        if self.prev_best_index is not None and self.continuity > 0.0:
            for idx, candidate in enumerate(first["candidates"]):
                prev_scores[idx] += self.continuity * self._transition_cost(self.prev_best_index, candidate["ann_index"])

        backs = []
        transition_cache = {}
        for grain_idx in range(1, len(grains)):
            prev_candidates = grains[grain_idx - 1]["candidates"]
            candidates = grains[grain_idx]["candidates"]
            next_scores = np.empty(len(candidates), dtype=np.float64)
            back = np.zeros(len(candidates), dtype=np.int32)
            for cand_idx, candidate in enumerate(candidates):
                ann = int(candidate["ann_index"])
                best_score = float("inf")
                best_prev = 0
                for prev_idx, prev_candidate in enumerate(prev_candidates):
                    prev_ann = int(prev_candidate["ann_index"])
                    key = (prev_ann, ann)
                    if key not in transition_cache:
                        transition_cache[key] = float(self._transition_cost(prev_ann, ann))
                    score = prev_scores[prev_idx] + float(candidate["emission"])
                    if self.continuity > 0.0:
                        score += self.continuity * transition_cache[key]
                    if score < best_score:
                        best_score = score
                        best_prev = prev_idx
                next_scores[cand_idx] = best_score
                back[cand_idx] = best_prev
            backs.append(back)
            prev_scores = next_scores

        best_idx = int(np.argmin(prev_scores))
        path = [0] * len(grains)
        path[-1] = best_idx
        for grain_idx in range(len(grains) - 1, 0, -1):
            best_idx = int(backs[grain_idx - 1][best_idx])
            path[grain_idx - 1] = best_idx
        return path

    def _select_path(self, grains):
        if not grains:
            return []

        started = time.perf_counter()
        if self.match_mode == "greedy":
            path = self._select_path_greedy(grains, use_transition=False)
        elif self.match_mode == "greedy_smooth":
            path = self._select_path_greedy(grains, use_transition=True)
        elif self.match_mode == "viterbi":
            path = self._select_path_viterbi(grains)
        else:
            path = self._select_path_beam(grains)
        select_ms = (time.perf_counter() - started) * 1000.0
        self.last_sequence_diagnostics = self._sequence_diagnostics(grains, path, select_ms)
        return path

    def morph_audio(self, target_file, return_debug=False):
        started = time.perf_counter()
        self.last_timings = {"encode_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0}

        def _fallback():
            audio = np.zeros(1024, dtype=np.int16)
            if return_debug:
                return self.sample_rate, audio, {"tokens": np.zeros((1, 1), dtype=np.int32), "match_indices": np.zeros(0, dtype=np.int32)}
            return self.sample_rate, audio

        target_path = self._resolve_file_entry(target_file)
        if target_path is None or not target_path.exists():
            return _fallback()

        if self.palette_codes is None or self.palette_codes.numel() == 0:
            return _fallback()

        print("Creating codes for source audio")
        target_segments = []
        encode_started = time.perf_counter()

        for chunk in self._stream_audio(str(target_path)):
            audio_codes, _ = self.encode(chunk)
            if audio_codes is not None and audio_codes.shape[-1] >= self.unit:
                target_segments.append(audio_codes)

        if not target_segments:
            return _fallback()

        target_codes = torch.cat(target_segments, dim=-1).squeeze(0)
        self.last_timings["encode_ms"] = (time.perf_counter() - encode_started) * 1000.0

        if target_codes.shape[-1] < self.unit:
            return _fallback()

        self._ensure_rvq_setup(target_codes.unsqueeze(0))
        output_codes = torch.zeros_like(target_codes) if self.swap_mode == "palette_only" else target_codes.clone()

        grain_starts = list(range(0, target_codes.shape[-1] - self.unit + 1, self.stride))
        if not grain_starts:
            return _fallback()

        weights = self._group_weights()
        grains = []

        print("Matching grains...")
        for start in tqdm(grain_starts):
            grain_codes = target_codes[:, start : start + self.unit]
            full_desc, group_descs = self._descriptors_for_codes(grain_codes)
            group_distances = self._group_distances(group_descs)

            emission = torch.zeros(group_distances[0].shape[0])
            for g_idx, dist in enumerate(group_distances):
                if dist.numel() == 0:
                    continue
                emission += float(weights[g_idx]) * dist

            candidate_count = min(self.candidate_count, emission.numel())
            if candidate_count == 0:
                continue

            top_indices = torch.topk(-emission, candidate_count).indices
            candidates = []
            fine_dist = group_distances[-1] if group_distances else emission
            for idx in top_indices.tolist():
                candidates.append(
                    {
                        "ann_index": idx,
                        "emission": float(emission[idx]),
                        "fine": float(fine_dist[idx]),
                        "group_dists": [
                            float(dist[idx]) if dist.numel() > 0 else 0.0 for dist in group_distances
                        ],
                    }
                )

            candidates.sort(key=lambda x: x["emission"])
            grains.append(
                {
                    "start": start,
                    "candidates": candidates,
                }
            )

        if not grains:
            return _fallback()

        path = self._select_path(grains)
        if path:
            last_candidate = grains[-1]["candidates"][path[-1]]
            self.prev_best_index = last_candidate["ann_index"]

        fine_group = self.rvq_groups[-1] if self.rvq_groups else []
        coarse_group = self.rvq_groups[0] if self.rvq_groups else []
        mid_group = self.rvq_groups[1] if self.rvq_groups else []

        matched_indices = []
        selected_grain_records = []
        coarse_transfer_flags = []

        def _copy_group_from_candidate(group, candidate_index, start, span):
            for q in group:
                output_codes[q, start : start + span] = self.palette_codes[candidate_index, q, :span]

        def _vote_group_from_topk(group, grain, start, span):
            top_k = min(self.top_k, len(grain["candidates"]))
            top_candidates = grain["candidates"][:top_k]
            fine_dists = torch.tensor([c["fine"] for c in top_candidates], dtype=torch.float32)
            temperature = max(float(self.temperature), 1.0e-4)
            logits = -fine_dists / temperature
            weights_k = torch.softmax(logits, dim=0).cpu().numpy()

            for q in group:
                for u in range(span):
                    scores = {}
                    for k, cand in enumerate(top_candidates):
                        code = int(self.palette_codes[cand["ann_index"], q, u].item())
                        scores[code] = scores.get(code, 0.0) + float(weights_k[k])
                    if scores:
                        best_code = max(scores.items(), key=lambda kv: kv[1])[0]
                        output_codes[q, start + u] = best_code

        for grain_idx, grain in enumerate(grains):
            candidate_idx = path[grain_idx]
            candidate = grain["candidates"][candidate_idx]
            path_index = candidate["ann_index"]
            matched_indices.append(path_index)
            start = grain["start"]
            if self.swap_mode == "palette_only" and grain_idx + 1 == len(grains):
                span = output_codes.shape[-1] - start
            else:
                span = min(self.unit, output_codes.shape[-1] - start)

            coarse_transfer = candidate["emission"] <= self.threshold
            coarse_transfer_flags.append(float(coarse_transfer))
            selected_grain_records.append(
                {
                    "grain_index": int(grain_idx),
                    "start": int(start),
                    "candidate_rank": int(candidate_idx),
                    "ann_index": int(path_index),
                    "emission": float(candidate["emission"]),
                    "fine": float(candidate["fine"]),
                    "group_dists": [float(x) for x in candidate.get("group_dists", [])],
                    "coarse_transfer": bool(coarse_transfer),
                }
            )

            if self.swap_mode == "identity":
                continue
            if self.swap_mode == "full_layer_gated":
                if coarse_transfer:
                    output_codes[:, start : start + span] = self.palette_codes[path_index, :, :span]
                continue
            if self.swap_mode in {"palette_only", "full_layer_forced"}:
                output_codes[:, start : start + span] = self.palette_codes[path_index, :, :span]
                continue
            if self.swap_mode == "coarse_gated":
                if coarse_transfer:
                    _copy_group_from_candidate(coarse_group, path_index, start, span)
                continue
            if self.swap_mode == "coarse_forced":
                _copy_group_from_candidate(coarse_group, path_index, start, span)
                continue
            if self.swap_mode == "middle_only":
                _copy_group_from_candidate(mid_group, path_index, start, span)
                continue
            if self.swap_mode == "fine_only":
                _vote_group_from_topk(fine_group, grain, start, span)
                continue
            if self.swap_mode == "middle_fine":
                _copy_group_from_candidate(mid_group, path_index, start, span)
                _vote_group_from_topk(fine_group, grain, start, span)
                continue

            if coarse_transfer:
                _copy_group_from_candidate(coarse_group, path_index, start, span)
            _copy_group_from_candidate(mid_group, path_index, start, span)
            _vote_group_from_topk(fine_group, grain, start, span)

        output_codes = output_codes.unsqueeze(0).to(torch.int64)
        target_codes_for_diag = target_codes.to(torch.int64)
        output_codes_for_diag = output_codes.squeeze(0)

        def _token_change_rate(group):
            if not group:
                return float("nan")
            src = target_codes_for_diag[group, :]
            out = output_codes_for_diag[group, :]
            if src.numel() == 0:
                return float("nan")
            return float(torch.mean((src != out).to(torch.float32)).item())

        token_change_rates = {
            "coarse": _token_change_rate(coarse_group),
            "middle": _token_change_rate(mid_group),
            "fine": _token_change_rate(fine_group),
            "overall": float(torch.mean((target_codes_for_diag != output_codes_for_diag).to(torch.float32)).item()),
        }
        selected_emissions = [r["emission"] for r in selected_grain_records]
        selected_group_dists = np.asarray([r["group_dists"] for r in selected_grain_records if r["group_dists"]], dtype=np.float64)
        band_distance_means = {}
        if selected_group_dists.size > 0:
            names = ["coarse", "middle", "fine"]
            for idx, name in enumerate(names[: selected_group_dists.shape[1]]):
                band_distance_means[name] = float(np.mean(selected_group_dists[:, idx]))
        diagnostics = {
            "codec": self.codec_id,
            "match_mode": self.match_mode,
            "swap_mode": self.swap_mode,
            "threshold": float(self.threshold),
            "rho": float(self.rvq_focus),
            "rvq_groups": [list(map(int, g)) for g in (self.rvq_groups or [])],
            "sequence": dict(self.last_sequence_diagnostics),
            "token_change_rates": token_change_rates,
            "coarse_transfer_fraction": float(np.mean(coarse_transfer_flags)) if coarse_transfer_flags else float("nan"),
            "coarse_fallback_fraction": float(1.0 - np.mean(coarse_transfer_flags)) if coarse_transfer_flags else float("nan"),
            "selected_emission_mean": float(np.mean(selected_emissions)) if selected_emissions else float("nan"),
            "selected_emission_median": float(np.median(selected_emissions)) if selected_emissions else float("nan"),
            "selected_band_distance_mean": band_distance_means,
            "selected_grains": selected_grain_records,
        }
        decode_started = time.perf_counter()
        decoded = self.decode(audio_codes=output_codes)
        self.last_timings["decode_ms"] = (time.perf_counter() - decode_started) * 1000.0

        if decoded is None or getattr(decoded, "audio_values", None) is None:
            return _fallback()

        audio_output = decoded.audio_values

        if hasattr(audio_output, "squeeze"):
            final_audio = audio_output
        elif hasattr(audio_output, "data"):
            final_audio = audio_output.data
        elif hasattr(audio_output, "tensor"):
            final_audio = audio_output.tensor
        else:
            final_audio = torch.as_tensor(audio_output)

        final_np = final_audio.detach().cpu().numpy()
        final_np = np.squeeze(final_np)

        if final_np.ndim == 0:
            final_np = np.expand_dims(final_np, axis=0)

        if final_np.ndim == 1:
            prepared = final_np
        elif final_np.shape[0] <= final_np.shape[-1]:
            prepared = np.moveaxis(final_np, 0, -1)
        else:
            prepared = final_np

        prepared = self._prepare_output_audio(prepared)
        scaled = np.round(prepared * 32767.0).astype(np.int16, copy=False)
        self.last_timings["total_ms"] = (time.perf_counter() - started) * 1000.0

        if return_debug:
            return self.sample_rate, scaled, {
                "tokens": output_codes.squeeze(0).detach().cpu().numpy().astype(np.int32),
                "match_indices": np.asarray(matched_indices, dtype=np.int32),
                "timings": dict(self.last_timings),
                "diagnostics": diagnostics,
            }

        return self.sample_rate, scaled


_synth = None


def _available_codecs():
    choices = ["dac"]
    try:
        import importlib.util

        if importlib.util.find_spec("magenta_rt") is not None:
            choices.append("spectrostream")
    except Exception:
        pass
    return choices


def _get_synth():
    global _synth
    if _synth is None:
        _synth = LatentGranularSynthesis()
    return _synth


def build_dataset(files, aug_checkbox):
    return _get_synth().build_dataset(files, aug_checkbox)


def morph_audio(target_file):
    return _get_synth().morph_audio(target_file)


def set_codec(codec_id):
    return _get_synth().set_codec(codec_id)


def set_codec_ui(codec_id):
    synth = _get_synth()
    message = synth.set_codec(codec_id)
    params = synth.runtime_params()
    return (
        message,
        params["temperature"],
        params["threshold"],
        params["continuity"],
        params["rvq_focus"],
        params["unit"],
        params["stride"],
        params["top_k"],
    )


def set_ablation_mode(match_mode, swap_mode):
    _get_synth().set_ablation(match_mode, swap_mode)
    return f"Matching mode set to '{match_mode}', swap mode set to '{swap_mode}'."


def temperature(temperature, threshold):
    return _get_synth().set_temperature(temperature, threshold)


def unit(unit, stride):
    return _get_synth().set_unit(unit, stride)


def matching(continuity, rvq_focus):
    return _get_synth().set_matching(continuity, rvq_focus)


def topk(top_k):
    return _get_synth().set_topk(top_k)


def load_demo_example(example_name):
    for name, sources, target in _available_demo_examples():
        if name == example_name:
            return sources, True, target, f"Loaded example: {name}."
    return gr.update(), gr.update(), gr.update(), "Example files are missing."


def morph_audio_with_mix(target_file, dry_wet):
    result = _get_synth().morph_audio(target_file)
    if result is None:
        return None, None, None

    sr, wet = result
    wet_arr = np.asarray(wet, dtype=np.float32)
    if np.issubdtype(wet.dtype, np.integer):
        wet_arr = wet_arr / 32767.0

    dry = np.zeros_like(wet_arr, dtype=np.float32)
    target_path = _get_synth()._resolve_file_entry(target_file)
    if target_path is not None and target_path.exists():
        wet_is_stereo = wet_arr.ndim == 2 and wet_arr.shape[1] > 1
        dry_audio, _ = librosa.load(str(target_path), sr=sr, mono=not wet_is_stereo)
        dry_audio = np.asarray(dry_audio, dtype=np.float32)
        if wet_is_stereo:
            if dry_audio.ndim == 1:
                dry_audio = np.repeat(dry_audio[:, np.newaxis], wet_arr.shape[1], axis=1)
            elif dry_audio.ndim == 2 and dry_audio.shape[0] < dry_audio.shape[1]:
                dry_audio = dry_audio.T
            if dry_audio.shape[1] > wet_arr.shape[1]:
                dry_audio = dry_audio[:, : wet_arr.shape[1]]
            elif dry_audio.shape[1] < wet_arr.shape[1]:
                reps = [dry_audio[:, min(i, dry_audio.shape[1] - 1)] for i in range(wet_arr.shape[1])]
                dry_audio = np.stack(reps, axis=1)
        n = min(dry_audio.shape[0], wet_arr.shape[0])
        dry[:n] = dry_audio[:n]
        wet_arr = wet_arr[:n]
        dry = dry[:n]

    mix_ratio = float(np.clip(dry_wet, 0.0, 1.0))
    mixed = np.clip((1.0 - mix_ratio) * dry + mix_ratio * wet_arr, -1.0, 1.0)
    wet_arr = np.clip(wet_arr, -1.0, 1.0)
    dry = np.clip(dry, -1.0, 1.0)
    return (sr, dry), (sr, wet_arr), (sr, mixed)


def _build_demo():
    defaults = LatentGranularSynthesis.DAC_DEFAULTS
    codec_id = "dac"
    match_mode = "beam"
    swap_mode = "palette_only"
    with gr.Blocks(
        elem_id="neural-morphing-app",
        fill_width=True,
        title="Neural Morphing",
    ) as demo:
        example_names = _demo_example_names()
        load_example_btn = None
        example_dropdown = None

        gr.HTML(HERO_HTML)
        with gr.Row(elem_classes=["nm-main-grid"]):
            with gr.Column(scale=4, min_width=320, elem_classes=["nm-panel"]):
                gr.Markdown("### Palette Sounds")
                db_file = gr.File(file_count="multiple", label="Palette Sounds")
                if example_names:
                    with gr.Row(elem_classes=["nm-example-row"]):
                        example_dropdown = gr.Dropdown(
                            choices=example_names,
                            value=example_names[0],
                            label="Demo Example",
                            scale=4,
                        )
                        load_example_btn = gr.Button(
                            "Load Example",
                            elem_classes=["nm-secondary"],
                            scale=1,
                            min_width=120,
                        )
                aug_checkbox = gr.Checkbox(label="Palette Augmentation", value=True, visible=False)
                b1 = gr.Button("Process palette sounds", elem_classes=["nm-secondary"])
                text = gr.Textbox(label="Result")

            with gr.Column(scale=6, min_width=360, elem_classes=["nm-panel"]):
                gr.Markdown("### Morph Engine")
                with gr.Row():
                    codec_dropdown = gr.Dropdown(
                        choices=_available_codecs(),
                        value=codec_id,
                        label="Codec",
                    )
                    match_mode_dropdown = gr.Dropdown(
                        choices=["beam", "greedy", "greedy_smooth", "viterbi"],
                        value=match_mode,
                        label="Match Mode",
                    )
                    swap_mode_dropdown = gr.Dropdown(
                        choices=[
                            "palette_only",
                            "full_layer_gated",
                            "rvq_group_current",
                            "full_layer_forced",
                            "coarse_gated",
                            "coarse_forced",
                            "middle_only",
                            "fine_only",
                            "middle_fine",
                            "identity",
                        ],
                        value=swap_mode,
                        label="Swap Mode",
                    )

                target_file = gr.File(label="Source Sound")

                with gr.Row():
                    temp_slider = gr.Slider(0.1, 2.0, value=defaults["temperature"], label="Temperature")
                    threshold_slider = gr.Slider(0.1, 2.0, value=defaults["threshold"], label="Threshold")

                with gr.Row():
                    continuity_slider = gr.Slider(0.0, 1.0, value=defaults["continuity"], label="Continuity")
                    rvq_focus_slider = gr.Slider(0.0, 1.0, value=defaults["rvq_focus"], label="RVQ Focus")

                with gr.Row():
                    unit_slider = gr.Slider(1, 16, value=defaults["unit"], step=1, label="Unit Size")
                    stride_slider = gr.Slider(1, 16, value=defaults["stride"], step=1, label="Stride")

                with gr.Row():
                    topk_slider = gr.Slider(1, 8, value=defaults["top_k"], step=1, label="Top-K")

                with gr.Row():
                    drywet_preview = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Playback Dry/Wet")

                b2 = gr.Button("Morph Audio", elem_classes=["nm-primary"])
                gr.Markdown("### Playback")
                with gr.Row(elem_classes=["nm-audio-grid"]):
                    dry_player = gr.Audio(label="Dry")
                    wet_player = gr.Audio(label="Wet")
                    mix_player = gr.Audio(label="Dry/Wet Mix")

        temp_slider.change(temperature, inputs=[temp_slider, threshold_slider])
        threshold_slider.change(temperature, inputs=[temp_slider, threshold_slider])
        continuity_slider.change(matching, inputs=[continuity_slider, rvq_focus_slider])
        rvq_focus_slider.change(matching, inputs=[continuity_slider, rvq_focus_slider])
        unit_slider.change(unit, inputs=[unit_slider, stride_slider])
        stride_slider.change(unit, inputs=[unit_slider, stride_slider])
        topk_slider.change(topk, inputs=[topk_slider])
        codec_dropdown.change(
            set_codec_ui,
            inputs=[codec_dropdown],
            outputs=[text, temp_slider, threshold_slider, continuity_slider, rvq_focus_slider, unit_slider, stride_slider, topk_slider],
        )
        match_mode_dropdown.change(set_ablation_mode, inputs=[match_mode_dropdown, swap_mode_dropdown], outputs=text)
        swap_mode_dropdown.change(set_ablation_mode, inputs=[match_mode_dropdown, swap_mode_dropdown], outputs=text)

        if load_example_btn is not None and example_dropdown is not None:
            load_example_btn.click(
                load_demo_example,
                inputs=[example_dropdown],
                outputs=[db_file, aug_checkbox, target_file, text],
            )
        b1.click(build_dataset, inputs=[db_file, aug_checkbox], outputs=text)
        b2.click(morph_audio_with_mix, inputs=[target_file, drywet_preview], outputs=[dry_player, wet_player, mix_player])

    return demo


def main():
    demo = _build_demo()
    launch_kwargs = {"show_error": True, "css": APP_CSS}
    allowed_paths = [str(path) for path in (ASSETS_DIR, EXAMPLES_DIR) if path.exists()]
    if allowed_paths:
        launch_kwargs["allowed_paths"] = allowed_paths
    server_name = os.getenv("GRADIO_SERVER_NAME")
    server_port = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT")
    if server_name:
        launch_kwargs["server_name"] = server_name
    if server_port:
        launch_kwargs["server_port"] = int(server_port)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
