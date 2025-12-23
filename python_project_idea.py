"""
Latent Granular Synthesis with DAC (Descript Audio Codec)
========================================================
creates a "granular codebook" by encoding a source audio corpus
into latent vector segments, then matches each latent grain of a
target audio signal to its closest counterpart in the codebook.
"""

import contextlib
import os
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, DacModel

class LatentGranularSynthesis:
    def __init__(self, model_name="descript/dac_44khz", device=None, chunk_duration_s=8.0, match_batch=2048):
        """Initialize with DAC model from HuggingFace."""
        preferred_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(preferred_device)
        self.compute_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.chunk_duration_s = max(chunk_duration_s, 1.0)
        self.match_batch = max(int(match_batch), 1)

        self.model = DacModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Get sampling rate from processor
        self.sample_rate = self.processor.sampling_rate
        print(f"Using DAC model: {model_name}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Compute device: {self.device}")

        self.unit = 2
        self.stride = 2
        self.temperature = 0.01
        self.threshold = 1.0
        self.files = None
        self.pitch_aug = [-5, -2, 2, 5]
        self.vol_aug = [0.3, 0.7]

        self.codedb_segments = []
        self.codedb = None
        self.db = None
        self.db_flat = None
        self.db_flat_device = None

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
        """Encode audio using DAC."""
        if audio_array is None:
            return None, None

        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.detach().cpu().numpy()

        audio_array = np.asarray(audio_array, dtype=np.float32)

        if audio_array.ndim > 1:
            audio_array = librosa.to_mono(audio_array)

        if audio_array.size == 0:
            return None, None

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
    
    def decode(self, quantized_representation):
        """Decode quantized representation back to audio using DAC."""
        if quantized_representation is None:
            return None

        target_dtype = self.compute_dtype if self.device.type == "cuda" else torch.float32
        quantized_representation = quantized_representation.to(self.device, dtype=target_dtype)

        with torch.no_grad():
            with self._autocast_context():
                audio_values = self.model.decode(quantized_representation)

        return audio_values
    
    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return contextlib.nullcontext()

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

                    mono = librosa.to_mono(frames.T)
                    if source_sr != self.sample_rate:
                        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=self.sample_rate)
                    mono = librosa.util.normalize(mono)
                    if mono.size == 0:
                        continue
                    yield mono
        except Exception as exc:
            print(f"Failed streaming {path}: {exc}")

    def _ingest_audio_segment(self, segment):
        """Encode a segment and append it to the codebook cache."""
        if segment is None:
            return False

        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim > 1:
            segment = librosa.to_mono(segment)

        if segment.size == 0:
            return False

        _, quantized = self.encode(segment)
        if quantized is None or quantized.shape[-1] < self.unit:
            return False

        self.codedb_segments.append(quantized.contiguous())
        return True

    def _finalize_codebook(self):
        """Finalize tensor views used for matching."""
        if not self.codedb_segments:
            self.codedb = None
            self.db = None
            self.db_flat = None
            self.db_flat_device = None
            return

        self.codedb = torch.cat(self.codedb_segments, dim=-1)

        segment_views = []
        for chunk in self.codedb_segments:
            # chunk shape: (batch, features, frames)
            unfolded = chunk.unfold(-1, self.unit, self.stride)
            if unfolded.numel() == 0:
                continue
            # -> (batch, features, segments, unit)
            unfolded = unfolded.squeeze(0).permute(1, 0, 2).contiguous()
            segment_views.append(unfolded)

        if not segment_views:
            self.db = torch.empty(0, dtype=torch.float32)
            self.db_flat = torch.empty(0, dtype=torch.float32)
            self.db_flat_device = None
            return

        self.db = torch.cat(segment_views, dim=0).to(torch.float32)
        self.db_flat = self.db.reshape(self.db.shape[0], -1).contiguous()
        if self.db_flat.numel() > 0:
            self.db_flat = torch.nan_to_num(F.normalize(self.db_flat, dim=1), nan=0.0, posinf=0.0, neginf=0.0)
        self._prepare_match_tensors()

    def _prepare_match_tensors(self):
        self.db_flat_device = None
        if self.device.type != "cuda" or self.db_flat is None:
            return
        try:
            self.db_flat_device = self.db_flat.to(self.device)
        except RuntimeError:
            # Not enough GPU memory; fall back to CPU matching.
            self.db_flat_device = None
            torch.cuda.empty_cache()
            print("Warning: codebook is too large to keep on the GPU. Falling back to CPU matching.")

    def _compute_distances(self, target_code):
        if self.db_flat is None or self.db_flat.numel() == 0:
            return torch.empty(0)

        target_vec = target_code.reshape(-1).to(torch.float32)
        target_vec = torch.nan_to_num(F.normalize(target_vec, dim=0), nan=0.0, posinf=0.0, neginf=0.0)
        distances = []

        if self.db_flat_device is not None:
            target_vec = target_vec.to(self.db_flat_device.dtype).to(self.device)
            for start in range(0, self.db_flat_device.shape[0], self.match_batch):
                chunk = self.db_flat_device[start:start + self.match_batch]
                sims = torch.matmul(chunk, target_vec)
                distances.append((1 - sims).cpu())
        else:
            target_vec = target_vec.to(self.db_flat.dtype)
            for start in range(0, self.db_flat.shape[0], self.match_batch):
                chunk = self.db_flat[start:start + self.match_batch]
                sims = torch.matmul(chunk, target_vec)
                distances.append(1 - sims)

        if not distances:
            return torch.empty(0)

        return torch.cat(distances, dim=0)

    def set_temperature(self, temperature, threshold):
        self.temperature = temperature * 0.01
        self.threshold = threshold

    def set_unit(self, unit, stride):
        self.unit = max(int(unit), 1)
        self.stride = max(int(stride), 1)
        if self.codedb_segments:
            self._finalize_codebook()

    def build_dataset(self, files, aug_checkbox: bool):
        resolved_files = self._materialize_files(files)
        if not resolved_files:
            return {"message": "Please upload at least one audio file before building the palette."}

        self.files = resolved_files
        self.codedb_segments = []
        self.codedb = None
        self.db = None
        self.db_flat = None
        self.db_flat_device = None

        n_files = 0

        for path in resolved_files:
            processed_any = False
            path_str = str(path)
            try:
                if aug_checkbox:
                    y, _ = librosa.load(path_str, sr=self.sample_rate, mono=True)
                    y = librosa.util.normalize(y.astype(np.float32, copy=False))
                    processed_any |= self._ingest_audio_segment(y)

                    for vol in self.vol_aug:
                        processed_any |= self._ingest_audio_segment(np.clip(y * vol, -1.0, 1.0))

                    for pitch in self.pitch_aug:
                        y_pitch = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=pitch)
                        y_pitch = librosa.util.normalize(y_pitch.astype(np.float32, copy=False))
                        processed_any |= self._ingest_audio_segment(y_pitch)
                else:
                    for chunk in self._stream_audio(path_str):
                        processed_any |= self._ingest_audio_segment(chunk)

                if processed_any:
                    n_files += 1
            except Exception as exc:
                print(f"Error processing {path}: {exc}")

        if not self.codedb_segments:
            self.codedb = None
            self.db = None
            self.db_flat = None
            self.db_flat_device = None
            return {"message": "No audio processed. Please verify the input files."}

        self._finalize_codebook()
        codebook_size = 0 if self.db is None else self.db.shape[0]

        return {"message": f"Done! {n_files} files processed. Codebook grains: {codebook_size}."}
 
    def morph_audio(self, target_file):
        target_path = self._resolve_file_entry(target_file)
        if target_path is None or not target_path.exists():
            return self.sample_rate, np.zeros(1024, dtype=np.int16)

        if self.db is None or self.db.numel() == 0:
            return self.sample_rate, np.zeros(1024, dtype=np.int16)

        print("Creating codes for target audio")
        target_segments = []

        for chunk in self._stream_audio(str(target_path)):
            _, quantized = self.encode(chunk)
            if quantized is not None and quantized.shape[-1] >= self.unit:
                target_segments.append(quantized)

        if not target_segments:
            return self.sample_rate, np.zeros(1024, dtype=np.int16)

        target_codes = torch.cat(target_segments, dim=-1)

        if target_codes.shape[-1] < self.unit:
            return self.sample_rate, np.zeros(1024, dtype=np.int16)

        reconstructed = torch.zeros_like(target_codes)

        # produce stereo when needed
        if reconstructed.shape[0] == 1:
            reconstructed = torch.vstack([reconstructed, reconstructed])

        print("Matching grains...")
        for i in tqdm(range(0, target_codes.shape[-1], self.stride)):
            target_code = target_codes[:, :, i : i + self.unit]
            if target_code.shape[-1] != self.unit:
                continue

            distances = self._compute_distances(target_code)
            if distances.numel() == 0:
                continue

            temperature = max(self.temperature, 1e-4)
            logits = torch.nan_to_num(-distances / temperature, nan=-1e9, posinf=-1e9, neginf=1e9)
            probabilities = torch.softmax(logits, dim=0)

            if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
                probabilities = torch.full_like(distances, 1.0 / distances.numel())

            idx = torch.multinomial(probabilities, num_samples=1).item()
            min_distance = distances.min().item()

            code_closest = self.db[idx]
            if min_distance > self.threshold:
                code_closest = target_code.squeeze(0)

            code_closest = code_closest.to(reconstructed.dtype)
            end = min(i + self.unit, reconstructed.shape[-1])
            span = end - i
            code_slice = code_closest[:, :span]

            for channel in range(reconstructed.shape[0]):
                reconstructed[channel, :, i:end] = code_slice

        print("Decoding reconstructed/morphed audio...")
        decoded = self.decode(reconstructed)

        if decoded is None or getattr(decoded, "audio_values", None) is None:
            return self.sample_rate, np.zeros(1024, dtype=np.int16)

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
            # Heuristic: treat leading axis as channels when it is the smaller dimension.
            prepared = np.moveaxis(final_np, 0, -1)
        else:
            prepared = final_np

        clipped = np.clip(prepared, -1.0, 1.0)
        scaled = (clipped * 32767).astype(np.int16, copy=False)

        return self.sample_rate, scaled


synth = LatentGranularSynthesis()

def build_dataset(files, aug_checkbox):
    return synth.build_dataset(files, aug_checkbox)

def morph_audio(target_file):
    return synth.morph_audio(target_file)

def temperature(temperature, threshold):
    return synth.set_temperature(temperature, threshold)

def unit(unit, stride):
    return synth.set_unit(unit, stride)


def _build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("**Step 1:** Upload and process source sounds before morphing a target clip.")
        with gr.Row():
            with gr.Column():
                db_file = gr.File(file_count="multiple", label="Source Sounds")
                aug_checkbox = gr.Checkbox(label="Apply Augmentation")
                b1 = gr.Button("Process source sounds")
                text = gr.Textbox(label="Result")

            with gr.Column():
                target_file = gr.File(label="Target sound")

                with gr.Row():
                    temp_slider = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
                    threshold_slider = gr.Slider(0.1, 2.0, value=1.0, label="Threshold")
                with gr.Row():
                    unit_slider = gr.Slider(1, 10, value=2, step=1, label="Unit Size")
                    stride_slider = gr.Slider(1, 10, value=2, step=1, label="Stride")

                b2 = gr.Button("Morph Audio")
                audioplayer = gr.Audio(label="Output")

        temp_slider.change(temperature, inputs=[temp_slider, threshold_slider])
        threshold_slider.change(temperature, inputs=[temp_slider, threshold_slider])
        unit_slider.change(unit, inputs=[unit_slider, stride_slider])
        stride_slider.change(unit, inputs=[unit_slider, stride_slider])

        b1.click(build_dataset, inputs=[db_file, aug_checkbox], outputs=text)
        b2.click(morph_audio, inputs=target_file, outputs=audioplayer)

    return demo


def main():
    demo = _build_demo()
    demo.launch(show_error=True)


if __name__ == "__main__":
    main()
