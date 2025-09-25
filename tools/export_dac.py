#!/usr/bin/env python3
"""Export Descript Audio Codec (DAC) encoder/decoder to ONNX and TorchScript.

Usage:
    python tools/export_dac.py --output artefacts/dac --model descript/dac_44khz

The script produces:
    encoder.onnx / decoder.onnx
    encoder.ts   / decoder.ts
    embeddings.npy (concatenated codebook embeddings)
    metadata.json (model configuration for the C++ backend)
"""

import argparse
import json
import pathlib
from typing import List

import numpy as np
import torch
from transformers import DacModel


class DacEncoderWrapper(torch.nn.Module):
    """Thin wrapper exposing encode -> audio_codes with [B, T] output."""

    def __init__(self, model: DacModel):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3:
            raise ValueError("Expected waveform tensor with shape [batch, samples] or [batch, channels, samples]")

        outputs = self.model.encode(waveform)
        # audio_codes: [batch, num_codebooks, frames]
        codes = outputs.audio_codes
        return codes.squeeze(0)


class DacDecoderWrapper(torch.nn.Module):
    """Thin wrapper exposing decode(audio_codes) -> audio waveform."""

    def __init__(self, model: DacModel):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 3:
            raise ValueError("Expected tokens tensor with shape [num_codebooks, frames] or [batch, num_codebooks, frames]")

        tokens = tokens.long()
        outputs = self.model.decode(audio_codes=tokens)
        audio = outputs.audio_values  # [batch, channels, samples]
        # Ensure mono output to match plugin expectations.
        if audio.dim() == 3:
            audio = audio[:, :1, :]
        return audio.squeeze(0)


def collect_codebooks(model: DacModel) -> List[torch.Tensor]:
    codebooks = []
    if hasattr(model.quantizer, "codebooks"):
        for cb in model.quantizer.codebooks:
            if hasattr(cb, "embedding"):
                codebooks.append(cb.embedding.weight.detach().cpu())
    elif hasattr(model.quantizer, "embedding"):
        codebooks.append(model.quantizer.embedding.weight.detach().cpu())
    elif hasattr(model.quantizer, "codebook"):
        codebooks.append(model.quantizer.codebook.detach().cpu())

    if not codebooks:
        raise RuntimeError("Unable to locate DAC quantizer codebooks")

    return codebooks


def export_embeddings(codebooks: List[torch.Tensor], destination: pathlib.Path) -> dict:
    np_codebooks = [cb.numpy().astype(np.float32) for cb in codebooks]
    stacked = np.stack(np_codebooks, axis=0)
    np.save(destination / "embeddings.npy", stacked)
    stacked.tofile(destination / "embeddings.bin")
    return {
        "num_codebooks": stacked.shape[0],
        "codebook_size": stacked.shape[1],
        "embedding_dim": stacked.shape[2],
        "embedding_file": "embeddings.bin",
    }


def export_torchscript(module: torch.nn.Module, example_inputs: torch.Tensor, path: pathlib.Path) -> None:
    module_cpu = module.to("cpu").eval()
    with torch.no_grad():
        scripted = torch.jit.trace(module_cpu, example_inputs)
    scripted.save(str(path))


def export_onnx(module: torch.nn.Module, example_inputs: torch.Tensor, path: pathlib.Path, input_name: str, output_name: str, dynamic_axes: dict) -> None:
    module_cpu = module.to("cpu").eval()
    with torch.no_grad():
        torch.onnx.export(
            module_cpu,
            example_inputs,
            str(path),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DAC encoder/decoder for JUCE backend")
    parser.add_argument("--model", default="descript/dac_44khz", help="Hugging Face model id or local path")
    parser.add_argument("--output", required=True, help="Directory to write exported artefacts")
    parser.add_argument("--sample-length", type=int, default=44100, help="Dummy waveform length (samples) for tracing")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device used while loading the model")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--no-torchscript", action="store_true", help="Skip TorchScript export")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"Loading DAC model '{args.model}' on {device}")
    model = DacModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    sample_rate = int(getattr(model.config, "sampling_rate", 44100))

    encoder = DacEncoderWrapper(model)
    decoder = DacDecoderWrapper(model)

    codebooks = collect_codebooks(model)
    embedding_info = export_embeddings(codebooks, output_dir)

    dummy_wave = torch.zeros(1, args.sample_length, dtype=torch.float32, device=device)
    dummy_tokens = torch.zeros(embedding_info["num_codebooks"], args.sample_length // 256 + 1, dtype=torch.long, device=device)

    metadata = {
        "model": args.model,
        "sample_rate": sample_rate,
        **embedding_info,
    }

    if not args.no_torchscript:
        print("Exporting TorchScript modules")
        export_torchscript(encoder, dummy_wave, output_dir / "encoder.ts")
        export_torchscript(decoder, dummy_tokens, output_dir / "decoder.ts")

    if not args.no_onnx:
        print("Exporting ONNX modules")
        export_onnx(
            encoder,
            dummy_wave,
            output_dir / "encoder.onnx",
            input_name="waveform",
            output_name="tokens",
            dynamic_axes={"waveform": {1: "samples"}, "tokens": {1: "frames"}},
        )
        export_onnx(
            decoder,
            dummy_tokens,
            output_dir / "decoder.onnx",
            input_name="tokens",
            output_name="audio",
            dynamic_axes={"tokens": {1: "frames"}, "audio": {1: "samples"}},
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Export complete. Artefacts written to {output_dir}")


if __name__ == "__main__":
    main()
