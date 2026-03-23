#!/usr/bin/env python3
"""Gradio dashboard for tuning/evaluation/system-health workflows."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "tools" / "evaluate_morphing.py"


def _run_shell(cmd: list[str]) -> str:
    rendered = " ".join(shlex.quote(c) for c in cmd)
    try:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    except Exception as exc:
        return f"$ {rendered}\n\nERROR: {exc}"

    out = f"$ {rendered}\n\n[exit={completed.returncode}]\n"
    if completed.stdout:
        out += f"\nSTDOUT\n{completed.stdout}\n"
    if completed.stderr:
        out += f"\nSTDERR\n{completed.stderr}\n"
    return out


def run_search(manifest, output_dir, runner_cmd, trials, codecs, ablations, clip_limit):
    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "search",
        "--manifest",
        manifest,
        "--output-dir",
        output_dir,
        "--runner-cmd",
        runner_cmd,
        "--trials",
        str(int(trials)),
        "--codecs",
        codecs,
        "--ablations",
        ablations,
        "--clip-limit",
        str(int(clip_limit)),
    ]
    log = _run_shell(cmd)
    results_path = Path(output_dir) / "search_results.json"
    preview = ""
    if results_path.exists():
        payload = json.loads(results_path.read_text())
        preview = json.dumps(payload.get("top_presets", []), indent=2)
    return log, preview


def run_evaluate(
    manifest,
    output_dir,
    runner_cmd,
    codecs,
    ablations,
    clip_limit,
    temperature,
    threshold,
    continuity,
    rvq_focus,
    unit,
    stride,
    top_k,
):
    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "evaluate",
        "--manifest",
        manifest,
        "--output-dir",
        output_dir,
        "--runner-cmd",
        runner_cmd,
        "--codecs",
        codecs,
        "--ablations",
        ablations,
        "--clip-limit",
        str(int(clip_limit)),
        "--temperature",
        str(float(temperature)),
        "--threshold",
        str(float(threshold)),
        "--continuity",
        str(float(continuity)),
        "--rvq-focus",
        str(float(rvq_focus)),
        "--unit",
        str(int(unit)),
        "--stride",
        str(int(stride)),
        "--top-k",
        str(int(top_k)),
    ]
    log = _run_shell(cmd)
    summary_path = Path(output_dir) / "reports" / "summary.json"
    preview = ""
    if summary_path.exists():
        preview = summary_path.read_text()
    return log, preview


def run_validate(manifest, output_dir, report_json):
    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "validate",
        "--manifest",
        manifest,
        "--output-dir",
        output_dir,
        "--report-json",
        report_json,
    ]
    log = _run_shell(cmd)
    preview = ""
    path = Path(report_json)
    if path.exists():
        preview = path.read_text()
    return log, preview


def run_export_presets(report_dir, output_json, top_n):
    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "export-presets",
        "--report-dir",
        report_dir,
        "--output-json",
        output_json,
        "--top-n",
        str(int(top_n)),
    ]
    log = _run_shell(cmd)
    preview = ""
    path = Path(output_json)
    if path.exists():
        preview = path.read_text()
    return log, preview


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Neural Morphing Evaluation Dashboard") as demo:
        gr.Markdown("# Neural Morphing Tuning + Evaluation Dashboard")
        gr.Markdown("Tune, evaluate, validate system health, and export ranked presets.")

        with gr.Tab("Tune"):
            manifest = gr.Textbox(label="Manifest JSON", value="artifacts/validation/manifest.json")
            output_dir = gr.Textbox(label="Output Directory", value="artifacts/eval_search")
            runner_cmd = gr.Textbox(
                label="Runner Command Template",
                value=(
                    "./.venv/bin/python tools/run_morph_ablation.py --codec {codec} --palette-manifest {palette_manifest} "
                    "--source {source} --output-wav {output_wav} --tokens-npy {tokens_npy} "
                    "--match-indices-npy {match_indices_npy} --latency-json {latency_json} "
                    "--matcher {matcher} --swap {swap} --temperature {temperature} --threshold {threshold} "
                    "--continuity {continuity} --rvq-focus {rvq_focus} --unit {unit} --stride {stride} "
                    "--top-k {top_k} --seed {seed}"
                ),
                lines=4,
            )
            with gr.Row():
                trials = gr.Number(label="Trials", value=12, precision=0)
                codecs = gr.Textbox(label="Codecs", value="dac,spectrostream")
                ablations = gr.Textbox(
                    label="Ablations",
                    value="greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group",
                )
                clip_limit = gr.Number(label="Clip Limit", value=0, precision=0)
            run_btn = gr.Button("Run Search")
            tune_log = gr.Textbox(label="Search Log", lines=16)
            tune_preview = gr.Code(label="Top Presets Preview", language="json")
            run_btn.click(
                run_search,
                inputs=[manifest, output_dir, runner_cmd, trials, codecs, ablations, clip_limit],
                outputs=[tune_log, tune_preview],
            )

        with gr.Tab("Evaluate"):
            eval_manifest = gr.Textbox(label="Manifest JSON", value="artifacts/validation/manifest.json")
            eval_output = gr.Textbox(label="Output Directory", value="artifacts/eval_run")
            eval_runner = gr.Textbox(label="Runner Command Template", value="", lines=4)
            with gr.Row():
                eval_codecs = gr.Textbox(label="Codecs", value="dac,spectrostream")
                eval_ablations = gr.Textbox(
                    label="Ablations",
                    value="greedy_full_layer,greedy_rvq_group,beam_full_layer,beam_rvq_group",
                )
                eval_clip_limit = gr.Number(label="Clip Limit", value=0, precision=0)
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.47, label="Temperature")
                threshold = gr.Slider(0.2, 1.6, value=0.55, label="Threshold")
                continuity = gr.Slider(0.0, 1.0, value=0.93, label="Continuity")
                rvq_focus = gr.Slider(0.0, 1.0, value=0.3, label="RVQ Focus")
            with gr.Row():
                unit = gr.Number(label="Unit", value=7, precision=0)
                stride = gr.Number(label="Stride", value=2, precision=0)
                top_k = gr.Number(label="Top-K", value=7, precision=0)
            eval_btn = gr.Button("Run Evaluate")
            eval_log = gr.Textbox(label="Evaluate Log", lines=16)
            eval_summary = gr.Code(label="Summary JSON", language="json")
            eval_btn.click(
                run_evaluate,
                inputs=[
                    eval_manifest,
                    eval_output,
                    eval_runner,
                    eval_codecs,
                    eval_ablations,
                    eval_clip_limit,
                    temperature,
                    threshold,
                    continuity,
                    rvq_focus,
                    unit,
                    stride,
                    top_k,
                ],
                outputs=[eval_log, eval_summary],
            )

        with gr.Tab("System Health"):
            val_manifest = gr.Textbox(label="Manifest JSON", value="artifacts/validation/manifest.json")
            val_output = gr.Textbox(label="Evaluate Output Directory", value="artifacts/eval_run")
            val_report = gr.Textbox(label="Validation Report JSON", value="artifacts/eval_run/reports/validation.json")
            val_btn = gr.Button("Run Validate")
            val_log = gr.Textbox(label="Validate Log", lines=16)
            val_json = gr.Code(label="Validation JSON", language="json")
            val_btn.click(run_validate, inputs=[val_manifest, val_output, val_report], outputs=[val_log, val_json])

        with gr.Tab("Presets"):
            report_dir = gr.Textbox(label="Report Directory", value="artifacts/eval_run/reports")
            presets_out = gr.Textbox(label="Export Presets JSON", value="artifacts/eval_run/reports/exported_presets.json")
            top_n = gr.Number(label="Top-N", value=8, precision=0)
            export_btn = gr.Button("Export Presets")
            export_log = gr.Textbox(label="Export Log", lines=16)
            export_json = gr.Code(label="Presets JSON", language="json")
            export_btn.click(run_export_presets, inputs=[report_dir, presets_out, top_n], outputs=[export_log, export_json])

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(show_error=True)


if __name__ == "__main__":
    main()
