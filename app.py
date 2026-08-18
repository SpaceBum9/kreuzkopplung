"""Hugging Face Space — Kreuzkopplung / DualEntangledSystem.run()."""

from __future__ import annotations

import numpy as np
import gradio as gr

from dual_entangled import DualEntangledSystem, SystemConfig


def make_signal(kind: str, n: int, amplitude: float, cycles: float) -> np.ndarray:
    n = int(max(8, n))
    t = np.linspace(0.0, 1.0, n, dtype=float)
    if kind == "Sinus":
        return amplitude * np.sin(2.0 * np.pi * cycles * t)
    if kind == "Rechteck":
        return amplitude * np.sign(np.sin(2.0 * np.pi * cycles * t) + 1e-12)
    if kind == "Sprung":
        return np.where(t < 0.35, 0.0, amplitude)
    if kind == "Rampe":
        return amplitude * (2.0 * t - 1.0)
    if kind == "Chirp":
        phase = 2.0 * np.pi * (0.4 * cycles * t + 3.2 * cycles * t * t)
        return amplitude * np.sin(phase)
    rng = np.random.default_rng(1)
    return amplitude * (2.0 * rng.random(n) - 1.0)


def run_lab(
    kind: str,
    n: int,
    amplitude: float,
    cycles: float,
    weight_update_rate: float,
    initial_coupling: float,
    min_coupling: float,
    max_coupling: float,
    target_rel: float,
    adapt_rate: float,
    max_step: float,
    deadband: float,
):
    lo, hi = sorted((float(min_coupling), float(max_coupling)))
    if hi - lo < 0.01:
        hi = min(1.0, lo + 0.01)
    initial = float(np.clip(initial_coupling, lo, hi))
    config = SystemConfig(
        weight_update_rate=float(weight_update_rate),
        initial_coupling=initial,
        min_coupling=lo,
        max_coupling=hi,
        target_relative_divergence=float(target_rel),
        coupling_adaptation_rate=float(adapt_rate),
        max_coupling_step=float(max_step),
        divergence_deadband=float(deadband),
    )
    inputs = make_signal(kind, n, amplitude, cycles)
    system = DualEntangledSystem(config)
    outputs, telemetry = system.run(inputs)
    mirror = np.array([system.mirror_invert(float(x)) for x in inputs])
    coupling = np.array([row["coupling_used"] for row in telemetry])
    rel = np.array([row["relative_divergence"] for row in telemetry])
    target = np.full_like(rel, config.target_relative_divergence)
    last = telemetry[-1]
    summary = (
        f"n={len(telemetry)}  phase={last['system_phase']}  "
        f"c={last['coupling_used']:.4f}  Δc={last['coupling_delta']:+.4f}  "
        f"err={last['error']:+.4f}  rel={last['relative_divergence']:.4f}"
    )
    waves = {
        "step": list(range(len(inputs))) * 4,
        "value": list(inputs) + list(mirror) + list(outputs[:, 0]) + list(outputs[:, 1]),
        "series": (
            ["A"] * len(inputs)
            + ["B = −tanh(A)"] * len(inputs)
            + ["out0"] * len(inputs)
            + ["out1"] * len(inputs)
        ),
    }
    control = {
        "step": list(range(len(inputs))) * 3,
        "value": list(coupling) + list(rel) + list(target),
        "series": (
            ["c"] * len(inputs)
            + ["rel. Div."] * len(inputs)
            + ["Soll"] * len(inputs)
        ),
    }
    return waves, control, summary


with gr.Blocks(title="Kreuzkopplung") as demo:
    gr.Markdown(
        "# Kreuzkopplung\n"
        "Klassischer Zwei-Kanal-Regler. `DualEntangledSystem.run(inputs)` "
        "hält Gewichte und Kopplung über die Sequenz.\n\n"
        "[GitHub](https://github.com/SpaceBum9/kreuzkopplung)"
    )
    with gr.Row():
        kind = gr.Dropdown(
            ["Sinus", "Rechteck", "Sprung", "Rampe", "Chirp", "Rauschen"],
            value="Sinus",
            label="Signal",
        )
        n = gr.Slider(32, 512, value=240, step=8, label="Schritte n")
        amplitude = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Amplitude")
        cycles = gr.Slider(0.25, 8.0, value=2.0, step=0.25, label="Zyklen")
    with gr.Accordion("SystemConfig", open=False):
        with gr.Row():
            weight_update_rate = gr.Slider(0.01, 1.0, value=0.15, step=0.01, label="weight_update_rate")
            initial_coupling = gr.Slider(0.05, 0.95, value=0.30, step=0.01, label="initial_coupling")
            min_coupling = gr.Slider(0.0, 0.9, value=0.05, step=0.01, label="min_coupling")
            max_coupling = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="max_coupling")
        with gr.Row():
            target_rel = gr.Slider(0.0, 1.5, value=0.25, step=0.01, label="target_relative_divergence")
            adapt_rate = gr.Slider(0.005, 0.4, value=0.05, step=0.005, label="coupling_adaptation_rate")
            max_step = gr.Slider(0.005, 0.2, value=0.02, step=0.005, label="max_coupling_step")
            deadband = gr.Slider(0.0, 0.1, value=0.001, step=0.001, label="divergence_deadband")
    run_btn = gr.Button("run(inputs)", variant="primary")
    summary = gr.Textbox(label="Letzter Schritt", interactive=False)
    waves = gr.LinePlot(x="step", y="value", color="series", title="Wellen")
    control = gr.LinePlot(x="step", y="value", color="series", title="Regler")

    inputs = [
        kind,
        n,
        amplitude,
        cycles,
        weight_update_rate,
        initial_coupling,
        min_coupling,
        max_coupling,
        target_rel,
        adapt_rate,
        max_step,
        deadband,
    ]
    run_btn.click(run_lab, inputs=inputs, outputs=[waves, control, summary])
    demo.load(run_lab, inputs=inputs, outputs=[waves, control, summary])


if __name__ == "__main__":
    demo.launch()
