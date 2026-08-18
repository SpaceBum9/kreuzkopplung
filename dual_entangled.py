from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    """Configuration and mathematical constraints of the reduced system."""

    epsilon: float = 1e-6
    weight_update_rate: float = 0.15
    initial_coupling: float = 0.30
    min_coupling: float = 0.05
    max_coupling: float = 0.95
    target_relative_divergence: float = 0.25
    coupling_adaptation_rate: float = 0.05
    max_coupling_step: float = 0.02
    divergence_deadband: float = 1e-3

    def validate(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and > 0")
        if not 0.0 < self.weight_update_rate <= 1.0:
            raise ValueError("weight_update_rate must be in (0, 1]")
        if not 0.0 <= self.min_coupling < self.max_coupling <= 1.0:
            raise ValueError("must satisfy: 0 <= min_coupling < max_coupling <= 1")
        if not self.min_coupling <= self.initial_coupling <= self.max_coupling:
            raise ValueError("initial_coupling must lie within coupling limits")
        if (
            not np.isfinite(self.target_relative_divergence)
            or self.target_relative_divergence < 0.0
        ):
            raise ValueError("target_relative_divergence must be finite and >= 0")
        if (
            not np.isfinite(self.coupling_adaptation_rate)
            or self.coupling_adaptation_rate <= 0.0
        ):
            raise ValueError("coupling_adaptation_rate must be finite and > 0")
        if not np.isfinite(self.max_coupling_step) or self.max_coupling_step <= 0.0:
            raise ValueError("max_coupling_step must be finite and > 0")
        if not np.isfinite(self.divergence_deadband) or self.divergence_deadband < 0.0:
            raise ValueError("divergence_deadband must be finite and >= 0")


class DualEntangledSystem:
    """Adaptive, classically constrained two-channel system.

    Processes scalar inputs step by step, computes proportional
    weights and applies a cross-coupling matrix inside a control loop.
    """

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig()
        self.config.validate()
        self.prev_weights = np.array([0.5, 0.5], dtype=float)
        self.rules: dict[str, Any] = {}
        self.reset()

    @property
    def coupling(self) -> float:
        """Current operator coupling factor."""
        return float(self.rules["coupling_factor"])

    def reset(self) -> None:
        """Reset weights, state and telemetry."""
        self.prev_weights = np.array([0.5, 0.5], dtype=float)
        self.rules = {
            "absolute_divergence": 0.0,
            "relative_divergence": 0.0,
            "target_relative_divergence": float(self.config.target_relative_divergence),
            "coupling_used": float(self.config.initial_coupling),
            "coupling_factor": float(self.config.initial_coupling),
            "coupling_delta": 0.0,
            "error": 0.0,
            "weights": [0.5, 0.5],
            "system_phase": "INIT",
        }

    @staticmethod
    def _as_finite_scalar(value: float) -> float:
        array = np.asarray(value, dtype=float)
        if array.ndim != 0:
            raise TypeError("input must be a scalar, not an array")
        scalar = float(array)
        if not np.isfinite(scalar):
            raise ValueError("input must be finite")
        return scalar

    @staticmethod
    def mirror_invert(state_a: float) -> float:
        """Mirror-invert and squash into [-1, 1]."""
        return float(-np.tanh(state_a))

    def _calculate_weights(self, state_a: float, state_b: float) -> np.ndarray:
        """Inverse-proportional weights without singularity."""
        epsilon = self.config.epsilon
        denominator = abs(state_a) + abs(state_b) + 2.0 * epsilon
        target_weights = np.array(
            [
                (abs(state_a) + epsilon) / denominator,
                (abs(state_b) + epsilon) / denominator,
            ],
            dtype=float,
        )
        rate = self.config.weight_update_rate
        weights = (1.0 - rate) * self.prev_weights + rate * target_weights
        weights /= float(np.sum(weights))
        self.prev_weights = weights
        return weights

    def _adapt_coupling(
        self,
        state_a: float,
        state_b: float,
        entangled: np.ndarray,
        coupling_used: float,
        weights: np.ndarray,
    ) -> dict[str, Any]:
        """Regulate coupling toward the target relative divergence."""
        absolute_divergence = float(abs(entangled[0] - entangled[1]))
        input_scale = max(abs(state_a), abs(state_b), self.config.epsilon)
        relative_divergence = absolute_divergence / input_scale
        error = relative_divergence - self.config.target_relative_divergence
        raw_delta = self.config.coupling_adaptation_rate * error
        coupling_delta = float(
            np.clip(
                raw_delta,
                -self.config.max_coupling_step,
                self.config.max_coupling_step,
            )
        )
        coupling_next = float(
            np.clip(
                coupling_used + coupling_delta,
                self.config.min_coupling,
                self.config.max_coupling,
            )
        )
        if abs(error) <= self.config.divergence_deadband:
            phase = "STABLE"
        elif coupling_next > coupling_used:
            phase = "COUPLING_UP"
        elif coupling_next < coupling_used:
            phase = "COUPLING_DOWN"
        elif error > 0.0:
            phase = "MAX_COUPLING_LIMIT"
        else:
            phase = "MIN_COUPLING_LIMIT"
        return {
            "absolute_divergence": absolute_divergence,
            "relative_divergence": float(relative_divergence),
            "target_relative_divergence": float(self.config.target_relative_divergence),
            "coupling_used": float(coupling_used),
            "coupling_factor": coupling_next,
            "coupling_delta": float(coupling_next - coupling_used),
            "error": float(error),
            "weights": [float(weights[0]), float(weights[1])],
            "system_phase": phase,
        }

    def step(self, state_a: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one evaluation step of the invert."""
        state_a = self._as_finite_scalar(state_a)
        state_b = self.mirror_invert(state_a)
        weights = self._calculate_weights(state_a, state_b)
        coupling_used = self.coupling
        coupling_matrix = np.array(
            [
                [weights[0], coupling_used * weights[1]],
                [coupling_used * weights[0], weights[1]],
            ],
            dtype=float,
        )
        states = np.array([state_a, state_b], dtype=float)
        entangled = coupling_matrix @ states
        self.rules = self._adapt_coupling(
            state_a,
            state_b,
            entangled,
            coupling_used,
            weights,
        )
        return entangled, self.rules.copy()

    def run(
        self,
        inputs: Iterable[float],
        *,
        reset: bool = True,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Evaluate the system over a sequence of scalar inputs.

        Each value is one ``step``. Internal state (weights, coupling)
        carries across the batch so the controller can adapt. Pass
        ``reset=False`` to continue a previous run without reinitialising.
        """
        if reset:
            self.reset()

        try:
            iterator = iter(inputs)
        except TypeError as exc:
            raise TypeError("inputs must be an iterable of finite scalars") from exc

        outputs: list[np.ndarray] = []
        telemetry: list[dict[str, Any]] = []
        for value in iterator:
            entangled, rules = self.step(value)
            outputs.append(np.asarray(entangled, dtype=float))
            telemetry.append(rules)

        if not outputs:
            return np.empty((0, 2), dtype=float), []
        return np.vstack(outputs), telemetry


def _sine_probe(n: int) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * np.pi, n, dtype=float)
    return np.sin(t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive two-channel DualEntangledSystem")
    parser.add_argument(
        "values",
        nargs="*",
        type=float,
        help="scalar inputs (omit to run a sine probe)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="length of the default sine probe (default: 64)",
    )
    args = parser.parse_args()
    system = DualEntangledSystem()
    inputs = np.asarray(args.values, dtype=float) if args.values else _sine_probe(args.steps)
    outputs, telemetry = system.run(inputs)
    last = telemetry[-1] if telemetry else None
    print(f"n={len(telemetry)}")
    if last is not None:
        print(f"last_phase={last['system_phase']}")
        print(f"last_coupling={system.coupling:.4f}")
    print(outputs)


if __name__ == "__main__":
    main()
