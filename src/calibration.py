"""Deterministic scalar temperature fit; never consumes policy or test logits."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.metrics import negative_log_likelihood, validate_logits


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, *, split: str,
    log_temperature_bounds: tuple[float, float] = (-5.0, 5.0),
    tolerance: float = 1e-8, objective_tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> dict[str, Any]:
    """Minimize NLL over log(T), including the starting point log(T)=0.

    NLL is convex in inverse temperature, hence unimodal in log temperature.
    Golden-section search needs no gradients or extra optimization dependency.
    A boundary optimum, exhaustion, or invalid/worse objective explicitly falls
    back to T=1. Provenance of input artifacts is checked by the fitting CLI.
    """
    if split != "calibration":
        raise ValueError("temperature may only be fitted on the calibration split")
    z = validate_logits(logits, labels)
    lo, hi = map(float, log_temperature_bounds)
    if not np.isfinite([lo, hi]).all() or not -700 < lo < 0 < hi < 700:
        raise ValueError("finite log-temperature bounds must contain zero and lie in (-700, 700)")
    if not np.isfinite([tolerance, objective_tolerance]).all() or tolerance <= 0 or objective_tolerance < 0:
        raise ValueError("invalid optimizer tolerances")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    before = negative_log_likelihood(z, labels)
    record = {"split": split, "n": len(z), "optimizer": "golden_section_log_temperature",
              "initial_log_temperature": 0.0, "log_temperature_bounds": [lo, hi],
              "tolerance": tolerance, "objective_tolerance": objective_tolerance,
              "max_iterations": max_iterations, "objective": "full_set_mean_nll",
              "objective_before": before}
    ratio = (np.sqrt(5) - 1) / 2
    left, right = lo, hi
    iterations = 0
    evaluations = 1

    def objective(value: float) -> float:
        nonlocal evaluations
        evaluations += 1
        loss = negative_log_likelihood(z, labels, float(np.exp(value)))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite fitting objective")
        return loss

    reason = None
    candidate = 1.0
    candidate_nll = before
    try:
        a, b = right - ratio * (right - left), left + ratio * (right - left)
        fa, fb = objective(a), objective(b)
        for iterations in range(1, max_iterations + 1):
            if fa <= fb:
                right, b, fb = b, a, fa
                a = right - ratio * (right - left)
                fa = objective(a)
            else:
                left, a, fa = a, b, fb
                b = left + ratio * (right - left)
                fb = objective(b)
            if right - left <= tolerance:
                break
        best_log = (left + right) / 2
        candidate = float(np.exp(best_log))
        candidate_nll = objective(best_log)
        if right - left > tolerance:
            reason = "maximum_iterations_reached"
        elif min(best_log - lo, hi - best_log) <= 2 * tolerance:
            reason = "optimum_at_search_boundary"
        elif candidate_nll > before + objective_tolerance:
            reason = "objective_worsened"
        elif candidate_nll >= before:
            candidate, candidate_nll = 1.0, before
    except (FloatingPointError, OverflowError) as exc:
        reason = f"numerically_invalid: {exc}"
    temperature = candidate if reason is None else 1.0
    after = candidate_nll if reason is None else before
    return {**record, "temperature": temperature, "log_temperature": float(np.log(temperature)),
            "candidate_temperature": candidate, "candidate_objective": candidate_nll,
            "objective_after": after, "iterations": iterations, "evaluations": evaluations,
            "converged": reason is None, "status": "fitted" if reason is None else "fallback_T1",
            "failure_reason": reason,
            "argmax_preserved": bool(np.array_equal(z.argmax(1), (z / temperature).argmax(1)))}
