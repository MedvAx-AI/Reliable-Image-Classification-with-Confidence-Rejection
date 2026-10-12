"""A-D decisions, policy-only selection, and descriptive sensitivity diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.metrics import classification_metrics, confidence_from_logits, selective_counts, validate_logits

THRESHOLD_GRID = tuple(i / 100 for i in range(101))


def accept_mask(confidences: np.ndarray, threshold: float) -> np.ndarray:
    conf = np.asarray(confidences, dtype=np.float64)
    if conf.ndim != 1 or not len(conf) or not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidences must be nonempty finite [N] in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite in [0, 1]")
    return conf >= threshold


def threshold_sweep(confidences: np.ndarray, correct: np.ndarray) -> list[dict[str, Any]]:
    """Fixed 0.00..1.00 grid. Test sweeps describe tradeoffs, never tune policies."""
    return [{"threshold": tau, **selective_counts(correct, accept_mask(confidences, tau))}
            for tau in THRESHOLD_GRID]


def select_threshold(
    confidences: np.ndarray, correct: np.ndarray, *, split: str,
    target_accepted_accuracy: float = 0.90, minimum_coverage: float = 0.20,
    fallback_threshold: float = 0.80,
) -> dict[str, Any]:
    if split != "policy_selection":
        raise ValueError("operating thresholds may only be selected on policy_selection")
    if not np.isfinite([target_accepted_accuracy, minimum_coverage]).all():
        raise ValueError("targets must be finite")
    if not 0 <= target_accepted_accuracy <= 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("accuracy target must be in [0,1] and coverage in (0,1]")
    fallback = selective_counts(correct, accept_mask(confidences, fallback_threshold))
    sweep = threshold_sweep(confidences, correct)
    feasible = [row for row in sweep if row["accepted_accuracy"] is not None
                and row["accepted_accuracy"] >= target_accepted_accuracy
                and row["coverage"] >= minimum_coverage]
    selected = min(feasible, key=lambda r: (-r["accepted_count"], r["threshold"])) if feasible else {
        "threshold": fallback_threshold, **fallback}
    return {**selected, "selection_split": split, "feasible": bool(feasible),
            "feasible_threshold_count": len(feasible),
            "status": "target_met_on_policy_split" if feasible else "infeasible_fallback_tradeoff",
            "target_accepted_accuracy": target_accepted_accuracy, "minimum_coverage": minimum_coverage,
            "fallback_threshold": fallback_threshold,
            "selection_rule": "maximum coverage; smallest threshold breaks ties",
            "grid": {"start": 0.0, "stop": 1.0, "step": 0.01, "count": 101}}


def apply_policy(
    logits: np.ndarray, *, condition: str, temperature: float,
    raw_threshold: float, calibrated_threshold: float,
) -> dict[str, Any]:
    """Keep latent predictions; return REJECT as the user-facing deferred output."""
    z = validate_logits(logits)
    if condition not in {"A", "B", "C", "D"}:
        raise ValueError("condition must be A, B, C, or D")
    raw = confidence_from_logits(z)
    calibrated = confidence_from_logits(z, temperature)
    active_temperature = 1.0 if condition in {"A", "B"} else temperature
    conf = raw if condition in {"A", "B"} else calibrated
    tau = raw_threshold if condition == "B" else calibrated_threshold if condition == "D" else None
    mask = np.ones(len(z), dtype=bool) if tau is None else accept_mask(conf, tau)
    latent = z.argmax(axis=1)
    return {"condition": condition, "temperature": active_temperature, "threshold": tau,
            "latent_predictions": latent, "raw_confidence": raw, "calibrated_confidence": calibrated,
            "confidence": conf, "accepted": mask,
            "output": [int(pred) if accepted else "REJECT" for pred, accepted in zip(latent, mask)]}


def evaluate_policies(
    logits: np.ndarray, labels: np.ndarray, *, temperature: float,
    raw_threshold: float, calibrated_threshold: float,
    n_bins: int = 15, class_order: list[str] | None = None,
) -> dict[str, Any]:
    rows, decisions = [], {}
    for condition in "ABCD":
        decision = apply_policy(logits, condition=condition, temperature=temperature,
                                raw_threshold=raw_threshold, calibrated_threshold=calibrated_threshold)
        metrics = classification_metrics(logits, labels, temperature=decision["temperature"],
                                         accepted=decision["accepted"], n_bins=n_bins, class_order=class_order)
        rows.append({"condition": condition, "temperature": decision["temperature"],
                     "threshold": decision["threshold"], **metrics})
        decisions[condition] = decision
    return {"metrics": rows, "decisions": decisions}


def matched_coverage_diagnostic(
    confidences: np.ndarray, correct: np.ndarray, source_ids: list[str],
    coverages: tuple[float, ...] = (0.50, 0.80, 1.00),
) -> list[dict[str, Any]]:
    """Retrospective top-k ranking, ties broken lexicographically by sample ID.

    Ceil(N * coverage) defines the requested population for nonintegral counts.
    The result reports actual coverage. It is not a deployable threshold.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    accept_mask(conf, 0)  # Validate confidence range independently of correctness.
    if len(source_ids) != len(conf) or len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique and aligned with confidence")
    order = np.lexsort((np.asarray(source_ids, dtype=str), -conf))
    rows = []
    for coverage in coverages:
        if not np.isfinite(coverage) or not 0 < coverage <= 1:
            raise ValueError("matched coverage must be in (0, 1]")
        mask = np.zeros(len(conf), dtype=bool)
        mask[order[:int(np.ceil(len(conf) * coverage))]] = True
        rows.append({"requested_coverage": coverage, **selective_counts(correct, mask),
                     "interpretation": "retrospective ranking diagnostic; not a deployable threshold",
                     "tie_break": "source_id lexicographic ascending", "rounding": "ceil(N * requested_coverage)"})
    return rows


def sensitivity_analysis(
    logits: np.ndarray, labels: np.ndarray, source_ids: list[str], *, temperature: float,
    matched_coverages: tuple[float, ...] = (0.50, 0.80, 1.00),
) -> dict[str, Any]:
    z = validate_logits(logits, labels)
    correct = z.argmax(axis=1) == labels
    result = {"interpretation": "descriptive sensitivity only; never select policies on final test"}
    for name, temp in (("raw", 1.0), ("calibrated", temperature)):
        confidence = confidence_from_logits(z, temp)
        result[name] = {"threshold_sweep": threshold_sweep(confidence, correct),
                        "matched_coverage": matched_coverage_diagnostic(
                            confidence, correct, source_ids, matched_coverages)}
    return result


def plot_risk_coverage(analysis: dict[str, Any], path: str, *, title: str) -> None:
    """Draw sweeps computed from shared clean or corrupted logits, without forwards."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    peak_risk = 0.0
    for name in ("raw", "calibrated"):
        rows = [r for r in analysis[name]["threshold_sweep"] if r["selective_risk"] is not None]
        if rows:
            peak_risk = max(peak_risk, max(r["selective_risk"] for r in rows))
        ax.plot([r["coverage"] for r in rows], [r["selective_risk"] for r in rows],
                label=name, linestyle="-" if name == "raw" else "--", linewidth=2)
    ax.set(xlabel="Coverage (accepted / N)", ylabel="Selective risk (errors / accepted)",
           title=title + "\nDescriptive fixed-threshold sweep; no policy tuning",
           xlim=(0, 1), ylim=(0, min(1.0, max(0.05, peak_risk * 1.2))))
    ax.legend()
    ax.grid(alpha=0.2)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
