"""Hand-calculated metrics, independent fit optimum, selection and leakage checks."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fit_policies import fit_policies
from src.calibration import fit_temperature
from src.data import sha256_json, sha256_text_file, verify_manifest_directory
from src.metrics import (classification_metrics, confidence_from_logits, expected_calibration_error,
                         log_probabilities, negative_log_likelihood)
from src.policy_artifacts import (current_provenance, load_checked_logits, load_frozen_policy,
                                  read_json, write_json)
from src.rejection import (THRESHOLD_GRID, accept_mask, apply_policy, evaluate_policies,
                          matched_coverage_diagnostic, select_threshold, sensitivity_analysis)


@pytest.fixture
def fixture_logits():
    return np.log([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.9, 0.1]]), np.array([0, 0, 1, 1])


def test_hand_calculated_metrics(fixture_logits):
    z, y = fixture_logits
    row = classification_metrics(z, y, accepted=np.array([True, False, True, True]))
    assert row["n"] == 4 and row["correct"] == 2
    assert row["full_set_accuracy"] == 0.5
    assert row["coverage"] == 0.75 and row["rejection_rate"] == 0.25
    assert row["accepted_count"] == 3 and row["correct_accepted_count"] == 2
    assert row["accepted_accuracy"] == pytest.approx(2 / 3)
    assert row["selective_risk"] == pytest.approx(1 / 3)
    assert row["nll"] == pytest.approx(-np.log(0.8 * 0.4 * 0.7 * 0.1) / 4)
    assert row["ece"] == pytest.approx(0.5)
    assert row["per_class"][0]["n"] == 2
    assert row["per_class"][0]["coverage"] == 0.5


def test_ece_exact_boundaries_one_and_empty_bins():
    ece = expected_calibration_error(np.array([0.0, 1 / 15, 1.0]), np.array([True, True, False]))
    assert [r["count"] for r in ece["bins"]] == [1, 1] + [0] * 12 + [1]
    assert ece["ece"] == pytest.approx((1 + 14 / 15 + 1) / 3)
    assert ece["bins"][2]["accuracy"] is None


def test_rejection_keeps_full_set_probability_metrics(fixture_logits):
    z, y = fixture_logits
    full = classification_metrics(z, y)
    zero = classification_metrics(z, y, accepted=np.zeros(4, dtype=bool))
    assert full["accepted_accuracy"] == full["full_set_accuracy"]
    assert zero["coverage"] == 0 and zero["rejection_rate"] == 1
    assert zero["accepted_accuracy"] is None and zero["selective_risk"] is None
    assert zero["nll"] == full["nll"] and zero["ece"] == full["ece"]
    json.dumps(zero, allow_nan=False)


def test_missing_class_has_null_metrics():
    row = classification_metrics(np.zeros((2, 3)), np.array([0, 0]))["per_class"][2]
    assert row["n"] == 0 and row["coverage"] is None and row["accepted_accuracy"] is None


def test_log_softmax_extremes_and_offset():
    z = np.array([[1000.0, -1000.0], [1e12, 1e12]])
    assert negative_log_likelihood(z, np.array([1, 0])) == pytest.approx((2000 + np.log(2)) / 2)
    assert np.exp(log_probabilities(z)).sum(axis=1) == pytest.approx([1, 1])
    assert confidence_from_logits(z) == pytest.approx([1, 0.5])


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_invalid_temperature(temperature, fixture_logits):
    with pytest.raises(ValueError):
        confidence_from_logits(fixture_logits[0], temperature)


@pytest.mark.parametrize("logits,labels", [
    (np.empty((0, 2)), np.array([], dtype=int)),
    (np.array([[np.nan, 0]]), np.array([0])),
    (np.zeros((2, 2)), np.array([0.5, 1.0])),
    (np.zeros((2, 2)), np.array([0, 2])),
    (np.zeros((2, 2)), np.array([0, -1])),
    (np.zeros((2, 2)), np.array([0])),
])
def test_invalid_metric_inputs(logits, labels):
    with pytest.raises(ValueError):
        classification_metrics(logits, labels)


@pytest.mark.parametrize("conf,correct", [
    (np.array([1.1]), np.array([True])), (np.array([-0.1]), np.array([True])),
    (np.array([np.nan]), np.array([True])), (np.array([]), np.array([], dtype=bool)),
    (np.array([0.5]), np.array([1])),
])
def test_invalid_ece_inputs(conf, correct):
    with pytest.raises(ValueError):
        expected_calibration_error(conf, correct)


def test_fit_known_independent_optimum():
    # Four identical margins, 3/4 correct: optimal p(true predicted class)=.75,
    # so log(9)/T = log(3), giving T=2 without consulting the optimizer.
    z = np.tile([np.log(9), 0.0], (4, 1))
    fit = fit_temperature(z, np.array([0, 0, 0, 1]), split="calibration")
    assert fit["temperature"] == pytest.approx(2.0, abs=1e-6)
    assert fit["objective_after"] == pytest.approx(-0.75 * np.log(0.75) - 0.25 * np.log(0.25))
    assert fit["objective_after"] < fit["objective_before"]
    assert fit["converged"] and fit["argmax_preserved"]


@pytest.mark.parametrize("split", ["training", "model_validation", "policy_selection", "final_test"])
def test_fit_rejects_wrong_split(split, fixture_logits):
    with pytest.raises(ValueError, match="calibration split"):
        fit_temperature(*fixture_logits, split=split)


def test_fit_exhaustion_and_boundary_fall_back():
    z, y = np.tile([np.log(9), 0.0], (4, 1)), np.array([0, 0, 0, 1])
    exhausted = fit_temperature(z, y, split="calibration", max_iterations=1)
    assert exhausted["temperature"] == 1 and not exhausted["converged"]
    assert exhausted["failure_reason"] == "maximum_iterations_reached"
    # Every row is wrong: the optimum is T -> infinity, outside the declared bounds.
    boundary = fit_temperature(z, np.ones(4, dtype=int), split="calibration")
    assert boundary["temperature"] == 1 and boundary["status"] == "fallback_T1"
    assert boundary["failure_reason"] == "optimum_at_search_boundary"


def test_numerically_invalid_fit_falls_back(monkeypatch, fixture_logits):
    from src import calibration
    original = calibration.negative_log_likelihood

    def fail_after_baseline(logits, labels, temperature=1.0):
        if temperature != 1:
            raise FloatingPointError("probe")
        return original(logits, labels)

    monkeypatch.setattr(calibration, "negative_log_likelihood", fail_after_baseline)
    fit = fit_temperature(*fixture_logits, split="calibration")
    assert fit["temperature"] == 1 and not fit["converged"]
    assert "numerically_invalid" in fit["failure_reason"]


def test_worsened_objective_falls_back(monkeypatch, fixture_logits):
    from src import calibration
    original = calibration.negative_log_likelihood

    def worse(logits, labels, temperature=1.0):
        baseline = original(logits, labels)
        return baseline if temperature == 1 else baseline + 1 + np.log(temperature) ** 2

    monkeypatch.setattr(calibration, "negative_log_likelihood", worse)
    fit = fit_temperature(*fixture_logits, split="calibration")
    assert fit["status"] == "fallback_T1" and fit["failure_reason"] == "objective_worsened"
    assert fit["objective_after"] == fit["objective_before"]


def test_shared_metrics_match_saved_baseline():
    from src.predict import condition_a_report
    with np.load(ROOT / "artifacts/logits/model_validation_clean.npz") as data:
        z, y = data["logits"], data["labels"]
    baseline = read_json(ROOT / "artifacts/baseline/development_baseline.json")
    shared = classification_metrics(z, y, class_order=baseline["class_order"])
    legacy = condition_a_report(z, y, baseline["class_order"])
    for key in ("full_set_accuracy", "nll", "ece", "coverage", "accepted_accuracy"):
        assert shared[key] == pytest.approx(legacy[key], abs=1e-12)


def test_threshold_boundary_and_selection_tie():
    assert accept_mask(np.array([0.8, 0.799999, 1.0]), 0.8).tolist() == [True, False, True]
    result = select_threshold(np.array([0.1, 0.2, 0.2, 0.8, 0.9]),
                              np.array([False, True, True, True, True]), split="policy_selection")
    assert result["threshold"] == 0.11
    assert result["accepted_count"] == 4 and result["coverage"] == 0.8
    assert result["feasible"] and result["accepted_accuracy"] == 1


def test_infeasible_and_zero_accepted_fallback():
    result = select_threshold(np.array([0.1, 0.7]), np.array([False, False]), split="policy_selection")
    assert not result["feasible"] and result["threshold"] == 0.8
    assert result["status"] == "infeasible_fallback_tradeoff"
    assert result["accepted_accuracy"] is None and result["accepted_count"] == 0


@pytest.mark.parametrize("split", ["calibration", "model_validation", "final_test"])
def test_selection_rejects_wrong_split(split):
    with pytest.raises(ValueError, match="policy_selection"):
        select_threshold(np.array([0.9]), np.array([True]), split=split)


def test_a_to_d_argmax_and_all_accepted_equivalence(fixture_logits):
    z, y = fixture_logits
    comparison = evaluate_policies(z, y, temperature=2, raw_threshold=0, calibrated_threshold=0)
    rows = comparison["metrics"]
    assert len({r["full_set_accuracy"] for r in rows}) == 1
    assert rows[0]["nll"] == rows[1]["nll"]
    assert rows[2]["nll"] == rows[3]["nll"]
    for row in rows:
        assert row["coverage"] == 1 and row["accepted_accuracy"] == row["full_set_accuracy"]
    for decision in comparison["decisions"].values():
        assert decision["latent_predictions"].tolist() == z.argmax(1).tolist()
    decision = apply_policy(z, condition="D", temperature=2, raw_threshold=0, calibrated_threshold=1)
    assert decision["output"] == ["REJECT"] * 4
    assert decision["latent_predictions"].tolist() == z.argmax(1).tolist()


def test_sensitivity_grid_and_matched_ties(fixture_logits):
    z, y = fixture_logits
    result = sensitivity_analysis(z, y, ["d", "c", "b", "a"], temperature=2)
    assert len(THRESHOLD_GRID) == 101 and THRESHOLD_GRID[0] == 0 and THRESHOLD_GRID[-1] == 1
    for name in ("raw", "calibrated"):
        sweep = result[name]["threshold_sweep"]
        assert len(sweep) == 101
        assert sweep[0]["coverage"] == 1 and sweep[-1]["accepted_accuracy"] is None
    conf, ok, ids = np.array([0.7, 0.7, 0.7]), np.array([False, True, True]), ["c", "a", "b"]
    row = matched_coverage_diagnostic(conf, ok, ids)[0]
    assert row["accepted_count"] == 2 and row["accepted_accuracy"] == 1
    assert row["coverage"] == pytest.approx(2 / 3)
    perm = [2, 0, 1]
    reordered = matched_coverage_diagnostic(conf[perm], ok[perm], [ids[i] for i in perm])[0]
    assert reordered == row


@pytest.fixture
def repo_copy(tmp_path):
    for folder in ("configs", "docs", "artifacts/splits", "artifacts/model", "artifacts/logits"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    return tmp_path


def test_committed_artifacts_fit_and_reload_reproduce(repo_copy):
    before = (repo_copy / "artifacts/model/linear_head.npz").read_bytes()
    first = fit_policies(root=repo_copy)
    frozen = load_frozen_policy(repo_copy)
    assert frozen == first
    assert first["locked_before_final_test"] and not first["final_test_evaluated"]
    assert first["raw"]["feasible"] and first["calibrated"]["feasible"]
    second = fit_policies(root=repo_copy, output_dir=repo_copy / "reproduced")
    assert second["temperature"] == first["temperature"]
    assert second["raw"] == first["raw"] and second["calibrated"] == first["calibrated"]
    assert before == (repo_copy / "artifacts/model/linear_head.npz").read_bytes()


@pytest.mark.parametrize("mutation", ["hash", "split", "labels", "ids", "checkpoint", "corruption"])
def test_fit_rejects_corrupted_inputs_before_fitting(repo_copy, monkeypatch, mutation):
    import scripts.fit_policies as fitting
    path = repo_copy / "artifacts/logits/policy_selection_clean.npz"
    meta_path = path.with_suffix(".json")
    meta = read_json(meta_path)
    if mutation == "hash":
        meta["npz_sha256"] = "0" * 64
    elif mutation == "split":
        meta["split"] = "final_test"
    elif mutation == "checkpoint":
        meta["checkpoint_sha256"] = "0" * 64
    elif mutation == "corruption":
        meta["corruption"] = "jpeg"
    else:
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        if mutation == "labels":
            arrays["labels"][0] = (arrays["labels"][0] + 1) % 10
        else:
            arrays["source_ids"][0] = "train:0"
        np.savez(path, **arrays)
        from src.data import sha256_file
        meta["npz_sha256"] = sha256_file(path)
    write_json(meta_path, meta)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("unverified input reached calibration optimizer")

    monkeypatch.setattr(fitting, "fit_temperature", forbidden_fit)
    with pytest.raises(ValueError):
        fit_policies(root=repo_copy)
    assert not (repo_copy / "artifacts/temperature.json").exists()


@pytest.mark.parametrize("field", ["temperature", "config", "checkpoint"])
def test_frozen_policy_detects_drift(repo_copy, field):
    fit_policies(root=repo_copy)
    if field == "temperature":
        path = repo_copy / "artifacts/temperature.json"
        payload = read_json(path)
        payload["temperature"] = 2.0
        write_json(path, payload)
    elif field == "config":
        with (repo_copy / "configs/evaluation.yaml").open("a", encoding="utf-8") as stream:
            stream.write("\nextra_changed_setting: true\n")
    else:
        with (repo_copy / "artifacts/model/linear_head.npz").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        load_frozen_policy(repo_copy)


def test_telman_leakage_check_even_with_recomputed_hashes(repo_copy):
    """Review: maintain balance and fully repair hashes; actual overlap must fail."""
    folder = repo_copy / "artifacts/splits"
    training = read_json(folder / "training.json")
    calibration = read_json(folder / "calibration.json")
    leaked = copy.deepcopy(calibration["records"][0])
    leaked["split"] = "training"
    index = next(i for i, row in enumerate(training["records"]) if row["label"] == leaked["label"])
    training["records"][index] = leaked
    training["ids"] = [r["source_id"] for r in training["records"]]
    training["sha256"] = sha256_json(training["ids"])
    write_json(folder / "training.json", training)
    master = read_json(folder / "manifest.json")
    master["split_ids"]["training"] = training["ids"]
    master["split_file_hashes"]["training"] = sha256_text_file(folder / "training.json")
    master["manifest_sha256"] = sha256_json({key: master[key] for key in ("seed", "class_order", "split_ids", "counts")})
    write_json(folder / "manifest.json", master)
    with pytest.raises(AssertionError, match="overlap"):
        verify_manifest_directory(folder)


@pytest.mark.parametrize("split,corruption,severity", [
    ("model_validation", "jpeg", "severe"), ("final_test", "clean", "none"),
])
def test_analyzer_handoff_and_final_guard(repo_copy, monkeypatch, split, corruption, severity):
    """Synthetic logits exercise reporting; no final-test image inference occurs."""
    import scripts.analyze_policy_logits as analysis
    from src.data import sha256_file

    fit_policies(root=repo_copy)
    provenance, manifest = current_provenance(repo_copy)
    records = manifest["splits"][split]
    path = repo_copy / "synthetic_logits.npz"
    logits = np.zeros((len(records), 10))
    logits[:, 0] = 1.0
    np.savez(path, logits=logits, labels=np.array([r["label"] for r in records]),
             source_ids=np.array([r["source_id"] for r in records]))
    write_json(path.with_suffix(".json"), {**provenance, "split": split, "corruption": corruption,
                                          "severity": severity, "final_test": split == "final_test",
                                          "n": len(records), "npz_sha256": sha256_file(path)})
    monkeypatch.setattr(analysis, "ROOT", repo_copy)
    monkeypatch.setattr(analysis, "load_frozen_policy", lambda: load_frozen_policy(repo_copy))
    monkeypatch.setattr(analysis, "current_provenance", lambda: current_provenance(repo_copy))
    from src.policy_artifacts import load_evaluation_config
    monkeypatch.setattr(analysis, "load_evaluation_config", lambda: load_evaluation_config(repo_copy / "configs/evaluation.yaml"))
    output = repo_copy / "analysis"
    if split == "final_test":
        with pytest.raises(ValueError, match="allow-final-test"):
            analysis.analyze_logits(path, output)
        assert not output.exists()
    before = (repo_copy / "artifacts/policy.json").read_bytes()
    report = analysis.analyze_logits(path, output, allow_final_test=split == "final_test")
    assert len(report["metrics"]) == 4
    assert report["corruption"] == corruption and report["severity"] == severity
    assert report["metrics"][0]["coverage"] == 1
    assert report["metrics"][1]["accepted_accuracy"] is None
    assert report["metrics"][3]["accepted_accuracy"] is None
    assert (output / "risk_coverage.png").is_file()
    import csv
    with (output / "per_image.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * len(records)
    assert rows[-1]["output"] == "REJECT" and rows[-1]["run_id"] == report["run_id"]
    assert before == (repo_copy / "artifacts/policy.json").read_bytes()


def test_all_ten_image_conditions_share_frozen_policies(fixture_logits):
    from src.corruptions import corruption_specs
    z, y = fixture_logits
    # Controlled numerical fixtures stand in for ten independently cached conditions.
    rows = []
    for i, spec in enumerate(corruption_specs()):
        comparison = evaluate_policies(z / (i + 1), y, temperature=2, raw_threshold=0.8, calibrated_threshold=0.9)
        rows.extend(comparison["metrics"])
        assert comparison["decisions"]["B"]["threshold"] == 0.8
        assert comparison["decisions"]["D"]["threshold"] == 0.9
        assert comparison["decisions"]["D"]["temperature"] == 2
        assert comparison["metrics"][0]["full_set_accuracy"] == comparison["metrics"][2]["full_set_accuracy"]
    assert len(rows) == 40
