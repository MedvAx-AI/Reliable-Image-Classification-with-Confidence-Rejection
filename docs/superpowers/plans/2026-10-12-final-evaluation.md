# Final evaluation and submission implementation plan

> For agentic workers: use superpowers:executing-plans to implement this existing team plan inline, with a final independent review.

**Goal:** Complete MedvAx-AI's Issue #4 technical deliverables with measured final results and reproducible submission artifacts.

**Architecture:** Reuse the committed classifier, split manifests and frozen policies. Infer each of ten image conditions once, validate ordered IDs/labels, compute all four policies from shared logits, and store complete provenance. Use those outputs for figures, deterministic failure selection, a local live-inference demo, and the final summary/slides.

**Tech stack:** Existing NumPy, PyTorch/torchvision, Pillow, matplotlib; standard-library local HTTP server for demo; ReportLab for submission PDFs.

**Spec:** `docs/EXPERIMENT_PROTOCOL.md`, MedvAx-AI section of `docs/TEAM_PLAN.md`, `docs/REQUIREMENTS.md`, original course PDF.

## Global constraints

- Preserve checkpoint, split IDs, T=0.9561687801607236, raw tau=0.56 and calibrated tau=0.57; no test-based fitting or tuning.
- Four policies x ten conditions x 2,000 final-test examples; primary NLL/ECE include rejected images.
- At least three actual failures with causes, evidence and limitations.
- Exactly two summary pages; at most three slides; five-minute demo script.
- Preserve user edits in the original checkout. No Cursor co-author trailers.
- Human course registration, actual rehearsal, supervised understanding and submission cannot be certified by code; record them honestly.

## Review focus

1. Swapped condition rows or mismatched source IDs must be rejected.
2. Cached logits must not bypass model, data, policy or software provenance checks.
3. Tests must not rewrite frozen fitting inputs in the working checkout.
4. Demo must use the saved weights and policies and must distinguish live inference from saved results.
5. Artifact regeneration must preserve numerical results without refitting policies.

## Task 1: Frozen evaluation and evidence

Files: `src/evaluate.py`, `src/plots.py`, `scripts/evaluate_final.py`, `scripts/verify_results.py`, `tests/test_evaluation.py`, `requirements-evaluation.txt`.
Consumes: existing model/data/policy APIs. Produces: validated condition bundles, 40 aggregate rows, 80,000 policy-decision rows, threshold sweeps, runtime/provenance manifest and figures.

- [x] Write failing tests for matrix alignment, policy population counts, missing/duplicate conditions, artifact tampering, and failure selection.
- [x] Run the tests and observe failure before implementing.
- [x] Implement strict validation, single-forward/shared-logit evaluation, checked cache reuse and figure export.
- [x] Run targeted checks, then the full suite in isolation; verify committed fitting inputs are unchanged.
- [x] Run actual frozen final inference and independently verify exported counts/metrics. Commit this task.

## Task 2: Failure analysis and live demonstration

Files: `src/failures.py`, `demo.py`, `demo/index.html`, `scripts/analyze_failures.py`, `tests/test_demo.py`, `reports/failure_analysis.md`, `reports/demo_script.md`.
Consumes: final condition bundles/decisions and saved model/policy. Produces: three deterministic real cases, clean/corrupt views, live image predictions and saved-output fallback.

- [x] Write failing behavior tests for case selection and demo inference responses.
- [x] Observe failure, then implement minimum data/model integration and the local UI.
- [x] Verify each selected case against saved rows and inspect images; test live HTTP inference against recorded predictions.
- [x] Document likely causes as hypotheses and provide all-team Q&A/rehearsal prompts. Commit this task.

## Task 3: Submission packaging and independent review

Files: `reports/technical_summary.md`, `reports/technical_summary.pdf`, `reports/slides.pdf`, `tools/render_final_submission.py`, README, contribution/completion documents.
Consumes: measured final artifacts. Produces: exactly two summary pages, three slides, exact run instructions and honest completion audit.

- [x] Reproduce cached outputs in a fresh environment without changing policy files; verify numbers independently.
- [x] Generate report/slides from recorded measurements and inspect every rendered page.
- [x] Run full tests, verify all required links/artifacts and request one independent whole-branch review.
- [ ] Fix substantive review findings, publish an attached PR and report any remaining human completion gates.
