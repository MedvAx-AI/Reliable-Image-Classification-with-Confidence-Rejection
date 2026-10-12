# Team and contribution evidence

User-supplied handles and initial assignments:

| Member | GitHub | Assigned work package | Course/full name |
|---|---|---|---|
| 1 | [Telman3000](https://github.com/Telman3000) | Data, splits and image corruptions | Telman Nuruzov |
| 2 | [LeoVesinML](https://github.com/LeoVesinML) | Model, baseline and reproducibility | Leo Vesin |
| 3 | [Mysteri0K1ng](https://github.com/Mysteri0K1ng) | Calibration, rejection and metrics | Nikita Shankin |
| 4 | [MedvAx-AI](https://github.com/MedvAx-AI) | Evaluation, failure analysis and demo | Aleksandr Medvedev |

These are work-plan owners. Recording a username here does not grant repository write access or establish a GitHub issue assignee. Repository owner coordinates collaborator access and role swaps at kickoff.

Record actual work below as it occurs. Do not copy planned assignments into the final report as if already completed.

| Date | Member | Implemented/analyzed/reviewed work | Commit / PR / result artifact | Interpretation or learning |
|---|---|---|---|---|
| 2026-10-04 | Telman3000 | Implemented CIFAR-10 download, seed-42 five-way stratified manifests, leakage checks, deterministic blur/noise/JPEG, prepare CLI, tests, data README, corruption sample panel | branch `telman/data-pipeline`; `src/data.py`, `src/corruptions.py`, `scripts/prepare_data.py`, `artifacts/splits/*`, `artifacts/corruption_samples/*` | Planned subset sizes are now executable and reproducible; teammates can load `Cifar10SplitDataset` without re-deriving splits |
| 2026-10-11 | LeoVesinML | Frozen ImageNet ResNet-18, ten-class head trained on the training split only, epoch chosen by model-validation NLL, development condition-A baseline, calibration and policy logits, runtime record, reload check. Corrected stale split-file hashes without changing IDs. Coding assistant used for the implementation. | branch `leo/frozen-resnet-baseline`; `src/model.py`, `src/train.py`, `src/predict.py`, `scripts/train_baseline.py`, `artifacts/model/linear_head.npz`, `artifacts/baseline/development_baseline.json` | Protocol defaults were not searched. Final test was not run. Epoch 30 had the lowest validation NLL, so the 30-epoch cap was kept. Same-device reload matched logits within 1.1e-5. |
| 2026-10-12 | LeoVesinML | Manifest verification now folds CR LF to LF; feature-cache reuse checks the npz hash and extraction provenance; inference timing includes image fetch. Remeasured full inference at 2.91 s. Coding assistant used. | follow-up on the baseline package | Issue #2 had been closed before these reproduction checks were in place. The 0.8645 model-validation accuracy is unchanged. |
| 2026-10-12 | Mysteri0K1ng | Implemented shared metrics, positive temperature fitting, A–D rejection, policy-only threshold selection/fallback, fixed-grid sensitivity and matched-coverage diagnostics; fitted frozen policies on committed clean logits, exported development evidence, reviewed actual-record leakage checks, wrote handoff/interpretation/Q&A. Coding assistant used. | branch `Mysteri0K1ng/calibration-rejection`; `src/{metrics,calibration,rejection,policy_artifacts}.py`, `configs/evaluation.yaml`, `scripts/{fit_policies,analyze_policy_logits}.py`, `tests/test_policies.py`, `artifacts/{temperature,policy}.json`, `artifacts/policy_development/*`, `reports/calibration_rejection.md` | T=0.95617; raw/calibrated tau=0.56/0.57 meet the policy-split target. Development B/D cover 88.5% at accepted accuracy 91.47%/91.36%; labels are unchanged. Final-test evaluation was subsequently completed by Issue #4 integration; this row records the earlier handoff. |
| 2026-10-12 | MedvAx-AI | Integrated all ten final-test image conditions and A-D metrics, independent NumPy/count verification, fresh environment, deterministic failure analysis, live model demo, final two-page summary, three slides and Q&A script. Codex assisted implementation and verification. | `scripts/evaluate_final.py`, `scripts/verify_results.py`, `results/*`, `demo.py`, `reports/technical_summary.pdf`, `reports/slides.pdf` | Clean D achieves 90.90% accepted accuracy at 87.90% coverage; corruption reveals confident accepted errors. Policies/checkpoint were not retuned. Human rehearsal/individual understanding remain unverified. |

Each person should have evidence of technical implementation, experimental reasoning, review/reproduction, writing and understanding. The final two-page summary needs a compact contribution statement for all four people; the repository log can retain detailed evidence.

The initial repository and plan were prepared with Codex assistance at the user's request. This planning assistance is not evidence that a student implemented or understands the final model. Record any further assistance accurately and follow applicable course guidance if supplied later.
