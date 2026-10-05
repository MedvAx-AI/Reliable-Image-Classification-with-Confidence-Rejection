# Team and contribution evidence

User-supplied handles and initial assignments:

| Member | GitHub | Assigned work package | Course/full name |
|---|---|---|---|
| 1 | [Telman3000](https://github.com/Telman3000) | Data, splits and image corruptions | To be entered by member |
| 2 | [LeoVesinML](https://github.com/LeoVesinML) | Model, baseline and reproducibility | To be entered by member |
| 3 | [Mysteri0K1ng](https://github.com/Mysteri0K1ng) | Calibration, rejection and metrics | To be entered by member |
| 4 | [MedvAx-AI](https://github.com/MedvAx-AI) | Evaluation, failure analysis and demo | To be entered by member |

These are work-plan owners. Recording a username here does not grant repository write access or establish a GitHub issue assignee. Repository owner coordinates collaborator access and role swaps at kickoff.

Record actual work below as it occurs. Do not copy planned assignments into the final report as if already completed.

| Date | Member | Implemented/analyzed/reviewed work | Commit / PR / result artifact | Interpretation or learning |
|---|---|---|---|---|
| 2026-10-04 | Telman3000 | Implemented CIFAR-10 download, seed-42 five-way stratified manifests, leakage checks, deterministic blur/noise/JPEG, prepare CLI, tests, data README, corruption sample panel | branch `telman/data-pipeline`; `src/data.py`, `src/corruptions.py`, `scripts/prepare_data.py`, `artifacts/splits/*`, `artifacts/corruption_samples/*` | Planned subset sizes are now executable and reproducible; teammates can load `Cifar10SplitDataset` without re-deriving splits |
| Pending | LeoVesinML / Mysteri0K1ng / MedvAx-AI | Remaining work packages | — | — |

Each person should have evidence of technical implementation, experimental reasoning, review/reproduction, writing and understanding. The final two-page summary needs a compact contribution statement for all four people; the repository log can retain detailed evidence.

The initial repository and plan were prepared with Codex assistance at the user's request. This planning assistance is not evidence that a student implemented or understands the final model. Record any further assistance accurately and follow applicable course guidance if supplied later.
