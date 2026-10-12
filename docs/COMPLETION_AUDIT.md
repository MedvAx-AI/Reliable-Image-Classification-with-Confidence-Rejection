# Completion audit — 12 October 2026

Technical readiness against the supplied course PDF, four-person plan and milestone screenshot. Prepared files do not prove course upload, rehearsal or independent student understanding. This audit supplements the original requirement matrix; the experimental protocol remains frozen.

| Requirement | Evidence | Status |
|---|---|---|
| Modern pretrained model and controlled baseline/comparison | Frozen ResNet-18 + trained head; fixed A/B/C/D in `results/final_metrics.csv` | Verified |
| Official data loading/preprocessing/checksums | `src/data.py`, `src/corruptions.py`, `data/README.md`, five committed source-ID manifests | Verified |
| No leakage or test fitting | Ordered ID/label/hash checks, five disjoint splits, unchanged checkpoint/T/tau | Verified |
| Two or more metrics and correct denominators | Accuracy, coverage, NLL, ECE, risk, per-class counts; independent NumPy verification | Verified |
| Clean/blur/noise/JPEG evaluation | Ten image conditions, 20,000 forwards, 40 aggregate rows and 80,000 decisions | Verified |
| Sensitivity and interpretation | 101-point raw/scaled curves per condition, matched coverage, no test-based selection | Verified |
| Three real failures with causes/limitations | `reports/failure_analysis.md`, source IDs/paired images/counts, hypotheses separated from observations | Verified |
| Environment/training/inference clocks | Baseline runtime artifact and `results/run_manifest.json`; README timing inclusions | Verified |
| Reproducible code/checkpoint retrieval | Committed head/logits/policies, official data/pretrained URLs, tested README commands | Verified locally; publication checked separately |
| Independent fresh-environment evidence | 94 tests; all 40 metric sets, 80,000 decisions, 2,020 sweep counts; six cached outputs byte-identical | Verified; fresh 20,000-forward rerun has identical logits and six outputs, recorded in `results/verification.json` |
| Exactly two pages/eight sections/table/figure/contributions | `reports/technical_summary.md` and PDF, every page inspected | Verified |
| At most three slides and live path | Three-page slides, real model inference in `demo.py`, browser checks of measured cases | Verified |
| Five-minute demo/fallback | `reports/demo_script.md`, saved slides/panel | Prepared; actual two timed rehearsals pending |
| Four identities/contribution evidence | Report and `docs/CONTRIBUTIONS.md`; assistant support disclosed | Recorded; team confirms course identities if necessary |
| Each student understands all components | Individual Q&A prompts and measured answers prepared | Human supervised verification pending |
| Topic registration/deadlines/demo format | Course PDF; first-working-model screenshot gives 13 October 2026 | Team confirmation pending |
| Course upload and grading | Team's course submission channel | Pending; no course upload performed |

Issue #4 combines technical implementation with actual rehearsal. Technical items are ready; its combined reproduction/rehearsal checkbox stays open until the team rehearses. Do not claim the entire course project is 100% completed because code/tests/PDFs are ready.

`reports/first_working_model.pdf` covers the 13 October milestone: implemented pipeline, configuration/training, measured development baseline, observations and public repository link. The two-page final summary is a separate later deliverable.
