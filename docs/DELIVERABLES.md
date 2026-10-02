# Deliverables, demonstration, and completion gates

The first-submission proposal is prepared; the model, measured results, final summary and slides remain planned outputs. Keep the proposal and final results summary as separate deliverables.

## First submission: implementation and evaluation proposal

The supplied screenshot asks how the team will implement and evaluate the idea. Submit [first_submission.pdf](../reports/first_submission.pdf), with [editable source](../reports/first_submission.md). The proposal uses the five requested sections:

1. **Project scope:** objective, closed-set task, all ten classes, fixed model and bounded experimental settings.
2. **Dataset preparation:** official source, selected subset, exact preprocessing and disjoint train/development/test allocation.
3. **Experimental plan:** A–D comparisons, head training, calibration, threshold rule, metrics, sensitivity and planned failure analysis.
4. **Implementation plan:** pipeline steps, libraries, named pretrained weights, reproducibility and relative milestones.
5. **Team responsibilities:** four explicit technical packages and each member's report/review contribution.

This initial report describes intended work. The screenshot does not request measured results or set a page limit; we use a compact two-page PDF. The later two-page technical summary still requires actual results, figures and failure explanations under the general guidelines. Do not treat completion of this proposal as completion of the experiment or final submission.

First-submission acceptance: all five sections present; settings agree with the protocol; four member identifiers correct; PDF readable with functioning links; wording reflects planned work; team confirms course names and submission deadline. First and final deadlines remain unprovided. Coursework is submitted by the team through its course channel.

## Reproducible repository

LeoVesinML maintains the exact command sequence; Telman3000 owns dataset instructions; MedvAx-AI independently runs it before submission. Choose a simple local Python project or executable notebook. A web application is unnecessary.

Publish tested commands for these stages, with prerequisites and expected outputs:

1. Clone this repository and check out the reported commit/tag.
2. Create a virtual environment; install the exact tested dependency set. Give OS-specific activation commands where needed.
3. Download official CIFAR-10 and the named pretrained weights, verify downloads, and generate the published split manifests.
4. Extract features, train the head, and select the validation checkpoint, or retrieve the final checkpoint from a stable documented location with SHA-256 verification.
5. Reproduce the no-rejection baseline, fit temperature, and select thresholds from the designated held-out splits.
6. Evaluate the locked test matrix, save per-image and aggregate outputs, then regenerate all tables and figures.
7. Launch the live demo from cached local weights and examples.

The final README must contain real commands that were run, actual file paths, expected artifacts, estimated times based on measurement, CPU/GPU requirements, seeds, known nondeterminism and numerical tolerances. If a checkpoint is omitted from Git, supply both a working retrieval method and a retraining path. A missing external artifact is a submission blocker.

Suggested final layout:

```text
README.md
requirements.txt
configs/                 # final data/model/evaluation settings
src/                     # data, corruption, model, train, calibration, metrics, evaluation, plots
tests/                   # targeted correctness checks
data/README.md           # download and split instructions; raw images excluded from Git
artifacts/splits/        # committed source-ID manifests
artifacts/policy.json    # frozen threshold-selection evidence
artifacts/temperature.json
results/final_metrics.csv
results/threshold_sweep.csv
results/per_image.csv    # or documented compressed artifact with schema/checksum
results/run_manifest.json
results/figures/
reports/failure_analysis.md
reports/technical_summary.pdf
reports/slides.pdf       # at most three presentation slides
demo.py                 # or an executable local demo notebook
docs/CONTRIBUTIONS.md
```

Measure training/feature extraction and inference time separately; cached feature/head evaluation is not full image inference. Save enough model, configuration and data provenance to trace each reported number to its run. Test all relative links and confirm instructor access to any external files.

## Final two-page technical summary

**Exactly two rendered pages**, not a two-page-looking Markdown file. Use all eight sections in the course template; do not invent results to fill the layout.

| Space budget | Content | Owner |
|---|---|---|
| Page 1, about 15% | Title, team names, problem and precise research question | Mysteri0K1ng + MedvAx-AI |
| Page 1, about 25% | Dataset sizes/splits/preprocessing and leakage prevention | Telman3000 |
| Page 1, about 30% | Baseline, calibrated/rejection conditions, fixed settings and experiment | LeoVesinML + Mysteri0K1ng |
| Page 1, about 30% | Compact main result table or figure, with coverage alongside accepted accuracy | MedvAx-AI + Mysteri0K1ng |
| Page 2, about 20% | Two or three interpretation sentences and threshold sensitivity takeaway | All |
| Page 2, about 40% | At least three concise representative failure descriptions/images and likely causes | MedvAx-AI + Telman3000 |
| Page 2, about 20% | Limitations, environment, measured runtime, repository/reproduction reference | LeoVesinML |
| Page 2, about 20% | Each student's contribution and compact references | All |

These proportions are layout guidance. Render the final PDF, verify exactly two pages, readable text/figures, all sections present, accurate numerical transcription, and no clipped elements. Keep the extended failure analysis and complete results in the repository.

The main conclusion must answer the question using actual numbers: e.g. accepted accuracy changed by a measured amount while coverage fell to a measured level. Do not describe calibration as improving classification simply because probabilities changed. State if a target was infeasible or gains disappeared under corruption.

## Five-minute demo and maximum three slides

The PDF also calls the full demo-format session ten minutes. Provisional schedule: **five-minute demonstration + five-minute instructor Q&A**. Telman3000 confirms the interpretation. Until clarified, do not expand prepared content beyond five minutes or three slides.

| Time | Presenter | Content |
|---|---|---|
| 0:00–0:40 | Telman3000 | Slide 1: problem, dataset, split, and why an uncertain prediction might be rejected |
| 0:40–1:25 | LeoVesinML | Slide 2: fixed model and A–D conditions; show the baseline |
| 1:25–2:15 | Mysteri0K1ng | Slide 3: measured accuracy/coverage and calibration result, threshold sensitivity |
| 2:15–3:15 | MedvAx-AI | Live demo: same image under baseline and rejection; show raw/calibrated confidence, frozen tau and decision |
| 3:15–4:10 | MedvAx-AI + Telman3000 | Live clean/corrupt comparison and one real failure, likely cause and limitation |
| 4:10–4:45 | Mysteri0K1ng | Answer the research question and state what the method cannot guarantee |
| 4:45–5:00 | LeoVesinML | Reproducibility pointer and transition to questions |

Run from local cached assets; rehearse at least twice with a timer. If the UI allows changing tau, label the slider as an exploration control and show the locked operating point separately. Do not present an interactively chosen test threshold as the reported policy. Prepare a saved-output fallback while retaining a working live demonstration.

## Individual understanding check

Each person answers all questions independently before submission, including questions outside their module:

1. Why this pretrained model and why a newly trained ten-class head?
2. How do model validation, temperature calibration, policy selection and final test differ?
3. How could generating corruptions before splitting create leakage?
4. What is the difference between full accuracy, accepted accuracy and coverage?
5. Why is high accepted accuracy alone insufficient? What happens when everything is rejected?
6. What do NLL, ECE and a reliability diagram measure? Why is ECE bin-dependent?
7. What does positive temperature scaling change, and what should remain unchanged?
8. How was tau selected? What does the 90% validation target fail to guarantee?
9. How does the experiment change only one major factor at a time?
10. What is the sensitivity experiment, and why must its test curves not select the final policy?
11. Explain one actual quantitative result and a plausible competing interpretation.
12. Explain one confidently wrong prediction or unnecessary rejection, evidence for its likely cause, and one limitation.
13. What do the runtime measurements include, and how would another student reproduce the figure?

During practice, rotate questions; no person relies on another teammate to answer. Record remaining weak areas and repeat only those. The supervised individual check is part of the course, separate from team deliverables.

## Final submission checklist

- [ ] Assigned topic and four recorded names are confirmed following the historical 16 September registration deadline; final deadline and demo timing confirmed.
- [ ] First-submission proposal covers the five screenshot sections and is submitted by the team by the confirmed initial deadline.
- [ ] Source code or executable notebook exists; no missing files or inaccessible artifacts.
- [ ] README has exact tested commands for setup, data, training/checkpoint loading, evaluation, plots and demo.
- [ ] `requirements.txt` / environment description and real hardware/software/runtime record are present.
- [ ] Dataset source/download/checksum, class order, preprocessing, split sizes/IDs and leakage checks are recorded.
- [ ] Baseline and second condition(s) use a controlled comparison and a fixed checkpoint.
- [ ] At least two appropriate metrics are computed correctly; accepted accuracy is paired with coverage/counts.
- [ ] Clean, blurred, noisy and JPEG-compressed evaluations are complete.
- [ ] Calibration and threshold selection use their own held-out data; no test tuning.
- [ ] Threshold sensitivity is complete with a clear interpretation.
- [ ] Final metrics, figures, per-image evidence, temperature/policy values and run identifiers are saved.
- [ ] At least three failures have likely causes and limitations; one is ready for the demo.
- [ ] An independent fresh-environment run reproduces the reported outputs within declared tolerance.
- [ ] Technical summary PDF is exactly two pages and contains all eight template sections, one compact figure/table, experimental result, and every student's contribution.
- [ ] Maximum three slides; five-minute demonstration rehearsed, with a live path and clear research-question answer.
- [ ] All four names/course identities and actual contributions are recorded.
- [ ] Every member can independently answer model, metric, result, threshold and failure questions.
- [ ] Repository and submission links are accessible to the instructor; report/figures correspond to the final commit.

Do not mark unchecked future work as complete merely because it appears in this plan.
