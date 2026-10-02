# Work plan for four people

Revised on **2 October 2026** after reviewing the supplied guidelines and first-submission screenshot. This is an implementation plan, not a record of completed experiments. Course obligations are distinguished from proposed design choices in [REQUIREMENTS.md](REQUIREMENTS.md). Technical definitions are fixed in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Assignment and workload

Budget **20–24 hours per member**, approximately 80–96 team hours in a **relative seven-day implementation sprint after the first submission**. This requires roughly 3–4 focused hours per person per day, followed by a one-day repair buffer. The September dates were internal targets and have been superseded. These are planning estimates, excluding unattended computation. Keep the scope to one dataset, one backbone, one calibration method, and three corruption families. No second architecture is required because the course explicitly permits a second condition instead.

The user supplied these four handles and requested completion as soon as possible. Assignments below are ready to start; confirm daily availability and repository access at kickoff. People may swap complete packages if their skills fit better. No member is assigned only writing or slides. Full course names still need to be entered by the team.

### Telman3000 — Data and image corruptions

**Tasks and estimated effort**

1. Confirm the team/topic assignment and four names following the historical **16.09.2026** registration deadline; obtain first/final submission dates and clarify demo timing (1 hour).
2. Implement official CIFAR-10 retrieval, record source/checksum/class order, and build fixed stratified manifests with stable original sample IDs (4 hours).
3. Verify disjoint training, model-validation, calibration, threshold-selection, and test partitions; persist split seed and hashes (2 hours).
4. Implement deterministic Gaussian blur, Gaussian noise, and JPEG variants at three severity levels, plus clean control (5 hours).
5. Review visual corruption samples on development data; document exact preprocessing order, RGB/uint8 conversions, and noise seeds (2 hours).
6. Integrate the data interface, review MedvAx-AI's sample alignment and failure panels, and help reproduce a run on a second machine (3 hours).
7. Write dataset/preprocessing text, inspect failures with MedvAx-AI, maintain contribution evidence, and rehearse whole-project Q&A (3–5 hours).

**Planned outputs:** `src/data.py`, `src/corruptions.py`, `configs/data.yaml`, `artifacts/splits/*.json`, completed `data/README.md`, and corruption sample panel.

**Definition of done:** manifests reproduce the same IDs; all subsets are disjoint; class counts match the protocol; corruptions retain source labels and IDs; identical seeds yield identical images; each corrupted test image derives only from a test image; downloads and preprocessing work from the README.

**Dependencies:** none to start. Hand off data contract to everyone at M1 and corruption generator to MedvAx-AI at M2.

### LeoVesinML — Model, baseline, and reproducibility

**Tasks and estimated effort**

1. Establish the shared environment, pin the tested versions, and publish the source/module layout and configuration conventions (2 hours).
2. Load the explicitly named pretrained ResNet-18 weights, freeze the backbone in evaluation mode, implement preprocessing and cache extraction (4 hours).
3. Train the ten-class linear head on training features; choose the checkpoint using model-validation NLL; record seed, optimizer, settings, and weights provenance (4 hours).
4. Save the clean no-rejection baseline first, including predictions, full-set accuracy, NLL, ECE, class results, checkpoint hash, and runtime (3 hours).
5. Export the shared logit interface, profile full inference and cached evaluation separately, and implement reliable model/weight loading (3 hours).
6. Review Mysteri0K1ng's calibration fit and checkpoint isolation; finish exact run instructions and supervise a clean-environment reproduction (2–3 hours).
7. Write the method and runtime sections, contribute to failure interpretation, and rehearse demo/Q&A (2–4 hours).

**Planned outputs:** `src/model.py`, `src/train.py`, `src/predict.py`, `configs/model.yaml`, tested `requirements.txt`, checkpoint retrieval/retraining instructions, baseline outputs, environment/runtime record.

**Definition of done:** a ten-class head is actually trained; frozen BatchNorm statistics do not drift; no held-out split trains the head; fresh loading reproduces logits within documented numerical tolerance; baseline output predates comparisons; reported inference time includes the backbone; a teammate runs the README successfully.

**Dependencies:** Telman3000's manifests at M1. Handoff a tiny development logit sample early, then the frozen checkpoint and calibration/policy logits at M2. Do not expose final test results before the experiment is frozen.

### Mysteri0K1ng — Calibration, rejection, and metrics

**Tasks and estimated effort**

1. Implement and independently sanity-check accuracy, full-set NLL/ECE, coverage, rejection rate, accepted accuracy, and selective risk (4 hours).
2. Fit one positive temperature on the calibration split only; save it with checkpoint/split identifiers and objective before/after (3 hours).
3. Implement raw and calibrated confidence policies, the validation-only operating-threshold rule, and explicit infeasible-target handling (4 hours).
4. Implement the threshold sensitivity sweep, risk–coverage curves, and matched-coverage diagnostic; keep test curves descriptive (4 hours).
5. Review Telman3000's leakage checks and align metrics with MedvAx-AI's evaluation runner (2 hours).
6. Write comparison/metric explanations and limitations; prepare threshold and calibration Q&A for all four members (3–5 hours).

**Planned outputs:** `src/calibration.py`, `src/rejection.py`, `src/metrics.py`, `configs/evaluation.yaml`, `artifacts/temperature.json`, `artifacts/policy.json`, metric checks and sensitivity outputs.

**Definition of done:** positive temperature; no change in predicted labels from temperature scaling; all policy decisions locked before final test; threshold boundary and zero-acceptance cases handled; accepted accuracy always accompanied by coverage and counts; clean/corrupt metrics use the same definitions; no implied accuracy guarantee from a small validation sample.

**Dependencies:** can start with hand-calculated fixtures at M1. Needs LeoVesinML's frozen calibration/policy logits to fit and select policies at M2. Handoff stable metric/policy interfaces to MedvAx-AI before M3.

### MedvAx-AI — Integrated evaluation, failure analysis, and demo

**Tasks and estimated effort**

1. Implement the evaluation runner and result schema against small development fixtures (4 hours).
2. Integrate the 4 × 10 experiment matrix, retain per-image decisions, and produce compact tables, reliability diagrams, and risk–coverage plots (4 hours).
3. Lead analysis of at least three representative failures; compare clean/corrupt views, quantify confidence and decisions, and distinguish evidence from hypotheses (4 hours).
4. Build a small local live demo showing image, prediction, confidence, threshold, and accept/reject state, including a saved failure example (3 hours).
5. Integrate the first-submission proposal from team-owned sections; later assemble the final two-page results summary and up to three slides (2 hours).
6. Review LeoVesinML's clean-environment reproduction, audit final artifacts against the requirement matrix, and coordinate timed rehearsals and individual Q&A (3–5 hours).

**Planned outputs:** `src/evaluate.py`, `src/plots.py`, `demo.py` or demo notebook, `results/final_metrics.csv`, `results/per_image.csv`, `results/figures/*`, `reports/failure_analysis.md`, `reports/technical_summary.pdf`, demo slides/script.

**Definition of done:** every aggregate traces to sample-level outputs; model/policy are frozen for all final conditions; at least three failures have likely causes and limitations; plots show denominators and metric direction; live demo uses the saved model and policies; final summary is exactly two pages; demonstration lasts at most five minutes and uses at most three slides.

**Dependencies:** starts the runner with fixtures immediately; complete evaluation needs Telman3000 corruptions, LeoVesinML checkpoint, and Mysteri0K1ng policy. Other members write their own report sections; MedvAx-AI integrates them.

## Milestones and handoffs

The first-submission proposal is prepared on **2 October 2026**; it documents intended work and contains no measured results. Start the relative implementation sprint after that submission, subject to confirmed course deadlines and member availability. Days below are internal targets; the first and final submission deadlines are not stated in the supplied materials. If either deadline requires an earlier gate, adjust the schedule and preserve an independent reproduction check. Reduce optional polish and bootstrap analysis first; preserve mandatory evidence. The historical 16 September topic-registration deadline is recorded separately and is not a new project deadline.

| Gate | Target | Telman3000 | LeoVesinML | Mysteri0K1ng | MedvAx-AI | Exit evidence |
|---|---|---|---|---|---|---|
| S0: first submission | Prepared 2 Oct; submission date to confirm | Review data section; verify assignment | Review model/implementation section | Review experiment/metrics section | Integrate proposal PDF and check five required sections | First-submission proposal ready; team submits through course channel |
| M0: assignment check | Before implementation sprint | Confirm assignment/names/deadlines | Confirm role/hardware | Read protocol | Record availability | Assigned topic and actual course dates verified |
| M1: protocol + interfaces | Implementation Days 1–2 | Frozen manifests; download path | Environment + feature/model pilot | Metric fixtures + policy specification | Runner skeleton + report/demo outline | Everyone agrees split IDs, settings, schemas; baseline pilot runs on development data |
| M2: baseline + components | Implementation Days 3–4 | Corruptions ready and checked | Saved development baseline + fixed checkpoint | Temperature and operating thresholds fitted | Components integrated on development samples | Baseline saved; protocol/config/checkpoint/T/threshold hashes frozen before test |
| M3: final experiments | Implementation Day 5 | Audit alignment; investigate failures | Run inference and runtime profile | Validate metrics and sensitivity | Execute matrix; figures + 3 failures | Final metrics/figures saved; conclusions supported, including negative outcomes |
| M4: submission-ready | Implementation Days 6–7 | Verify data instructions | Fresh-environment run + README | Audit numerical claims | Final report/demo packaging | Checklist complete; exactly 2-page PDF; ≤3 slides; ≤5-minute demo; all four pass practice Q&A |
| Buffer | Implementation Day 8, before course deadline | All four fix verified problems and submit | | | | Instructor can access repository and all submitted artifacts |

Critical path: split manifests → classifier/checkpoint → calibration and policy → locked test evaluation → figures/failures → verified report and demo. Mysteri0K1ng and MedvAx-AI use development fixtures while the baseline is prepared so they do not wait for training.

## Working agreements

- Use branches and small pull requests linked to the member's work package. Review cycle: Telman3000 reviews MedvAx-AI; MedvAx-AI reviews LeoVesinML; LeoVesinML reviews Mysteri0K1ng; Mysteri0K1ng reviews Telman3000.
- Hold a 15-minute checkpoint daily during the sprint; each member shows an artifact, a blocker, and the next handoff. Record experiment-setting changes before final testing.
- Each member writes their own results/method contribution and logs actual commits, reviews, experiments, and interpretations in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
- Every member must be able to explain the dataset split, baseline, calibration, coverage tradeoff, one threshold decision, and one failure outside their own module.
- Rebalance work if a package exceeds the estimate; transfer a concrete subtask and update ownership rather than silently assigning all integration work to MedvAx-AI.

## Risks and bounded responses

| Risk | Response | Owner |
|---|---|---|
| Topic assignment uncertain | Verify the recorded instructor/team assignment after the historical registration deadline; repository creation alone does not reserve a topic | Telman3000 |
| Final deadline unknown | Obtain date at kickoff; move sprint gates earlier if needed and preserve a pre-submission check | Telman3000 |
| Slow compute | Cache frozen features, use the fixed subset, measure a development pilot before committing; avoid architecture searches | LeoVesinML |
| Calibration does not help | Report it honestly; test whether clean calibration transfers under corruption and explain limits | Mysteri0K1ng |
| No useful threshold meets target | Report infeasibility and show the predeclared tradeoff; no test-driven target adjustment | Mysteri0K1ng |
| Few failures of a planned type | Document the count and use another real failure; never fabricate examples | MedvAx-AI |
| Demo internet or hardware failure | Cache weights and samples locally; prepare saved outputs as a clearly identified fallback, retain a working live path | LeoVesinML and MedvAx-AI |
| Result or report cannot be reproduced | Run on a second environment before M4, fix missing files/settings, re-export affected evidence | All four |
