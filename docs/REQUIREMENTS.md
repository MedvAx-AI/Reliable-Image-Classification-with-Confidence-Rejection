# Course requirements and evidence matrix

Source: user-supplied **Computer_Vision_Project_Guidelines.pdf**, pages 1–3, inspected in full. Topic-specific requirements come from the original topic screenshot; first-submission requirements come from the screenshot supplied on 2 October 2026. The newly supplied PDF was reviewed in full on that date. This document paraphrases the requirements and maps each to planned evidence. It is not a claim that future deliverables already exist.

The PDF supplies course constraints. The user's request authorizes creating the repository and a four-person plan. The PDF does not independently instruct this assistant to submit coursework, edit the course sheet, or contact other people.

## Mandatory evidence

| ID | Requirement and source | Planned evidence / completion check | Accountable owner |
|---|---|---|---|
| R01 | Small, controlled public/custom-data project; modern CV method (p1) | Fixed CIFAR-10 subset and pretrained ResNet-18 with a trained linear head; no large-scale training | LeoVesinML + Telman3000 |
| R02 | Four-person team (p1) | All four names/handles and contributions in final summary and contribution log | MedvAx-AI + all |
| R03 | Choose a listed, unique topic; register names by 16.09.2026 (p1) | Verify topic availability and record names in course sheet; log completion. Instructor assignment after the date is mandatory and cannot be revised | Telman3000 |
| R04 | Clear research/engineering question (p2) | README/protocol and summary state reliability-versus-coverage question and corruption comparison | Mysteri0K1ng |
| R05 | Establish baseline before change (p1–2) | Timestamped baseline development run A before introducing B–D; final baseline retained in all comparisons | LeoVesinML |
| R06 | Second condition/model and controlled comparison, one major factor at a time when possible (p1–2) | A vs B rejection; A vs C calibration; same checkpoint and image inputs; disclose coverage in B vs D | Mysteri0K1ng |
| R07 | At least two appropriate quantitative metrics (p2) | Full accuracy, ECE, NLL, accepted accuracy and coverage; defined denominators and saved counts | Mysteri0K1ng |
| R08 | At least one concise ablation/sensitivity experiment (p2) | Predeclared threshold grid and risk–coverage plot; all other settings fixed | Mysteri0K1ng |
| R09 | At least three failures, with likely causes rather than images alone (p2) | Failure analysis with IDs, predictions/confidences, decisions, causes/evidence and limitations | MedvAx-AI + all |
| R10 | State dataset split; prevent train/test leakage (p2) | Published manifests and disjointness checks; separate calibration/policy partitions; variants inherit source split | Telman3000 |
| R11 | Report runtime environment and approximate training/inference time (p2) | Actual measured hardware/software manifest, head-training and full inference time, extraction/calibration time | LeoVesinML |
| R12 | Reproducible code/notebook with clear exact run instructions and no missing files (p1, p3) | Fresh-environment run reproduces final tables/figures from documented commands, inputs, checkpoint and configuration | LeoVesinML; independently checked by MedvAx-AI |
| R13 | requirements.txt / environment information (p1, p3) | Tested dependency specification, Python version, accelerator details if used | LeoVesinML |
| R14 | Dataset download instructions (p3) | Official source, exact commands, directory layout, verification and split creation | Telman3000 |
| R15 | Saved final metrics and figures (p3) | Complete 40-row final matrix, sweep outputs, per-image evidence and rendered figures committed or accessible via documented artifact locations | MedvAx-AI |
| R16 | Two-page technical summary (p1–3) | Exactly two rendered pages covering all eight template sections; see section map below | MedvAx-AI integrates; all write |
| R17 | Five-minute demo, maximum three slides plus live demo (p1) | Timed script, ≤3 slides, functioning live path showing task/baseline/comparison/one failure | MedvAx-AI + all |
| R18 | Ten-minute demo-format session includes problem/dataset, baseline/comparison, quantitative result/visual, failure/limitation, conclusion, questions (p3) | Five-minute demonstration covers listed content; provisional five-minute Q&A; confirm timing ambiguity | MedvAx-AI; Telman3000 confirms |
| R19 | Individual supervised understanding check / Q&A (p1–2) | Each student independently explains model choice, metric, result, threshold and failure; rotate practice questions | Each person |
| R20 | Member names and individual contributions (p1, p3) | Actual contributions with commits/PRs/artifacts in log and compact final-report table; handles replaced/supplemented with course names | Each person; MedvAx-AI integrates |
| T01 | Reject low-confidence predictions rather than always returning a class (topic screenshot) | B/D implement REJECT; compare against A/C always-classify behavior; visible demo outcome | Mysteri0K1ng + MedvAx-AI |
| T02 | Evaluate accuracy and confidence calibration (topic screenshot) | Full/accepted accuracy, coverage, ECE/NLL, reliability diagram; raw versus scaled probabilities | Mysteri0K1ng |
| T03 | Evaluate rejection thresholds (topic screenshot) | Validation-selected operating points, feasibility record, fixed-grid sensitivity and risk–coverage plot | Mysteri0K1ng |
| T04 | Evaluate blurred, noisy, compressed images (topic screenshot) | All three corruption families at three levels plus clean control, paired by original ID | Telman3000 + MedvAx-AI |

## First-submission proposal requirements (2 October screenshot)

The screenshot adds an initial implementation/evaluation report; it does not replace final experiment evidence. These five rows are fulfilled by the proposal content, while the final-course evidence above remains future work.

| ID | Required section | Evidence in first_submission.md / PDF | Section owner |
|---|---|---|---|
| F01 | Project scope | Objective, exact task, all ten classes, fixed ResNet-18 and A–D scope | Mysteri0K1ng + LeoVesinML |
| F02 | Dataset preparation | CIFAR-10 source, 17,000-image subset, preprocessing and all five split counts/uses | Telman3000 |
| F03 | Experimental plan | Comparison table, training/calibration/threshold settings, metrics, sensitivity and failures | Mysteri0K1ng + LeoVesinML |
| F04 | Implementation plan | Six pipeline stages, libraries, named pretrained weights, artifacts and relative sprint | LeoVesinML + MedvAx-AI |
| F05 | Team responsibilities | Four named technical work packages, report contributions and review responsibilities | All; MedvAx-AI integrates |

Prepared report: [PDF](../reports/first_submission.pdf), [editable source](../reports/first_submission.md). The screenshot specifies neither a report page limit nor a due date. This proposal is two pages for convenience; the PDF's mandatory two-page summary applies to the later results report.

## All eight final-summary template sections

| Course template section | Required content in final two-page summary | Writer |
|---|---|---|
| 1. Problem and question | 1–2 sentences stating the exact comparison | Mysteri0K1ng |
| 2. Dataset | Name, size/subset, splits, preprocessing | Telman3000 |
| 3. Method | Baseline + second condition, pretrained weights, important settings | LeoVesinML + Mysteri0K1ng |
| 4. Results | One compact table/figure with main quantitative metrics | MedvAx-AI + Mysteri0K1ng |
| 5. Interpretation | 2–3 sentences explaining what the measured result means | All; MedvAx-AI edits |
| 6. Failure analysis | Representative failures and likely causes | MedvAx-AI + Telman3000 |
| 7. Reproducibility | Repository link, environment, approximate measured runtime | LeoVesinML |
| 8. Members' contributions | Clearly state each person's actual contribution | All |

Page 1's deliverable definition also requires the experiment and key quantitative result: include both explicitly under Method/Results, not only the algorithm description.

## Grade allocation and priorities

The project contributes **25% of the course grade**: **20 percentage points shared** and **5 individual**. Team rubric totals 20 points:

| Team category | Points | Planning implication |
|---|---:|---|
| Problem and experimental question | 2 | Freeze clear question and justified baseline early |
| Correct implementation and reproducibility | 4 | Reserve independent rerun time |
| Experimental design / comparison | 2 | Keep checkpoint and inputs fixed |
| Evaluation and metrics | 2 | Verify formulas, denominators and readable evidence |
| Failure analysis and technical reasoning | 4 | Explain at least three failures and limitations |
| Demo quality and technical clarity | 6 | Rehearse a complete, concise five-minute demonstration |

Individual Q&A contributes 5 points for each student. Failure analysis and demo together account for half of the team rubric, so they have explicit time allocations rather than being left until submission day.

## Assumptions and open items

- Dataset, backbone, subset sizes, numeric settings, threshold target, work estimates and sprint dates are proposed team decisions, not imposed by the PDF.
- First and final submission deadlines are absent. Obtain them; 16 September was the historical topic-registration deadline, not a project submission date. The expired September internal sprint has been replaced by relative implementation milestones.
- The PDF gives a five-minute demo requirement and a ten-minute session heading. Plan five minutes of content plus five minutes of questions pending instructor confirmation.
- Sheet access, topic uniqueness and registration remain unverified. No edit or submission to the sheet has been performed.
- User supplied four GitHub handles. Course/legal names, collaborator access and availability still need the team's confirmation at kickoff.
- A complete plan covers all requirements, but actual course compliance depends on executing the plan and passing the final checklist.
