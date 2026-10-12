# Five-minute demonstration and individual preparation

Prepared 12 October 2026. This script is not evidence of an actual rehearsal. Three slides: `reports/slides.pdf`. Start `python demo.py --device cpu` beforehand; cache all data/weights, open http://127.0.0.1:8765. Fallback: the saved failure panel and slide metrics, clearly identified as saved results.

| Time | Presenter | Content/action |
|---|---|---|
| 0:00–0:40 | Telman Nuruzov | Slide 1: ten-class CIFAR-10 with abstention; question: accepted accuracy vs discarded images. Explain five disjoint source-ID splits and native corruptions. |
| 0:40–1:25 | Leo Vesin | Slide 1: frozen ImageNet ResNet-18 and trained linear head, frozen BatchNorm; model validation alone chooses epoch. A/B/C/D share checkpoint and predicted labels. |
| 1:25–2:15 | Nikita Shankin | Slide 2: 85.05% full accuracy → 90.90% accepted at 87.90% coverage. Separate calibration/policy splits fit T/tau. ECE improves slightly; severe noise remains poor. Sweeps are descriptive. |
| 2:15–3:15 | Aleksandr Medvedev | Live UI: example 2, Clean, Run classifier. Correct dog at 56.84% calibrated confidence is rejected below 57%. A/C output dog, B/D REJECT. Threshold stays frozen. |
| 3:15–4:10 | Aleksandr + Telman | Example 3: Clean, run → correct/accepted ship at about 85.7%. JPEG/Severe, run → wrong/accepted airplane at about 99.0%. Same source test:4720; lost outline detail/blocking is a cause hypothesis. |
| 4:10–4:45 | Nikita | Slide 3/example 1: even clean airplane→ship has very high confidence. Rejection improves the clean tradeoff, cannot guarantee robust confidence after corruption; no test retuning. |
| 4:45–5:00 | Leo | Public README, checkpoint/hashes, verifier and rerun commands. Separate cached head training and full image inference clocks; transition to questions. |

All four members must rehearse twice with a timer and independently rotate the questions below. Record actual times and weak answers before completing the rehearsal checklist.

1. **Why ResNet-18/linear head?** Small pretrained fixed features limit compute. Only the ten-class head is trained, keeping reliability comparisons controlled. Other architectures were not compared.
2. **Why five splits?** Training fits head; model validation chooses checkpoint; calibration fits T; policy selection chooses tau; final test estimates locked performance. Variants are generated after assignment; source IDs never overlap.
3. **What changes with T?** Softmax(logits/T) changes confidence; positive T preserves argmax. T<1 sharpens probabilities. It does not improve classification accuracy or provide OOD detection.
4. **What does 90.90% mean?** 1,598 correct / 1,758 accepted. Coverage 1,758 / 2,000. Full accuracy 1,701 / 2,000 = 85.05%. D rejects 103 correct images and accepts 160 errors.
5. **Reject everything?** Zero coverage, accepted accuracy/risk undefined (null), not 100% accuracy.
6. **Select tau?** Fixed 0.00–1.00 grid, step 0.01; maximum feasible policy-split coverage at >=90% accepted accuracy and >=20% coverage; smallest tau tie, fixed 0.80 fallback. Raw/scaled selected 0.56/0.57. Final sweeps never reselect it.
7. **NLL/ECE?** NLL penalizes low true-label probability and confident errors. ECE is weighted absolute confidence/accuracy gap in 15 equal-width bins; it depends on bins/sample size. Both use every image, even rejected ones.
8. **Calibration help?** Clean ECE 0.02011→0.01870, NLL nearly unchanged. Severe-corruption ECE/NLL worsen. B vs D mixes different accepted populations; A/C and matched coverage better isolate calibration.
9. **Robust/OOD?** No. Custom perturbations within ten classes expose a limitation; clean targets do not guarantee distribution-shift performance. Severe-noise D accepted accuracy is about 17.40%.
10. **Failure causes proven?** IDs, pixels, confidences and decisions are measured. Shape/background/detail explanations are hypotheses requiring interventions. Selected examples are not random prevalence estimates.
11. **Clocks include?** Final CPU timing includes fetch, corruption/hash, transform, backbone/head; 64 warmup images per condition excluded. Feature extraction/head training on MPS are separate. Cache replay has no forwards.
12. **Reproduce?** Documented CPU environment, verify manifests/checkpoint/policy, independent result verifier, cached replay, full inference to `reproduced/full`. Cross-device tolerance 1e-4; independently retrained heads need not be identical.

Remaining human course actions: confirm registration/final deadline/session timing, perform timed rehearsals, submit appropriate files and complete each member's supervised Q&A.
