# Execution ledger - plan: docs/superpowers/plans/2026-10-12-final-evaluation.md

Base: cb4dbd9d919dbd2d00070b1a93d7b95abd516d3a.

Pre-flight: evaluation consumes the documented ordered-logit handoff; plots, failure selection and demo consume the same frozen condition keys and IDs. No interface conflict.

Execution: the user requested execution of the existing four-person plan. Native managed worktree creation was unavailable because the chat root is not a Git repository; a linked Git worktree at outputs/project-completion protects the user's original edits.

Task 1: complete. Actual frozen run: 20,000 forwards, 40 aggregates, 80,000 decisions, 2,020 sweeps. Independent NumPy verification passed; saved checkpoint, T and thresholds unchanged. Fresh suite: 94 tests passed.

Task 1 test correction: the fixture expected a rejected output at an index belonging to the first, correct sample. Assert the explicit source ID and policy instead; this tests the intended rejection without imposing an unrelated output ordering.

Task 1 integration finding: existing data/corruption fixtures regenerate committed manifest metadata (creation time and NumPy version). The source IDs and per-split hashes remain unchanged. Regression demonstrated in an isolated copy; fixtures now write to temporary directories to prevent unnecessary repository metadata drift.

Task 2: complete. Failure selection tests and real-pixel demo test passed. Three cases and paired images were inspected; browser live outputs match the saved predictions and confidence/acceptance decisions. Cause explanations are visual hypotheses, not causal claims.

Task 3: packaging complete; independent review pending. Exactly two summary pages and three slides rendered and inspected. All eight sections and repository/source links are present. Fresh cached replay reproduced six outputs byte-for-byte. Full image rerun in the fresh environment completed: 20,000 forwards, identical ordered IDs/labels, maximum logit difference zero and six outputs byte-identical.

Baseline prediction CSV normalization changes CRLF to LF only; the logical content is identical (`git diff --ignore-space-at-eol` passed). No baseline prediction or fitting parameter changes.

Final review: independent fresh-context reviewer inspected cb4dbd9..4dc3d43, ran 94 tests and the count verifier, checked PDF counts/text and representative live HTTP cases. No Critical findings.

Final: fixed missing current-source validation in the independent verifier — four source-drift regression cases observed FAIL (DID NOT RAISE), then PASS with shared runner/verifier source/data-config/policy checks; complete suite 98/98 passed.

Final: Ruling: re-grade stale demo display from Minor to Important — changing a sample could put a new cause explanation beside an old prediction, misleading the presentation. Fix clears predictions on example/condition/severity changes and locks selection while running. Browser regression RED→GREEN, including observed lock state during an uncached condition. Cost if wrong: small additional interface state logic; numerical model behavior unchanged.

Final: Ruling: actual rehearsal/registration/course upload/supervised understanding remain human gates — prepared artifacts are not proof these happened. Cost if wrong: a student could otherwise mistake readiness for course completion.

Final: Ruling: pretraining duplicate-freedom and internal causal failure explanations are not established — retain visual causes as hypotheses and limited generalization. Cost if wrong: results could otherwise be overstated.

Final: Ruling: reviewer did not repeat the complete 20,000-forward run — executor did run it in the fresh environment and compared all logits/IDs/labels and six outputs exactly; reviewer independently checked metrics and representative live inference. Cost if wrong: an unobserved platform-specific issue outside the recorded CPU environment; no cross-device equality is promised.

Task 3: complete technical packaging and review fixes. Replayed cache after fix, six outputs still identical. No deferred Minor findings (the stale display issue was re-graded by its effect and fixed). Public PR publication remains the final integration step.
