# DAVE Agenda

## Golden Dataset Follow-Up

The first DAVE golden dataset has been created with 27 approved NFHS/high-school football clips.

Immediate next steps:

1. Run `eval_golden_dataset.py` against the live backend to produce the first baseline report.
2. Review misses and decide whether each miss is a model issue, prompt issue, rule-retrieval issue, or label issue.
3. Track accuracy by result (`FOUL` vs `NO FOUL`) and by expected label.
4. Flag regressions after any prompt, rule, backend, or model change.
5. Keep the `.mov` clips local or move them to an approved private storage location before adding any automated cloud eval.

Acceptance target for the first eval pass:

- Every clip in `golden-dataset/videos/` has a matching row in `golden-dataset/labels.csv`.
- The eval output clearly shows correct/incorrect for each clip.
- No-foul clips are scored separately so false positives are visible.
- Borderline plays remain in the dataset because they are the best regression checks.
