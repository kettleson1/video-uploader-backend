# DAVE Agenda

## Golden Dataset Follow-Up

The first DAVE golden dataset has been created with 27 approved NFHS/high-school football clips.

Immediate next steps:

1. Build a repeatable eval runner that uploads each local golden clip and records DAVE's prediction.
2. Compare each prediction against `golden-dataset/labels.csv`.
3. Report accuracy by result (`FOUL` vs `NO FOUL`) and by expected label.
4. Flag regressions after any prompt, rule, backend, or model change.
5. Keep the `.mov` clips local or move them to an approved private storage location before adding any automated cloud eval.

Acceptance target for the first eval pass:

- Every clip in `golden-dataset/videos/` has a matching row in `golden-dataset/labels.csv`.
- The eval output clearly shows correct/incorrect for each clip.
- No-foul clips are scored separately so false positives are visible.
- Borderline plays remain in the dataset because they are the best regression checks.
