# Milestone 1 acceptance set

The companion examples and blind-check reservation are prepared. Completing the milestone requires an unseen full high-school coaches film cleared for this test, an independently graded reference timeline for that film, and a verified example of a genuine short forward pass.

## Official’s review

1. Watch the original film with a player that shows elapsed time. Work from a private copy of [the blank timeline](official-timeline.csv). Identify the exact source file by filename and SHA-256, give it a film ID, and use that ID in both sheets. Record who reviewed it and when.
2. Record one row per actual snap, including special teams. Use seconds from the start of the original file for snap and dead-ball times. Count repeated camera angles only once. Record an obscured boundary or an interrupted play in the notes. Leave an unobservable time blank. Use as many rows as the film requires.
3. Record FOUL, NO FOUL, or UNCERTAIN for each play. Describe the exchange when relevant, including a direct forward handoff, an airborne forward pass, or an unclear exchange. Mark judgment calls and explain what the angle shows.
4. Use [the findings sheet](official-findings.csv) for one row per possible foul, keyed to its play. Leave it empty for a no-foul play. Optional fields such as jersey number may be left blank.
5. Before opening DAVE’s results, save a dated copy of the sheets with the film identity and the official’s sign-off. Check unique consecutive play numbers, valid finding references, ascending snap times, and a documented reason for every missing boundary. Keep private film and identifying review records outside Git.

## Companion examples and final blind check

The [evaluator manifest](companion-and-reserve.csv) identifies exact footage by source hash and elapsed seconds. Its labels are for scoring. Keep filenames, notes, expected labels, and official sheets out of model prompts.

The companion set contains four approved golden clips and two officially reviewed running plays from Leyden–Morton. The latter test forward handoffs, including a quarterback carrying out a fake. Add a genuine short forward pass once an official confirms visible airborne forward travel, records its file and times, and grades the play. It can come from the new full game.

Four separate golden clips are reserved for the final check. Their IDs and source URLs are absent from the four saved November blind-run reports and their evaluated sources, respectively. They have approved labels and distinct file hashes. This establishes a reservation against the recorded runs. Confirm any additional evaluation history with the owner before final scoring. Keep these clips out of tuning and check that the scored action excludes training narration or graphics that reveal the ruling. Four clips provide a spot check. Use additional independently graded examples to estimate the release targets.

## Completion record

- [x] Select and verify locally available companion examples.
- [x] Reserve separate clips for the final blind check.
- [ ] Record the new full-game source and confirmation it is cleared and unseen.
- [ ] Freeze the independent official timeline and findings.
- [ ] Verify and record the genuine short forward pass control.
