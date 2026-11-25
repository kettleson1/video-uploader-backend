# DAVE MVP 1 delivery plan

November 25, 2026

## Member experience

MVP 1 is a private pilot for high-school coaches film. A signed-in member uploads a game, checks the play timeline, and asks DAVE to screen each accepted play for potential fouls. The member can watch the clip, inspect the evidence, agree, dismiss, correct, or add a finding, and return later to the saved review. Each member has access to their own games. The official makes the ruling and any enforcement decision.

## Starting point

The upload and review workflow is ready for staging. The local Leyden–Morton test completed all 44 processing jobs for ten reviewed plays. Two-account testing verified uploader privacy. Foul-screening accuracy is the main release blocker. On a separate 16-clip blind test, DAVE matched the official’s foul/no-foul outcome on four clips. Full-length, unseen coaches film still needs an end-to-end test. See the [backend project status](README.md#coaches-film-mvp-status).

## Delivery milestones

1. **Establish the acceptance test.** Select one unseen, full-length coaches film, ideally near the typical 125 plays. Before seeing DAVE’s results, an official records the snaps, potential fouls, and judgment calls. Include known fouls, clean running plays, forward handoffs, and a genuine short forward pass in the companion test set. Finish with a reference timeline and official labels. Reserve separate film or clips for the final blind check after fixes.

2. **Prove the member workflow on staging.** Use a separate WordPress staging site with its own backend, worker, database, private bucket, and credentials. Upload through the member screen, interrupt and resume the worker, and reopen the review after a fresh sign-in. Every accepted play must produce a playable clip and a saved result. Verify zero duplicate plays after restart, persistence of corrections after sign-out, and denial of access to a second member attempting to view or edit the game. Record elapsed processing time and cost per game.

3. **Qualify the timeline and foul screening.** Compare detected clips with the official’s snap list. Count missed snaps, late starts, duplicates, and minutes spent correcting boundaries. Score confirmed fouls surfaced, fouls missed, false flags on clean plays, and uncertain findings separately. Fix recurring errors, then rerun the blind check. Advance when the corrected timeline includes every reference snap and its live action, oversized clips are held for review, and screening meets the agreed pilot targets on held-back examples.

4. **Run an invited-member pilot.** Give a small group of officials access to a dedicated full-game review page. Each uploads their own coaches film and completes a review. Track failed uploads, review time, and which findings were worth watching. Officials should be able to finish independently, explain why a play was flagged, and retrieve their corrections. The product owner reviews accuracy, time, and cost figures and approves release.

5. **Add MVP 1 to the live site.** Introduce a separate **Full Game Review** entry on www.davesystemsinc.com for approved members. Verify the existing single-play uploader before and after enabling the entry. Rehearse withdrawing the new entry while preserving uploaded games and reviews. Expand access in stages after one live member completes upload through saved review, the single-play check passes, and the rollback procedure has been tested.

## Proposed pilot targets

| Measure | Proposed target |
| --- | --- |
| Reference plays captured from snap through end of play before correction | At least 95% |
| Reference plays captured after correction | 100% |
| Boundary editing for a typical game | At most 15 minutes |
| Official-confirmed fouls surfaced for review | At least 80% |
| False foul flags | At most 10 per 100 official-clean plays |

Track uncertain flags separately because they also consume review time. Agree on these criteria before the final blind check. The existing sample is too small to establish these rates reliably.

## Timing and ownership

Allow one week for the reference set and staging run, followed by one to two weeks for fixes and blind retesting, assuming film and an official are available. Accuracy results determine the pilot date.

The product owner supplies one full-length coaches film cleared for testing, identifies an official who can grade it independently, and approves the release decision. The official prepares the reference timeline and labels. Engineering handles staging, measurement, test runs, fixes, and release packaging.
