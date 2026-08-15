# SP model ops: shadow run, cutover, and send idempotency

Companion to docs/SP-AUDIT.md (what was found) and research/SP-BACKTEST.md
(why v2's parameters are what they are).

## Shadow run (state on merge)

`mlb/daily/config.py` ships with:

    ACTIVE_MODEL = MODEL_V1   # team Elo - the graded, emailed record
    SHADOW_MODEL = MODEL_V2   # + SP/rest/travel - runs in parallel

Every daily run predicts BOTH models. v1 stays exactly as it was: same
probabilities, same ledger (`data/mlb/predictions/grades.csv`), same record.
v2 writes `data/mlb/predictions/v2-sp/slate_{date}.csv` and accrues its own
`grades.csv` there. The slate email shows both tables (v2 with the per-starter
Elo adjustment column); the grade email appends v2's running paired delta once
it has graded days. Neither ledger ever mixes with the other - that is the
model-card versioning contract.

Note on the first shadow days: v2's pitcher book is built from
`data/mlb/pitcher_starts.csv`, which contains 2010-2025 at merge time. The
first workflow run backfills the 2026 season from statsapi (checkpointed,
resumable, ~25 minutes once) and the file then advances daily. Until that
backfill lands, v2 falls back toward staff/league ratings - safe, just less
sharp - so start counting the 7 shadow days AFTER the first successful
"Update pitcher starts" step.

## Cutover checklist (after >= 7 shadow days)

1. Compare buckets: v2's `cum_d_ll_mean ± cum_d_ll_se` (its grades.csv, also
   on the model card's History tab) against v1's over the same days.
   Holdout expectation from the backtest: ~ -0.001 log-loss per game vs v1;
   over 7 days (~100 games) noise WILL dominate - the shadow window is a
   plumbing soak test, not a statistical referendum. Cut over unless v2 is
   malfunctioning (missing adjustments, TBD-heavy slates mis-handled,
   pathological adj values).
2. In `mlb/daily/config.py` swap the roles:

       ACTIVE_MODEL = MODEL_V2
       SHADOW_MODEL = None      # or keep MODEL_V1 shadowed for symmetry

3. Nothing else changes: v2's bucket keeps accruing (its shadow-period record
   continues seamlessly), v1's ledger freezes as the pre-change record, and
   the model card lists both with their date windows.

## Send idempotency (duplicate-email fix)

`reports/mlb_daily/sent.json` records every delivered email keyed
`{report_type}:{date}` with a content hash (`mlb/daily/send_ledger.py`):

- rerun with unchanged content -> `send=false` in the manifest -> the
  workflow's mail steps are skipped;
- content changed (score correction, new probables) -> re-sent with
  " (updated)" in the subject;
- delivery is recorded AFTER the SMTP step succeeds, so a failed send is
  retried next run instead of being falsely remembered.

This closes the 2026-08-10 duplicate-send incident (push-triggered rerun +
scheduled run, byte-identical reports emailed twice - see audit §3). The
workflow's push trigger is intentionally kept: with the ledger, the extra
run is harmless, and it still delivers first-day emails on merges.

## daily-report.yml status

The "No jobs were run" failures (late July - Aug 7) were the OLD workflow
file's empty `schedule:` key making the file invalid; the 2026-08-07 rewrite
(PR #21) already fixed it and every run since has been green. No deletion
needed. This branch's changes to the workflow are additive: the pitcher-starts
update step, per-email send gating, and the delivery-record step.
