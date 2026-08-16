# SP model ops: shadow run, cutover, and send idempotency

Companion to docs/SP-AUDIT.md (what was found) and research/SP-BACKTEST.md
(why v2's parameters are what they are).

## Cutover: done (2026-08-15)

`mlb/daily/config.py` now has:

    ACTIVE_MODEL = MODEL_V2   # + SP/rest/travel - the graded, emailed record
    SHADOW_MODEL = None

The shadow window below ran for one day (2026-08-14 merge to 2026-08-15
cutover) before being cut short on explicit instruction: v1's live record was
exploratory and noise-dominated at n=94 anyway, so the plumbing soak test
mattered more than the extra days of statistical comparison. v1's ledger
(`data/mlb/predictions/grades.csv`) freezes wherever it last graded and stays
on the model card as the pre-change record; nothing about the flip touches it
retroactively (`predictions_dir()` keys purely off model-version name, so v1
keeps the original flat bucket forever). To resume an ongoing v1/v2
comparison, set `SHADOW_MODEL = MODEL_V1`.

Applying `ACTIVE_MODEL = MODEL_V2` is not just the two-constant flip that
originally shipped with the shadow-run design - the pipeline was structured
so only the *shadow* slot ever computed pitcher/rest/travel adjustments
(`mlb/daily/run.py` hardcoded `adjustments=` onto the shadow branch only, and
every "primary" grade/slate path defaulted to the flat v1 bucket regardless
of which version was active). Flipping just the config constants would have
silently relabeled the plain team-Elo slate as "v2-sp" without ever computing
the adjustment. `run.py` and `export_site.py` were fixed alongside the config
flip so `adjustments=` follows whichever role (active/shadow) actually holds
`MODEL_V2`, the primary grade/slate paths follow `predictions_dir(ACTIVE_MODEL)`,
and the model-card bucket list and History tab track the active version
instead of hardcoding v1. The website's `MlbSlateTable` also picked up the
starter/adjustment columns and a note that switches on whether the slate
actually carries `*_sp_adj` values, matching `mlb/daily/emails.py`'s slate
table - it previously hardcoded "does not use starter identity" unconditionally,
which the cutover would otherwise have made false.

## Original shadow-run design (for a future cutover)

Every daily run predicts BOTH the active and shadow model, each in its own
bucket (`predictions_dir(version)`) with its own `grades.csv`, so neither
ledger ever mixes with the other - that is the model-card versioning
contract. The slate email shows both tables when a shadow is set (the v2-style
table with the per-starter Elo adjustment column); the grade email appends the
shadow's running paired delta once it has graded days.

Note on the first shadow days of any future cutover: v2's pitcher book is
built from `data/mlb/pitcher_starts.csv`. If that file is missing recent
games, the daily statsapi backfill (checkpointed, resumable, ~25 minutes
once) needs to complete first, or v2 falls back toward staff/league ratings -
safe, just less sharp.

## Cutover checklist (for switching again later)

1. Compare buckets: the candidate's `cum_d_ll_mean ± cum_d_ll_se` (its
   grades.csv, also on the model card's History tab) against the current
   active model's over the same days. Holdout expectation from the backtest:
   ~ -0.001 log-loss per game for v2 vs v1; small samples WILL be
   noise-dominated - treat a short shadow window as a plumbing soak test,
   not a statistical referendum, unless the candidate is actually
   malfunctioning (missing adjustments, TBD-heavy slates mis-handled,
   pathological adj values).
2. In `mlb/daily/config.py` swap the roles:

       ACTIVE_MODEL = <candidate>
       SHADOW_MODEL = None      # or keep the old active model shadowed

3. Nothing else changes: the promoted model's bucket keeps accruing (its
   shadow-period record continues seamlessly), the demoted model's ledger
   freezes as a historical record, and the model card lists every version
   that's ever been active or shadow with its own date window.

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
