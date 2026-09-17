#!/usr/bin/env bash
# Retry `git pull --rebase && git push` a few times with exponential
# backoff, for workflow steps that commit generated data back to the repo.
#
#   git-push-retry.sh <branch> [max_attempts]
#
# Exits 0 ONLY when a push actually reached the remote. Exits nonzero with a
# ::error annotation once max_attempts is exhausted, so the calling step -
# and the workflow run - fail loudly instead of quietly finishing "green"
# with the local commit never pushed (the previous version of this loop hit
# `break` only on success and otherwise fell through to a bare `sleep`,
# whose own exit 0 became the loop's exit status - a real commit could sit
# unpushed while the step still reported success). No sleep after the final
# attempt, since there is nothing left to wait for.
set -euo pipefail

BRANCH="${1:?usage: git-push-retry.sh <branch> [max_attempts]}"
MAX_ATTEMPTS="${2:-4}"

# The daily jobs are triggered by the same push and all write the same
# generated files (the email archive index, the cross-sport summary), so
# a rebase onto a sibling's commit can conflict on them. Those files are
# derived, never hand-edited: the index is the union of both sides and
# the summary is rebuilt from the data, so the conflict is resolved here
# rather than dropping the whole commit.
resolve_generated_conflicts() {
  local resolved=1
  for f in $(git diff --name-only --diff-filter=U); do
    case "$f" in
      web/public/emails/index.json)
        git show ":2:$f" > /tmp/index.ours.json
        git show ":3:$f" > /tmp/index.theirs.json
        python -m data_jobs.email_archive merge-index \
          --ours /tmp/index.ours.json --theirs /tmp/index.theirs.json --out "$f" \
          && git add "$f" || resolved=0
        ;;
      web/public/data/summary.json)
        python -m data_jobs.build_summary && git add "$f" || resolved=0
        ;;
      *)
        echo "git-push-retry: unresolvable conflict in $f"
        resolved=0
        ;;
    esac
  done
  return $(( 1 - resolved ))
}

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  if git pull --rebase origin "$BRANCH" && git push; then
    echo "git-push-retry: push succeeded on attempt $attempt"
    exit 0
  fi
  if [ -d "$(git rev-parse --git-path rebase-merge)" ] || [ -d "$(git rev-parse --git-path rebase-apply)" ]; then
    if resolve_generated_conflicts && GIT_EDITOR=true git rebase --continue && git push; then
      echo "git-push-retry: resolved generated-file conflict and pushed on attempt $attempt"
      exit 0
    fi
    git rebase --abort || true
  fi
  echo "git-push-retry: attempt $attempt/$MAX_ATTEMPTS failed"
  if (( attempt < MAX_ATTEMPTS )); then
    sleep "$(( 2 ** attempt ))"
  fi
  attempt=$(( attempt + 1 ))
done

echo "::error::git push to $BRANCH failed after $MAX_ATTEMPTS attempts - the commit was NOT pushed"
exit 1
