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

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  if git pull --rebase origin "$BRANCH" && git push; then
    echo "git-push-retry: push succeeded on attempt $attempt"
    exit 0
  fi
  echo "git-push-retry: attempt $attempt/$MAX_ATTEMPTS failed"
  if (( attempt < MAX_ATTEMPTS )); then
    sleep "$(( 2 ** attempt ))"
  fi
  attempt=$(( attempt + 1 ))
done

echo "::error::git push to $BRANCH failed after $MAX_ATTEMPTS attempts - the commit was NOT pushed"
exit 1
