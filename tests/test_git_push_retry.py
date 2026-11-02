"""End-to-end tests for .github/scripts/git-push-retry.sh and the usage
merge it calls.

Each test builds a bare "origin", two clones, and a race: clone B pushes a
commit first, then clone A commits and runs the retry script, which is
exactly the situation the scheduled jobs hit when they all push around the
same time. The script must rebase and push (no conflict), merge the
generated file and push (known generated file), or exit nonzero with
nothing pushed (a real conflict)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from data_jobs.odds_api.merge_usage import merge_usage

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "git-push-retry.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-q", "-b", "main", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(origin), str(seed))
    (seed / "README").write_text("seed\n")
    (seed / "data").mkdir()
    (seed / "data" / "api_usage.json").write_text(json.dumps({
        "history": [{"timestamp": "2026-11-01T00:00:00", "sport": "nfl",
                     "games_fetched": 1, "requests_remaining": 500, "requests_used": 0}],
        "last_remaining": 500}, indent=2))
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "seed")
    _git(seed, "push", "-q", "-u", "origin", "main")
    a = tmp_path / "a"
    b = tmp_path / "b"
    _git(tmp_path, "clone", "-q", str(origin), str(a))
    _git(tmp_path, "clone", "-q", str(origin), str(b))
    return origin, a, b


def _run_script(cwd: Path, attempts: int = 3) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["bash", str(SCRIPT), "main", str(attempts)], cwd=cwd,
                          env=env, capture_output=True, text=True)


def _commit_file(clone: Path, rel: str, text: str, msg: str) -> None:
    path = clone / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(clone, "add", rel)
    _git(clone, "commit", "-q", "-m", msg)


def _bare_push_is_rejected(clone: Path) -> bool:
    proc = subprocess.run(["git", "push"], cwd=clone, capture_output=True, text=True)
    return proc.returncode != 0 and "fetch first" in (proc.stderr + proc.stdout)


def _remote_files(origin: Path) -> set[str]:
    return set(_git(origin, "ls-tree", "-r", "--name-only", "main").split())


def test_rejected_push_is_rebased_and_pushed(tmp_path):
    origin, a, b = _setup(tmp_path)
    _commit_file(b, "data/odds_b.csv", "b\n", "B: odds")
    _git(b, "push", "-q")
    _commit_file(a, "reports/nfl.html", "a\n", "A: report")
    assert _bare_push_is_rejected(a), "the race must reproduce the real failure first"

    proc = _run_script(a)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "push succeeded" in proc.stdout
    files = _remote_files(origin)
    assert {"data/odds_b.csv", "reports/nfl.html"} <= files
    assert _git(origin, "rev-list", "--count", "main").strip() == "3"


def test_api_usage_conflict_is_merged_not_dropped(tmp_path):
    origin, a, b = _setup(tmp_path)
    base = json.loads((a / "data" / "api_usage.json").read_text())

    def with_entry(ts: str, sport: str, remaining: int) -> str:
        d = json.loads(json.dumps(base))
        d["history"].append({"timestamp": ts, "sport": sport, "games_fetched": 5,
                             "requests_remaining": remaining, "requests_used": 500 - remaining})
        d["last_remaining"] = remaining
        return json.dumps(d, indent=2)

    _commit_file(b, "data/api_usage.json", with_entry("2026-11-02T10:00:00", "nba", 490), "B: usage")
    _git(b, "push", "-q")
    _commit_file(a, "data/api_usage.json", with_entry("2026-11-02T10:00:01", "nfl", 488), "A: usage")
    assert _bare_push_is_rejected(a)

    proc = _run_script(a)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "resolved generated-file conflict" in proc.stdout
    merged_text = _git(origin, "show", "main:data/api_usage.json")
    merged = json.loads(merged_text)
    stamps = {(e["timestamp"], e["sport"]) for e in merged["history"]}
    assert ("2026-11-02T10:00:00", "nba") in stamps
    assert ("2026-11-02T10:00:01", "nfl") in stamps
    assert ("2026-11-01T00:00:00", "nfl") in stamps
    assert merged["last_remaining"] == 488


def test_whole_file_refresh_keeps_this_runs_copy(tmp_path):
    origin, a, b = _setup(tmp_path)
    _commit_file(b, "data/schedules/nflverse_games.csv", "game,score\n1,older\n", "B: schedule")
    _git(b, "push", "-q")
    _commit_file(a, "data/schedules/nflverse_games.csv", "game,score\n1,newer\n", "A: schedule")
    assert _bare_push_is_rejected(a)

    proc = _run_script(a)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "whole-file refresh" in proc.stdout
    assert _git(origin, "show", "main:data/schedules/nflverse_games.csv") == "game,score\n1,newer\n"


def test_real_conflict_fails_loudly_and_pushes_nothing(tmp_path):
    origin, a, b = _setup(tmp_path)
    _commit_file(b, "README", "theirs\n", "B: readme")
    _git(b, "push", "-q")
    _commit_file(a, "README", "ours\n", "A: readme")
    assert _bare_push_is_rejected(a)

    proc = _run_script(a, attempts=2)
    assert proc.returncode != 0
    assert "unresolvable conflict in README" in proc.stdout
    assert "::error::" in proc.stdout
    assert _git(origin, "show", "main:README") == "theirs\n"
    assert _git(origin, "rev-list", "--count", "main").strip() == "2"
    # the working clone is left on its own commit, not mid-rebase
    assert not (a / ".git" / "rebase-merge").exists()


def test_merge_usage_unions_and_caps():
    ours = {"history": [{"timestamp": f"2026-11-01T00:00:{i:02d}", "sport": "nfl",
                         "requests_remaining": 500 - i} for i in range(100)],
            "last_remaining": 401}
    theirs = {"history": [{"timestamp": "2026-11-01T00:01:40", "sport": "nba",
                           "requests_remaining": 399}],
              "last_remaining": 399}
    merged = merge_usage(ours, theirs)
    assert len(merged["history"]) == 100
    assert merged["history"][-1]["sport"] == "nba"
    assert merged["last_remaining"] == 399
    # duplicates collapse
    again = merge_usage(merged, merged)
    assert len(again["history"]) == 100
