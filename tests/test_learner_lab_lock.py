"""The research harness must not run twice at once.

Its artifact write is read-modify-write, so two runs finishing together
lose one. Worse, the grids are compiled multi-threaded code: two runs on
the same cores do not go half as fast, they collapse — a 2-second
boosting fit took minutes when two sports overlapped on four cores.
"""

from __future__ import annotations

import os

import pytest

from research import learner_lab as lab


def test_a_live_lock_refuses_the_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "OUT", tmp_path)
    lock = tmp_path / ".run.lock"
    lock.write_text(str(os.getpid()))  # this process is certainly alive
    with pytest.raises(SystemExit) as caught:
        with lab._single_run():
            pytest.fail("the second run should not have started")
    assert "another learner_lab run" in str(caught.value)
    assert lock.exists(), "the live run's lock must survive the refusal"


def test_a_stale_lock_is_taken_over(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "OUT", tmp_path)
    lock = tmp_path / ".run.lock"
    lock.write_text("999999")  # a pid that is not running
    with lab._single_run():
        assert lock.read_text() == str(os.getpid())
    assert not lock.exists(), "the lock must be released on the way out"


def test_the_lock_is_released_even_when_the_run_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "OUT", tmp_path)
    with pytest.raises(ValueError):
        with lab._single_run():
            raise ValueError("grid blew up")
    assert not (tmp_path / ".run.lock").exists()


def test_openmp_waits_passively():
    """Set at import time; an active spin-wait is what made two
    concurrent runs pathological rather than merely slow."""
    assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
