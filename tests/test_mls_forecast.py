"""Tests for the MLS season structure and the rest-of-season forecast."""

import numpy as np
import pandas as pd
import pytest

from soccer.clubs.data import mls
from soccer.clubs.data.mls import EAST, WEST
from soccer.clubs.model import mls_forecast as F
from soccer.clubs.model.elo import DATA_DIR


@pytest.fixture(scope="module")
def results():
    return pd.read_csv(DATA_DIR / "results.csv")


@pytest.fixture(scope="module")
def season(results):
    """The latest MLS season in the committed results."""
    return str(results[results["league"] == "mls"]["season"].max())


def synthetic_season(played_per_pair=1):
    """A toy MLS season: every ordered intra-conference pair played once
    (so exactly half the round robin is done) and no cross-conference
    matches at all. Deterministic 1-0 home wins, so the table is known."""
    rows = []
    for group in (EAST, WEST):
        for i, home in enumerate(group):
            for away in group[i + 1:]:
                rows.append({
                    "date": "2026-03-01", "season": "2026", "league": "mls",
                    "home_team": home, "away_team": away,
                    "home_score": 1.0, "away_score": 0.0,
                })
    return pd.DataFrame(rows)


class TestAlignment:
    def test_thirty_clubs_split_evenly(self):
        assert len(EAST) == len(WEST) == 15
        assert len(mls.CONFERENCE) == 30
        assert not set(EAST) & set(WEST)

    def test_conference_lookup_is_blank_for_unknown_clubs(self):
        assert mls.conference("Inter Miami CF") == "East"
        assert mls.conference("LA Galaxy") == "West"
        assert mls.conference("Wrexham AFC") == ""

    def test_season_shape_constants_are_self_consistent(self):
        assert mls.INTRA_MATCHES == 2 * (len(EAST) - 1)
        assert mls.INTER_HOME + mls.INTER_AWAY == mls.INTER_MATCHES
        assert mls.INTRA_MATCHES + mls.INTER_MATCHES == mls.MATCHES_PER_CLUB


class TestSeasonResults:
    def test_playoff_matches_are_excluded(self, results):
        """A completed season must come back as exactly the round robin:
        30 clubs x 34 matches / 2. Upstream mixes playoff matches into the
        same log, so this is the check that they are being stripped."""
        reg = mls.season_results(results, "2025")
        assert len(reg) == 30 * mls.MATCHES_PER_CLUB // 2
        counts = pd.concat([reg["home_team"], reg["away_team"]]).value_counts()
        assert set(counts) == {mls.MATCHES_PER_CLUB}

    def test_regular_season_stops_before_the_playoffs(self, results):
        reg = mls.season_results(results, "2025")
        all_2025 = results[(results["league"] == "mls")
                           & (results["season"] == "2025")]
        dropped = all_2025.loc[~all_2025.index.isin(reg.index)]
        assert len(dropped) == 30  # 2025's playoff bracket
        assert dropped["date"].min() > reg["date"].max()

    def test_unplayed_rows_are_ignored(self, results):
        frame = pd.concat([
            results,
            pd.DataFrame([{"date": "2026-11-01", "season": "2026",
                           "league": "mls", "home_team": "LA Galaxy",
                           "away_team": "Austin FC", "home_score": None,
                           "away_score": None}]),
        ], ignore_index=True)
        assert len(mls.season_results(frame, "2026")) == \
            len(mls.season_results(results, "2026"))


class TestStandings:
    def test_points_and_goals_from_a_known_table(self):
        df = synthetic_season()
        table = mls.standings(df, "2026")
        # Group leader hosted 14 matches and won them all; the last club
        # in each group only ever travelled.
        assert table[EAST[0]].points == 14 * mls.WIN_POINTS
        assert table[EAST[0]].wins == 14
        assert table[EAST[0]].goal_diff == 14
        assert table[EAST[-1]].points == 0
        assert table[EAST[-1]].goal_diff == -14

    def test_tiebreakers_run_points_wins_gd_goals_for(self):
        a = mls.Standing("a", 10, 15, 4, 12, 8)   # 15pts, 4W, +4, 12GF
        b = mls.Standing("b", 10, 15, 5, 10, 9)   # 15pts, 5W -> ahead of a
        c = mls.Standing("c", 10, 15, 4, 12, 9)   # 15pts, 4W, +3 -> behind a
        d = mls.Standing("d", 10, 15, 4, 11, 7)   # 15pts, 4W, +4, 11GF
        order = [s.team for s in sorted([a, b, c, d], key=lambda s: s.sort_key())]
        assert order == ["b", "a", "d", "c"]


class TestRemainingSchedule:
    def test_half_a_round_robin_leaves_the_reverse_fixtures(self):
        df = synthetic_season()
        rem = mls.remaining_fixtures(df, "2026")
        assert len(rem.intra) == 2 * len(EAST) * (len(EAST) - 1) // 2
        # Every remaining fixture is the reverse of one already played.
        played = {(r.home_team, r.away_team) for r in df.itertuples()}
        assert all((a, h) in played for h, a in rem.intra)
        # No cross-conference match was played, so all 6 are still owed.
        assert set(rem.inter_home.values()) == {mls.INTER_HOME}
        assert set(rem.inter_away.values()) == {mls.INTER_AWAY}

    def test_live_season_reconstruction_lands_every_club_on_34(
            self, results, season):
        assert mls.verify_structure(results, season) == []
        rem = mls.remaining_fixtures(results, season)
        table = mls.standings(results, season)
        home_left = pd.Series([h for h, _ in rem.intra]).value_counts()
        away_left = pd.Series([a for _, a in rem.intra]).value_counts()
        for club in mls.CONFERENCE:
            total = (table[club].played + home_left.get(club, 0)
                     + away_left.get(club, 0) + rem.inter_home[club]
                     + rem.inter_away[club])
            assert total == mls.MATCHES_PER_CLUB, club

    def test_cross_conference_quotas_balance(self, results, season):
        rem = mls.remaining_fixtures(results, season)
        assert (sum(rem.inter_home[t] for t in EAST)
                == sum(rem.inter_away[t] for t in WEST))
        assert (sum(rem.inter_home[t] for t in WEST)
                == sum(rem.inter_away[t] for t in EAST))

    def test_completed_season_has_nothing_left(self, results):
        assert len(mls.remaining_fixtures(results, "2025")) == 0


class TestVerifyStructure:
    def test_clean_on_the_thirty_club_seasons(self, results, season):
        assert mls.verify_structure(results, "2025") == []
        assert mls.verify_structure(results, season) == []

    def test_rejects_a_season_with_a_different_shape(self, results):
        """2023 and 2024 were 29-club seasons with uneven conferences —
        the constants here do not describe them, and the check has to say
        so rather than quietly producing a reconstruction."""
        assert mls.verify_structure(results, "2024") != []

    def test_reports_unknown_clubs(self, results):
        df = synthetic_season()
        df.loc[0, "home_team"] = "Las Vegas Villains FC"
        problems = mls.verify_structure(df, "2026")
        assert any("conference map" in p for p in problems)


class TestInterDraw:
    def test_sampled_pairing_matches_the_quotas(self, results, season):
        rem = mls.remaining_fixtures(results, season)
        rng = np.random.default_rng(3)
        for _ in range(20):
            drawn = F._sample_inter(rng, rem)
            assert len(drawn) == rem.inter_count
            home = pd.Series([h for h, _ in drawn]).value_counts()
            away = pd.Series([a for _, a in drawn]).value_counts()
            for club in mls.CONFERENCE:
                assert home.get(club, 0) == rem.inter_home[club]
                assert away.get(club, 0) == rem.inter_away[club]
            # Always across the conference divide, never within it.
            assert all(mls.conference(h) != mls.conference(a) for h, a in drawn)

    def test_draws_differ_between_simulations(self, results, season):
        rem = mls.remaining_fixtures(results, season)
        rng = np.random.default_rng(3)
        draws = {tuple(sorted(F._sample_inter(rng, rem))) for _ in range(10)}
        assert len(draws) > 1


class TestScoreTable:
    def test_rows_are_valid_cumulative_distributions(self):
        params = F.scoring.ScoreParams(-2.0, 4.2, {"mls": 3.1})
        cum = F._score_table(params, bins=20)
        assert cum.shape == (20, (F.MAX_GOALS + 1) ** 2)
        assert np.all(np.diff(cum, axis=1) >= -1e-12)   # non-decreasing
        assert np.allclose(cum[:, -1], 1.0)

    def test_a_stronger_home_side_gets_more_goals(self):
        params = F.scoring.ScoreParams(-2.0, 4.2, {"mls": 3.1})
        cum = F._score_table(params, bins=100)
        width = F.MAX_GOALS + 1
        def mean_margin(row):
            probs = np.diff(np.concatenate([[0.0], row]))
            idx = np.arange(len(probs))
            return ((idx // width) - (idx % width)) @ probs
        assert mean_margin(cum[80]) > mean_margin(cum[50]) > mean_margin(cum[20])


class TestBracket:
    """The bracket helpers, driven by a stub `play` so the outcome of each
    match is dictated rather than sampled."""

    @staticmethod
    def _play_factory(scores):
        """`scores` is a list of (home_goals, away_goals) consumed in order."""
        it = iter(scores)
        def play(h, a, ratings, u):
            hs, as_ = next(it)
            return hs, as_, 0.0
        return play

    def test_series_ends_early_when_one_side_wins_the_first_two(self):
        play = self._play_factory([(2, 0), (0, 3)])  # high wins G1 and G2
        rng = np.random.default_rng(0)
        assert F._series(1, 2, {}, rng, play) == 1

    def test_series_goes_to_a_decider_when_the_first_two_split(self):
        # G1 high wins at home, G2 low wins at home, G3 low wins away.
        play = self._play_factory([(2, 0), (1, 0), (0, 1)])
        rng = np.random.default_rng(0)
        assert F._series(1, 2, {}, rng, play) == 2

    def test_a_drawn_knockout_is_decided_and_never_left_level(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            play = self._play_factory([(1, 1)])
            assert F._knockout(1, 2, {}, rng, play) in (1, 2)

    def test_shootouts_are_a_coin_flip(self):
        rng = np.random.default_rng(11)
        wins = sum(
            F._knockout(1, 2, {}, rng, self._play_factory([(0, 0)])) == 1
            for _ in range(4000)
        )
        assert 0.46 < wins / 4000 < 0.54


@pytest.fixture(scope="module")
def out(results, season):
    """One shared 400-sim forecast of the live season for the shape and
    consistency assertions below."""
    return F.forecast(results, season, n_sims=400, seed=1)


class TestForecast:

    def test_every_club_appears_once(self, out):
        teams = [c["team"] for c in out["clubs"]]
        assert len(teams) == 30
        assert set(teams) == set(mls.CONFERENCE)

    def test_probabilities_sum_to_one_where_they_must(self, out):
        total = lambda key: sum(c[key] for c in out["clubs"])
        assert total("p_shield") == pytest.approx(1.0, abs=0.02)
        assert total("p_cup") == pytest.approx(1.0, abs=0.02)
        # One champion per conference, one top seed per conference, and
        # PLAYOFF_SPOTS qualifiers per conference.
        assert total("p_conf_title") == pytest.approx(2.0, abs=0.02)
        assert total("p_top_seed") == pytest.approx(2.0, abs=0.02)
        assert total("p_playoffs") == pytest.approx(
            2 * mls.PLAYOFF_SPOTS, abs=0.05)

    def test_probabilities_nest(self, out):
        """Winning MLS Cup implies reaching it implies reaching the
        conference final implies the semifinal implies the playoffs."""
        for c in out["clubs"]:
            assert c["p_cup"] <= c["p_conf_title"] + 1e-9
            assert c["p_conf_title"] <= c["p_conf_final"] + 1e-9
            assert c["p_conf_final"] <= c["p_conf_semi"] + 1e-9
            assert c["p_conf_semi"] <= c["p_playoffs"] + 1e-9
            assert c["p_shield"] <= c["p_playoffs"] + 1e-9

    def test_seed_distribution_is_a_distribution(self, out):
        for c in out["clubs"]:
            dist = c["seed_distribution"]
            assert len(dist) == mls.PLAYOFF_SPOTS + 1
            assert sum(dist) == pytest.approx(1.0, abs=1e-6)
            # The last bucket is "missed the playoffs".
            assert sum(dist[:-1]) == pytest.approx(c["p_playoffs"], abs=1e-6)

    def test_expected_points_never_fall_below_points_banked(self, out):
        for c in out["clubs"]:
            assert c["exp_points"] >= c["points"]
            ceiling = c["points"] + mls.WIN_POINTS * c["remaining"]
            assert c["exp_points"] <= ceiling

    def test_remaining_matches_are_consistent(self, out):
        assert (out["schedule"]["intra_conference"]
                + out["schedule"]["cross_conference"]
                == out["remaining_matches"])
        assert out["remaining_matches"] == sum(
            c["remaining"] for c in out["clubs"]) // 2

    def test_same_seed_reproduces(self, results, season):
        a = F.forecast(results, season, n_sims=120, seed=5)
        b = F.forecast(results, season, n_sims=120, seed=5)
        assert [c["p_cup"] for c in a["clubs"]] == [c["p_cup"] for c in b["clubs"]]

    def test_refuses_a_season_it_cannot_reconstruct(self, results):
        with pytest.raises(ValueError, match="does not match the format"):
            F.forecast(results, "2024", n_sims=10)


class TestChart:
    """The chart payload: Elo by matches played, history then projection."""

    def test_history_covers_every_club_and_counts_matches(self, out, results,
                                                          season):
        hist = out["chart"]["history"]
        table = mls.standings(results, season)
        assert set(hist) == set(mls.CONFERENCE)
        for club, points in hist.items():
            # One point per match played, plus the live rating on the end.
            assert len(points) == table[club].played + 1
            assert [x for x, _ in points] == list(range(len(points)))

    def test_projection_continues_the_history_line(self, out):
        """The first projected point must be exactly where the history
        ends — same match count, same rating — or the site draws a
        floating second series instead of a continuation."""
        hist = out["chart"]["history"]
        proj = out["chart"]["projection"]
        for club, p in proj.items():
            first = p["points"][0]
            assert [first[0], first[1]] == hist[club][-1]
            # The band is degenerate at the anchor: it is today's rating.
            assert first[1] == first[2] == first[3]

    def test_projection_ends_on_a_full_season(self, out):
        for club, p in out["chart"]["projection"].items():
            xs = [x for x, *_ in p["points"]]
            assert xs[-1] == mls.MATCHES_PER_CLUB, club
            assert xs == sorted(xs)
            assert len(set(xs)) == len(xs)  # strictly increasing

    def test_band_is_oriented_and_tracks_the_median_run(self, out):
        """The median run is picked by where a season *ends*, so at an
        intermediate checkpoint it can sit just outside its own 10-90
        band. On the committed 20k-sim artifact the worst excursion is
        7 Elo; the allowance here is wider only because the fixture runs
        far fewer sims."""
        for club, p in out["chart"]["projection"].items():
            for x, median, lo, hi in p["points"]:
                assert lo <= hi
                assert lo - 25 <= median <= hi + 25, (club, x)

    def test_the_band_widens_as_the_season_runs_out(self, out):
        """Uncertainty about a club's rating grows with every match left
        to play. If this ever inverted, the percentiles would be being
        computed across the wrong axis."""
        widths = [
            [hi - lo for _, _, lo, hi in p["points"]]
            for p in out["chart"]["projection"].values()
        ]
        n = min(len(w) for w in widths)
        mean = [sum(w[i] for w in widths) / len(widths) for i in range(n)]
        assert mean[1] < mean[n - 1]
        assert mean[0] == 0.0  # the anchor is today's rating, not a range

    def test_samples_are_whole_seasons(self, out):
        for club, p in out["chart"]["projection"].items():
            for path in p["samples"]:
                assert len(path) == len(p["points"])

    def test_x_axis_is_matches_not_dates(self, out):
        assert out["chart"]["x_axis"] == "matches_played"
        assert out["chart"]["season_matches"] == mls.MATCHES_PER_CLUB


class TestBracketPayload:
    def test_every_conference_has_nine_seed_slots(self, out):
        for conf in ("East", "West"):
            slots = out["bracket"]["conferences"][conf]["seeds"]
            assert [s["seed"] for s in slots] == list(
                range(1, mls.PLAYOFF_SPOTS + 1))

    def test_candidates_are_ranked_and_in_conference(self, out):
        for conf, members in (("East", EAST), ("West", WEST)):
            for slot in out["bracket"]["conferences"][conf]["seeds"]:
                ps = [c["p"] for c in slot["candidates"]]
                assert ps == sorted(ps, reverse=True)
                assert all(c["team"] in members for c in slot["candidates"])

    def test_conference_favorite_matches_the_club_rows(self, out):
        rows = {c["team"]: c for c in out["clubs"]}
        for conf in ("East", "West"):
            side = out["bracket"]["conferences"][conf]
            best = max((c for c in out["clubs"] if c["conference"] == conf),
                       key=lambda c: c["p_conf_title"])
            assert side["favorite"] == best["team"]
            assert side["p_favorite"] == rows[side["favorite"]]["p_conf_title"]

    def test_finals_pair_one_club_from_each_conference(self, out):
        finals = out["bracket"]["finals"]
        assert finals, "no MLS Cup matchups reported"
        ps = [f["p"] for f in finals]
        assert ps == sorted(ps, reverse=True)
        for f in finals:
            assert f["east"] in EAST
            assert f["west"] in WEST
        assert sum(ps) <= 1.0 + 1e-9

    def test_seeding_by_expected_finish_is_a_permutation(self, out):
        """The site fills bracket slots by expected conference finish, so
        that ordering has to give each club exactly one slot — which per-slot
        modes do not (one club can lead two slots at once)."""
        for conf in ("East", "West"):
            order = sorted((c for c in out["clubs"] if c["conference"] == conf),
                           key=lambda c: c["exp_conf_seed"])
            assert len({c["team"] for c in order}) == len(order) == 15


class TestBacktest:
    def test_as_of_hides_the_future(self, results):
        from soccer.clubs.model import backtest_mls as B
        frame = B.as_of(results, "2025", "2025-07-26")
        mls_rows = frame[frame["league"] == "mls"]
        assert mls_rows["date"].max() <= "2025-07-26"
        # Other leagues are untouched — only the MLS pool is being rewound.
        assert (frame[frame["league"] == "epl"].shape[0]
                == results[results["league"] == "epl"].shape[0])

    def test_actual_outcomes_reads_the_real_season(self, results):
        from soccer.clubs.model import backtest_mls as B
        truth = B.actual_outcomes(results, "2025")
        assert truth["shield"] == "Philadelphia Union"
        assert truth["cup"] == "Inter Miami CF"
        assert len(truth["qualified"]) == 2 * mls.PLAYOFF_SPOTS
        assert truth["playoff_matches"] == 30
