"""Tests for the Understat player-season export
(soccer/clubs/data/understat_players.py), on fixtures in the shapes the
league payload has been seen in. The site itself is unreachable from the
development sandbox."""

import json

from soccer.clubs.data import understat_players as up


def _player(pid, name, team, pos, minutes, **over):
    line = {"id": str(pid), "player_name": name, "games": "30", "time": str(minutes),
            "goals": "10", "xG": "9.25", "assists": "5", "xA": "4.1234567",
            "shots": "80", "key_passes": "40", "yellow_cards": "3", "red_cards": "0",
            "position": pos, "team_title": team, "npg": "8", "npxG": "7.7",
            "xGChain": "15.5", "xGBuildup": "5.5"}
    line.update(over)
    return line


def _players():
    return [
        _player(1, "Striker One", "Arsenal", "F S", 2700),
        _player(2, "Mid Two", "Chelsea,Arsenal", "M D S", 1800),
        _player(3, "Keeper Three", "Arsenal", "GK", 3420, goals="0", shots="0"),
        _player(4, "Bench Four", "Arsenal", "S", 40),
        _player(5, "Unused Five", "Arsenal", "D", 0),
    ]


class TestParse:
    def test_list_dict_and_encoded_shapes(self):
        players = _players()
        assert len(up.parse_players({"players": players})) == 5
        assert len(up.parse_players({"players": {p["id"]: p for p in players}})) == 5
        assert len(up.parse_players({"playersData": json.dumps(players)})) == 5

    def test_missing_block_is_empty(self):
        assert up.parse_players({"dates": [], "teams": {}}) == []


class TestPositionGroup:
    def test_first_real_position_wins(self):
        assert up.position_group("F M S") == "FW"
        assert up.position_group("S D M") == "DF"
        assert up.position_group("GK") == "GK"
        assert up.position_group("M") == "MF"

    def test_bench_only_or_blank_has_no_group(self):
        assert up.position_group("S") is None
        assert up.position_group("") is None
        assert up.position_group(None) is None


class TestBuildRows:
    def test_types_and_dropped_lines(self):
        rows = up.build_rows("epl", up.parse_players({"players": _players()}))
        assert [r["id"] for r in rows] == ["3", "1", "2", "4"]   # by minutes; 0-min dropped
        s = next(r for r in rows if r["id"] == "1")
        assert s["league"] == "epl" and s["group"] == "FW" and s["min"] == 2700
        assert s["goals"] == 10 and isinstance(s["goals"], int)
        assert s["xg"] == 9.25 and s["xa"] == 4.123 and s["npxg"] == 7.7
        assert next(r for r in rows if r["id"] == "4")["group"] is None

    def test_mid_season_move_keeps_both_clubs(self):
        rows = up.build_rows("epl", _players())
        assert next(r for r in rows if r["id"] == "2")["team"] == "Chelsea, Arsenal"

    def test_unreadable_number_is_none_not_zero(self):
        rows = up.build_rows("epl", [_player(9, "Odd", "Arsenal", "F", 900, xGChain="n/a")])
        assert rows[0]["xgchain"] is None


class TestWrite:
    def test_season_file_and_index(self, tmp_path):
        rows = up.build_rows("epl", _players())
        assert up.write_season(2024, {"epl": rows}, tmp_path) == 4
        body = json.loads((tmp_path / "2024.json").read_text())
        assert body["season"] == "2024-25" and body["columns"] == up.COLUMNS
        assert len(body["rows"]) == 4 and len(body["rows"][0]) == len(up.COLUMNS)

        index = up.write_index(tmp_path)
        assert index["seasons"] == [{"key": 2024, "label": "2024-25", "file": "2024.json",
                                     "players": 4, "leagues": ["epl"]}]

    def test_failed_league_keeps_its_committed_rows(self, tmp_path):
        epl = up.build_rows("epl", _players())
        liga = up.build_rows("la_liga", [_player(7, "Liga Seven", "Barcelona", "F", 2000)])
        up.write_season(2024, {"epl": epl, "la_liga": liga}, tmp_path)
        # Next run: only EPL answered, with one player fewer.
        up.write_season(2024, {"epl": epl[:2]}, tmp_path)
        body = json.loads((tmp_path / "2024.json").read_text())
        leagues = [dict(zip(body["columns"], r))["league"] for r in body["rows"]]
        assert leagues.count("epl") == 2 and leagues.count("la_liga") == 1

    def test_index_orders_newest_first(self, tmp_path):
        rows = up.build_rows("epl", _players())
        up.write_season(2023, {"epl": rows}, tmp_path)
        up.write_season(2025, {"epl": rows}, tmp_path)
        assert [s["key"] for s in up.write_index(tmp_path)["seasons"]] == [2025, 2023]
