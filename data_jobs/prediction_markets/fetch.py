"""Snapshot NFL game markets from Kalshi and Polymarket.

Both venues quote binary "will team X win" contracts whose price *is* an
implied probability — no vig to strip, though there is a bid/ask spread. This
stores midpoint prices in a tidy CSV per pull:

    ts, venue, market_id, game_date, home_team, away_team,
    home_prob, away_prob, volume, liquidity

Endpoints (public, keyless, read-only):
- Kalshi:     https://api.elections.kalshi.com/trade-api/v2
              (series ``KXNFLGAME``; one event per game, one market per side)
- Polymarket: https://gamma-api.polymarket.com
              (NFL tag; one market per game with a two-outcome token pair)

NOTE: written and unit-tested against recorded response shapes; the sandbox
this was developed in has no outbound network, so the first live run happens
in CI (the weekly-report workflow calls it with ``continue-on-error``). If an
API has drifted, the parsers below are the only thing to fix.

Usage
    python3 -m data_jobs.prediction_markets.fetch --sport nfl
    python3 -m data_jobs.prediction_markets.fetch --sport nfl --venue kalshi
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "prediction_markets"

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_NFL_SERIES = "KXNFLGAME"
POLYMARKET_BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 30

COLUMNS = ["ts", "venue", "market_id", "game_date", "home_team", "away_team",
           "home_prob", "away_prob", "volume", "liquidity"]

# Kalshi tickers embed team codes, e.g. KXNFLGAME-25SEP04DALPHI-PHI.
_KALSHI_EVENT_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})([A-Z]{2,3})([A-Z]{2,3})$")


def _mid(bid: float | None, ask: float | None) -> float | None:
    """Midpoint of a cents bid/ask, as a probability."""
    if bid is None or ask is None:
        return None
    if bid <= 0 and ask <= 0:
        return None
    return round((bid + ask) / 2.0 / 100.0, 4)


def fetch_kalshi(session: requests.Session | None = None) -> pd.DataFrame:
    """Open NFL game markets from Kalshi, one row per game."""
    s = session or requests.Session()
    markets, cursor = [], None
    while True:
        params = {"series_ticker": KALSHI_NFL_SERIES, "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = s.get(f"{KALSHI_BASE}/markets", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        markets.extend(payload.get("markets", []))
        cursor = payload.get("cursor")
        if not cursor:
            break

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    # Group per event: each game has one market per team ("PHI wins", "DAL wins").
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        by_event.setdefault(m.get("event_ticker", m.get("ticker", "?")), []).append(m)

    for event, ms in by_event.items():
        match = _KALSHI_EVENT_RE.search(event)
        probs = {}
        for m in ms:
            side = m.get("ticker", "").rsplit("-", 1)[-1]
            probs[side] = _mid(m.get("yes_bid"), m.get("yes_ask"))
        if match:
            _, away, home = match.groups()  # ticker order is away-then-home
        elif len(probs) == 2:
            away, home = sorted(probs)
        else:
            continue
        rows.append({
            "ts": ts, "venue": "kalshi", "market_id": event,
            "game_date": ms[0].get("expected_expiration_time", "")[:10],
            "home_team": home, "away_team": away,
            "home_prob": probs.get(home), "away_prob": probs.get(away),
            "volume": sum(m.get("volume") or 0 for m in ms),
            "liquidity": sum(m.get("liquidity") or 0 for m in ms),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_polymarket(session: requests.Session | None = None) -> pd.DataFrame:
    """Open NFL game markets from Polymarket's Gamma API."""
    s = session or requests.Session()
    rows, offset = [], 0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    while True:
        r = s.get(f"{POLYMARKET_BASE}/markets", params={
            "tag": "nfl", "closed": "false", "limit": 100, "offset": offset,
        }, timeout=TIMEOUT)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        for m in page:
            outcomes = m.get("outcomes") or []
            prices = m.get("outcomePrices") or []
            if isinstance(outcomes, str):  # gamma sometimes returns JSON strings
                import json
                outcomes, prices = json.loads(outcomes), json.loads(prices)
            if len(outcomes) != 2 or len(prices) != 2:
                continue
            # Game markets are titled "AwayTeam vs. HomeTeam" or "X @ Y".
            title = m.get("question") or m.get("title") or ""
            rows.append({
                "ts": ts, "venue": "polymarket",
                "market_id": m.get("slug") or m.get("id"),
                "game_date": (m.get("gameStartTime") or m.get("endDate") or "")[:10],
                "home_team": outcomes[1], "away_team": outcomes[0],
                "home_prob": round(float(prices[1]), 4),
                "away_prob": round(float(prices[0]), 4),
                "volume": m.get("volumeNum") or m.get("volume"),
                "liquidity": m.get("liquidityNum") or m.get("liquidity"),
            })
        offset += 100
        if len(page) < 100:
            break
    return pd.DataFrame(rows, columns=COLUMNS)


VENUES = {"kalshi": fetch_kalshi, "polymarket": fetch_polymarket}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sport", default="nfl", choices=["nfl"])
    ap.add_argument("--venue", default="all", choices=["all", *VENUES])
    args = ap.parse_args()

    venues = list(VENUES) if args.venue == "all" else [args.venue]
    frames, failures = [], []
    for v in venues:
        try:
            d = VENUES[v]()
            print(f"{v}: {len(d)} markets")
            if len(d):
                frames.append(d)
        except Exception as exc:  # one venue down shouldn't kill the other
            failures.append(v)
            print(f"{v}: FAILED ({exc})")

    if frames:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        path = OUT_DIR / f"pm_{args.sport}_{stamp}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    if failures and not frames:
        raise SystemExit(f"all venues failed: {failures}")


if __name__ == "__main__":
    main()
