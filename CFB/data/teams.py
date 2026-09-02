"""Team-name canonicalization and conference display names for the CFB spine.

ESPN (via cfbfastR-data) renamed a handful of FBS programs partway through
its history, which would split one program's Elo into two identities.
ALIASES maps every historical spelling to the current one; the fetch step
applies it, so `games.csv` — and everything downstream — sees one name per
program. Same posture as `soccer/clubs/data/leagues.ALIASES`.

FCS opponents are pooled by the Elo engine (one synthetic "FCS" rating), so
their spellings never need reconciling; a program that later moves up to
FBS (App State, James Madison, …) simply enters the FBS pool under its
current name on its first FBS game.
"""

ALIASES: dict[str, str] = {
    "Connecticut": "UConn",
    "Louisiana Monroe": "UL Monroe",
    "Southern Mississippi": "Southern Miss",
    "UMass": "Massachusetts",
    "UT San Antonio": "UTSA",
}

# Short labels for email/site tables; anything missing passes through.
CONFERENCE_SHORT: dict[str, str] = {
    "American Athletic": "American",
    "Conference USA": "C-USA",
    "FBS Independents": "Ind.",
    "Mid-American": "MAC",
    "Mountain West": "MWC",
}

# Synthetic pooled opponent for every non-FBS program (plan §3.4).
FCS_POOL = "FCS"
FBS = "fbs"


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def conference_short(name: str | None) -> str:
    if not name or name != name:  # None / NaN
        return "—"
    return CONFERENCE_SHORT.get(name, name)
