# NBA Actual Game Results

Place CSV files with actual game results here to enable:
- Bookmaker accuracy tracking (which sportsbook predicts winners best)
- Underdog wins highlighting
- Historical performance analysis

## Supported CSV Formats

### Format 1: Basketball Reference (Recommended)

Download schedule/results from Basketball Reference. The format is:

| Date | Start (ET) | Visitor/Neutral | PTS | Home/Neutral | PTS | Attend | LOG | Arena | Notes |
|------|------------|-----------------|-----|--------------|-----|--------|-----|-------|-------|
| Mon, Oct 28, 2025 | 7:00p | Boston Celtics | 108 | New York Knicks | 112 | 19,812 | | Madison Square Garden | |

### Format 2: Simple Format

A simpler custom format:

| Date | Home Team | Away Team | Home Score | Away Score |
|------|-----------|-----------|------------|------------|
| 2026-01-15 | Los Angeles Lakers | Boston Celtics | 112 | 108 |

## File Naming

Name files by month for easy organization:
- `october_2025.csv`
- `november_2025.csv`
- `december_2025.csv`
- etc.

The report generator will automatically load all CSV files from this directory.

## Team Name Matching

The team names in your results files should match the full names used in the odds data. Examples:
- "Boston Celtics" (not "BOS" or "Celtics")
- "Los Angeles Lakers" (not "LA Lakers" or "LAL")
- "Golden State Warriors" (not "GS Warriors")

Check `data/nba/nba_odds_api_data_latest.csv` for the exact team names used in odds data.

## Notes

- Empty rows or rows with missing scores are automatically skipped
- Dates are parsed automatically (supports multiple formats)
- The system tracks which bookmaker's favorite won each game
