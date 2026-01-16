# NBA Actual Game Results

Place CSV files with actual game results here to enable:
- Bookmaker accuracy tracking (which sportsbook predicts winners best)
- Underdog wins highlighting
- Historical performance analysis

## Expected CSV Format

Each CSV should have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| Date | Game date (YYYY-MM-DD) | 2026-01-15 |
| Home Team | Full team name | Los Angeles Lakers |
| Away Team | Full team name | Boston Celtics |
| Home Score | Home team final score | 112 |
| Away Score | Away team final score | 108 |

## Example CSV

```csv
Date,Home Team,Away Team,Home Score,Away Score
2026-01-15,Los Angeles Lakers,Boston Celtics,112,108
2026-01-15,Golden State Warriors,Miami Heat,105,98
2026-01-14,Phoenix Suns,Denver Nuggets,118,122
```

## File Naming

Name files by month for easy organization:
- `january_2026.csv`
- `february_2026.csv`
- etc.

The report generator will automatically load all CSV files from this directory.

## Team Name Matching

Make sure team names match exactly what's in the odds data:
- "Los Angeles Lakers" (not "LA Lakers" or "Lakers")
- "Golden State Warriors" (not "GS Warriors" or "Warriors")

Check `data/nba/nba_odds_api_data_latest.csv` for the exact team names used.
