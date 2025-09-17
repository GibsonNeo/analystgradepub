# Analyst-Grader (Config-Driven)

Run with:

```bash
python3 runnertype.py
```

Place your `tickers.csv` and `config.yml` in the repo root. Edit `config.yml` to tune weights/freshness.  
Set API keys in your environment:

```bash
export FINNHUB_API_KEY=your_key_here
export FMP_API_KEY=your_key_here
```

Outputs:
- `output/overview.csv` (columns ordered with `final_grade` right after `sector`)
- Per-ticker caches in `cache/` and state in `state/`

Safe to re-run daily. FMP calls are throttled via a ring buffer and failure backoffs.
