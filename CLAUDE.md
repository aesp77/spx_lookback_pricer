# CLAUDE.md — SPX Lookback Pricer

## Shared Skills

Before starting any work, read the relevant skills from `~/skills/skills/`.
Skills are located at `/c/source/repos/ale/skills/skills/`.

### Always read
- ~/skills/skills/project-scaffold/SKILL.md
- ~/skills/skills/env-setup/SKILL.md
- ~/skills/skills/git-workflow/SKILL.md

### Read for data work
- ~/skills/skills/market-data/SKILL.md
- ~/skills/skills/edav/SKILL.md
- ~/skills/skills/experiment-logging/SKILL.md

### Read for quant/finance work
- ~/skills/skills/vol-and-curves/SKILL.md
- ~/skills/skills/pricing/SKILL.md
- ~/skills/skills/quant-patterns/SKILL.md

### Read for testing
- ~/skills/skills/testing-conventions/SKILL.md

## Commands

### "init" — New Project
See shared skills for init workflow.

### "upgrade" — Existing Project
See shared skills for upgrade workflow. Always use feature branches.

## Keeping this file and PROGRESS.md up to date

**CLAUDE.md** is the source of truth for project configuration.
**PROGRESS.md** is the source of truth for what's been done and what's next.

## Project Rules

1. All pricing is custom-coded using scipy.stats.norm — no QuantLib or py_vollib.
2. Greeks: analytical where available, bump-and-reprice otherwise.
3. Standard bump sizes: 1% spot, 1% vol, 1bp rate, 1 day time.
4. Monte Carlo minimum 100,000 paths with antithetic variates.
5. Vol surface interpolation in log-moneyness x time space (RBF method for sparse data).
6. Rate curves: natural cubic spline with flat extrapolation.
7. Time conventions: 252 trading days for vol, 365 days for rates.
8. Discount factors: continuous compounding exp(-r * T).
9. Database uses standard market-data skill schema (time_series, vol_surfaces, term_structures, fetch_log).
10. All SQL queries use parameterised placeholders (?), never f-strings.
11. Uses parent Poetry environment (ale/pyproject.toml) — no local pyproject.toml.
12. Credentials in .env only — never in code.

## Architecture

```
spx_lookback_pricer/
├── CLAUDE.md
├── PROGRESS.md
├── README.md
├── .env.example
├── .gitignore
├── config/
│   └── config.yaml
├── src/
│   └── spx_lookback_pricer/
│       ├── __init__.py
│       ├── data/
│       │   ├── market_data.py      # SPXDataLoader (SQLite + PSC/BBG/Marquee)
│       │   └── vol_surface.py      # VolatilitySurface interpolation
│       ├── instruments/
│       │   ├── lookback.py         # Percentage, ratcheting, fixed-strike lookbacks
│       │   ├── lookback2.py        # Alternative lookback implementations
│       │   └── vanilla.py          # European/American vanilla options
│       ├── models/
│       │   ├── base_model.py       # Abstract BasePricingModel
│       │   ├── black_scholes.py    # BS with extensions (local vol, Heston)
│       │   └── monte_carlo.py      # MC engine with variance reduction
│       ├── pricing/
│       │   ├── analytical_pricer.py # Closed-form formulas
│       │   ├── mc_pricer.py        # MC pricing wrapper
│       │   └── pde_pricer.py       # Finite difference PDE solver
│       └── utils/
│           ├── db_manager.py       # Database operations
│           ├── greek_calculator.py # Numerical Greeks
│           ├── interpolation.py    # RBF, spline, bilinear
│           └── math_utils.py       # BS formulas, normal dist helpers
├── app/
│   └── streamlit_app.py           # Interactive pricing dashboard
├── scripts/
│   ├── update_data.py             # Standard CLI: status/update/copy/init
│   ├── update_database.py         # Legacy update script
│   └── migrate_db.py              # One-time DB migration
├── data/
│   └── db/
│       └── spx_lookback_pricer.db # SQLite (standard schema)
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── notebooks/
├── output/
└── requirements.txt               # Legacy — use parent Poetry env
```

## Current State

- Branch: `chore/skills-upgrade` — upgrading to skills compliance
- DB migrated to standard schema with legacy tables preserved
- Streamlit app at app/streamlit_app.py with updated imports
- Data from Oct 2025: 3 spot prices, 440 vol surface points, 160 OIS curve points, 4 SSVI param sets

## Do NOT

- Do not use QuantLib or py_vollib — all pricing from scratch
- Do not create a local pyproject.toml — use parent ale/ Poetry environment
- Do not delete legacy DB tables — they coexist with standard schema
- Do not change the get_latest_data() return structure — Streamlit app depends on it
- Do not use f-strings in SQL queries
- Do not commit .env or .db files
