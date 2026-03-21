# Progress

## Done
- 2026-03-21: Skills upgrade — restructured to src/ layout, migrated DB to standard schema
- 2026-03-21: Added standard update_data.py CLI and launch.json configs

## In Progress
- Skills upgrade validation — testing Streamlit app and pricing

## Next
- Add smoke tests for pricing and instruments
- Production test coverage (80%+)
- Add pre-commit hooks (ruff + mypy)

## Decisions
- 2026-03-21: Kept legacy DB tables alongside standard schema for backward compatibility
- 2026-03-21: Vol surface interpolation method: RBF (for irregular/sparse SPX data)
- 2026-03-21: Using parent Poetry environment (ale/pyproject.toml)

## References
- Goldman-Sosin-Gatto (1979): floating strike lookback formulas
- Conze-Viswanathan (1991): fixed strike lookback formulas
- SSVI (Gatheral & Jacquier): arbitrage-free vol surface parameterization
