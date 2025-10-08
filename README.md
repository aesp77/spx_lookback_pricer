# SPX Lookback Option Pricer

A comprehensive Python library for pricing lookback options on the S&P 500 index (SPX) using analytical formulas and Monte Carlo simulation.

## Features

- **Multiple Pricing Methods**
  - Analytical pricing using closed-form formulas
  - Monte Carlo simulation with variance reduction techniques
  - PDE-based pricing (optional)

- **Instrument Support**
  - Fixed strike lookback options (calls and puts)
  - Floating strike lookback options (calls and puts)
  - Vanilla European options (for calibration and benchmarking)
  - Partial lookback options

- **Market Data Management**
  - Flexible market data containers
  - Interest rate curves
  - Dividend schedules (discrete and continuous)
  - Volatility surface interpolation (RBF, bilinear, cubic spline)

- **Greek Calculation**
  - Analytical Greeks for vanilla options
  - Numerical Greeks via finite differences
  - Delta, Gamma, Vega, Theta, Rho
  - Second-order Greeks (Vanna, Volga)

- **Visualization**
  - Interactive Streamlit dashboard
  - Sensitivity analysis charts
  - Convergence studies
  - Volatility surface visualization

## Installation

1. Clone the repository or copy the `spx_lookback_pricer` directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Pricing Example

```python
from data.market_data import MarketData
from instruments.lookback import FixedStrikeLookback
from pricing.analytical_pricer import AnalyticalPricer

# Create market data
market_data = MarketData.from_flat_rates(
    spot=4500.0,
    risk_free_rate=0.05,
    dividend_yield=0.015
)

# Create a fixed strike lookback call
option = FixedStrikeLookback(
    strike=4500.0,
    expiry=1.0,
    is_call=True
)

# Price the option
pricer = AnalyticalPricer()
result = pricer.price(option, market_data, volatility=0.20)

print(f"Price: ${result.price:.4f}")
print(f"Delta: {result.delta:.4f}")
print(f"Gamma: {result.gamma:.6f}")
```

### Monte Carlo Pricing

```python
from pricing.mc_pricer import MonteCarloPricer

# Create Monte Carlo pricer
mc_pricer = MonteCarloPricer(
    num_paths=50000,
    num_steps=252,
    use_antithetic=True
)

# Price the option
result = mc_pricer.price(option, market_data, volatility=0.20)

print(f"MC Price: ${result.price:.4f}")
print(f"Std Error: ${result.std_error:.4f}")
print(f"95% CI: [{result.confidence_interval[0]:.4f}, "
      f"{result.confidence_interval[1]:.4f}]")
```

### Using the Streamlit Dashboard

```bash
streamlit run visualization/streamlit_app.py
```

This launches an interactive web interface for:
- Pricing lookback options
- Sensitivity analysis
- Convergence studies
- Parameter exploration

## Project Structure

```
spx_lookback_pricer/
│
├── data/                   # Market data management
│   ├── vol_surface.py      # Volatility surface interpolation
│   └── market_data.py      # Spot, rates, dividends
│
├── models/                 # Pricing models
│   ├── base_model.py       # Abstract base class
│   ├── black_scholes.py    # BS dynamics
│   └── monte_carlo.py      # MC engine
│
├── instruments/            # Option instruments
│   ├── lookback.py         # Lookback options
│   └── vanilla.py          # Vanilla options
│
├── pricing/                # Pricing engines
│   ├── analytical_pricer.py
│   ├── mc_pricer.py
│   └── pde_pricer.py
│
├── utils/                  # Utilities
│   ├── interpolation.py    # Interpolation methods
│   ├── math_utils.py       # Black-Scholes formulas
│   └── greek_calculator.py # Numerical Greeks
│
├── visualization/          # Dashboards
│   └── streamlit_app.py    # Streamlit interface
│
├── tests/                  # Unit tests
│   ├── test_pricing.py
│   ├── test_vol_surface.py
│   └── test_notebook.ipynb
│
├── config/                 # Configuration
│   └── config.yaml
│
└── requirements.txt        # Dependencies
```

## Key Concepts

### Lookback Options

**Fixed Strike Lookback Options:**
- Call payoff: max(M - K, 0) where M is the maximum price during the lookback period
- Put payoff: max(K - m, 0) where m is the minimum price during the lookback period

**Floating Strike Lookback Options:**
- Call payoff: S_T - m where S_T is the final price and m is the minimum
- Put payoff: M - S_T where M is the maximum price

### Analytical Formulas

The library implements analytical formulas from:
- Conze and Viswanathan (1991) for fixed strike lookbacks
- Goldman, Sosin, and Gatto (1979) for floating strike lookbacks

### Monte Carlo Variance Reduction

- **Antithetic variates**: Generate pairs of negatively correlated paths
- **Control variates**: Use vanilla options with known prices
- **Importance sampling**: Focus simulations on important regions

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run coverage analysis:

```bash
pytest tests/ --cov=. --cov-report=html
```

## Configuration

Edit `config/config.yaml` to customize:
- Default market parameters
- Monte Carlo settings
- Greek calculation parameters
- Visualization preferences

## Examples

See `tests/test_notebook.ipynb` for comprehensive examples including:
- Pricing various option types
- Sensitivity analysis
- Convergence studies
- Volatility surface construction
- Greek calculations

## Performance

Typical pricing times (Intel i7, single core):
- Analytical: < 1ms
- Monte Carlo (10k paths): ~100ms
- Monte Carlo (100k paths): ~1s

## Dependencies

- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- plotly >= 5.14.0
- streamlit >= 1.28.0

## License

This project is for educational and research purposes.

## References

1. Conze, A., & Viswanathan (1991). "Path Dependent Options: The Case of Lookback Options"
2. Goldman, M. B., Sosin, H. B., & Gatto, M. A. (1979). "Path Dependent Options: Buy at the Low, Sell at the High"
3. Hull, J. C. (2018). "Options, Futures, and Other Derivatives" (10th Edition)
4. Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering"

## Contact

For questions or contributions, please open an issue on the repository.

---

**Note**: This is a pricing library for educational purposes. Always validate results independently before using in production.
# spx_lookback_pricer


# Database Schema for SPX Lookback Pricer

## Tables Structure

### 1. `spx_spot_prices`
Stores daily SPX spot prices and returns
```sql
CREATE TABLE spx_spot_prices (
    date DATE PRIMARY KEY,
    last FLOAT NOT NULL,
    daily_return FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 2. `spx_ssvi_parameters`
Stores SSVI model parameters for vol surface construction
```sql
CREATE TABLE spx_ssvi_parameters (
    date DATE PRIMARY KEY,
    theta FLOAT NOT NULL,
    rho FLOAT NOT NULL,
    beta FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 3. `spx_vol_surface`
Stores implied volatility surface data
```sql
CREATE TABLE spx_vol_surface (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    strike FLOAT NOT NULL,  -- Relative strike (0.6 to 1.5)
    tenor VARCHAR(10) NOT NULL,  -- '1m', '2m', '3m', etc.
    implied_vol FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_date_strike_tenor (date, strike, tenor),
    INDEX idx_date (date)
);
```

### 4. `spx_dividend_yield`
Stores dividend yield data
```sql
CREATE TABLE spx_dividend_yield (
    date DATE PRIMARY KEY,
    dividend_yield FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 5. `ois_curve`
Stores OIS (risk-free rate) curve data
```sql
CREATE TABLE ois_curve (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    tenor_years FLOAT NOT NULL,  -- Time to maturity in years
    rate FLOAT NOT NULL,  -- Rate in decimal (0.05 = 5%)
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_date_tenor (date, tenor_years),
    INDEX idx_date (date)
);
```

### 6. `spx_master_dataset` (Optional - Denormalized View)
Pre-joined view or materialized table for quick access
```sql
CREATE TABLE spx_master_dataset (
    date DATE PRIMARY KEY,
    spot_price FLOAT,
    daily_return FLOAT,
    dividend_yield FLOAT,
    theta FLOAT,
    rho FLOAT,
    beta FLOAT,
    -- Add specific vol points as needed
    iv_60_1m FLOAT,
    iv_70_1m FLOAT,
    iv_80_1m FLOAT,
    -- ... more columns for each strike/tenor combination
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Database Connection Configuration

### For SQLite (Development)
```python
DATABASE_CONFIG = {
    'type': 'sqlite',
    'path': 'spx_lookback_data.db'
}
```

### For PostgreSQL (Production)
```python
DATABASE_CONFIG = {
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'spx_options',
    'user': 'your_user',
    'password': 'your_password'
}
```

### For your PSC Database
```python
DATABASE_CONFIG = {
    'type': 'psc',
    'environment': 'Prod',  # or 'Dev'
    'use_async': True
}
```

## Data Flow

1. **Initial Load**: Run historical data collection (2020-2025)
2. **Daily Updates**: Automated job to fetch new data points
3. **Query Pattern**: Join tables on date for complete dataset
4. **Caching**: Use materialized views for frequently accessed combinations