# data/market_data.py
"""
Market data loader with database integration for SPX lookback option pricer.
Integrates with PSC internal systems, Bloomberg, and GS Marquee.
"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import asyncio
import nest_asyncio
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager

# PSC internal imports (these will be available in your environment)
try:
    from psccommon.apiclient.PscMarqueeClient import PscMarqueeClient
    from psccommon.db.db import get_db_async
    from xbbg import blp
    PSC_AVAILABLE = True
except ImportError:
    print("Warning: PSC modules not available. Using mock mode.")
    PSC_AVAILABLE = False

nest_asyncio.apply()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: str = 'sqlite'  # 'sqlite', 'postgresql', 'psc'
    db_path: str = 'spx_lookback_data.db'
    psc_env: str = 'Prod'

class SPXDataLoader:
    """
    Comprehensive data loader for SPX options data.
    Handles spot prices, vol surfaces, SSVI parameters, dividends, and OIS curves.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.marquee_client = PscMarqueeClient() if PSC_AVAILABLE else None
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist"""
        if self.config.db_type == 'sqlite':
            with self._get_db_connection() as conn:
                self._create_tables_sqlite(conn)
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection based on config"""
        if self.config.db_type == 'sqlite':
            conn = sqlite3.connect(self.config.db_path)
            try:
                yield conn
            finally:
                conn.close()
        else:
            raise NotImplementedError(f"Database type {self.config.db_type} not yet implemented")
    
    def _create_tables_sqlite(self, conn):
        """Create SQLite tables for storing market data"""
        cursor = conn.cursor()
        
        # Spot prices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spx_spot_prices (
                date DATE PRIMARY KEY,
                last REAL NOT NULL,
                daily_return REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # SSVI parameters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spx_ssvi_parameters (
                date DATE PRIMARY KEY,
                theta REAL NOT NULL,
                rho REAL NOT NULL,
                beta REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Vol surface table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spx_vol_surface (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                strike REAL NOT NULL,
                tenor TEXT NOT NULL,
                implied_vol REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, strike, tenor)
            )
        ''')
        
        # Dividend yield table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spx_dividend_yield (
                date DATE PRIMARY KEY,
                dividend_yield REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # OIS curve table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ois_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                tenor_years REAL NOT NULL,
                rate REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, tenor_years)
            )
        ''')
        
        conn.commit()
    
    # =================== Data Fetching Methods ===================
    
    async def get_marquee_id_from_stock_ticker(self, stock_ticker: str) -> str:
        """Get GS Marquee ID from stock ticker"""
        if not PSC_AVAILABLE:
            return "MA4B66MW5E27U8P32SY"  # Mock SPX ID
        
        length = len(stock_ticker)
        db = await get_db_async(self.config.psc_env)
        sql = "SELECT id, bbid FROM tbl_psc_static_gs"
        db_res = await db.fetch(sql)
        marquee_data = pd.DataFrame([dict(r) for r in db_res])
        result = marquee_data[marquee_data['bbid'].str[:length] == stock_ticker]
        return result.iloc[0]['id'] if not result.empty else None
    
    async def fetch_spot_data(self, ticker: str, begin: str, end: str) -> pd.DataFrame:
        """Fetch spot price data from PSC database"""
        if not PSC_AVAILABLE:
            # Return mock data for testing
            dates = pd.date_range(begin, end, freq='D')
            # Filter to business days only
            dates = dates[dates.weekday < 5]
            return pd.DataFrame({
                'date': dates,
                'last': np.random.uniform(4000, 5000, len(dates)),
                'daily_return': np.random.normal(0, 0.01, len(dates))
            })
        
        print(f"Querying spot prices for {ticker} from {begin} to {end}")
        db = await get_db_async(self.config.psc_env)
        sql = f"""
            SELECT date, last FROM tbl_psc_market_data_spot_prices
            WHERE ticker = '{ticker}' AND date BETWEEN '{begin}' AND '{end}'
            ORDER BY date
        """
        print(f"SQL: {sql}")
        
        result = await db.fetch(sql)
        print(f"Query returned {len(result)} rows")
        
        if not result:
            print(f"No spot data found. Checking what tickers are available...")
            # Debug: Check available tickers
            check_sql = f"""
                SELECT DISTINCT ticker FROM tbl_psc_market_data_spot_prices 
                WHERE date BETWEEN '{begin}' AND '{end}'
                LIMIT 10
            """
            available = await db.fetch(check_sql)
            if available:
                print("Available tickers in date range:")
                for row in available[:5]:
                    print(f"  - {row['ticker']}")
            
            # Also check if the table has any data at all for SPX
            check_spx = f"""
                SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date 
                FROM tbl_psc_market_data_spot_prices 
                WHERE ticker LIKE '%SPX%'
            """
            spx_info = await db.fetch(check_spx)
            if spx_info:
                info = spx_info[0]
                print(f"SPX data in database: {info['cnt']} records from {info['min_date']} to {info['max_date']}")
        
        df = pd.DataFrame([dict(r) for r in result])
        
        if not df.empty:
            df.sort_values('date', inplace=True)
            df['daily_return'] = df['last'].pct_change()
            df = df.dropna()
        
        return df
    
    async def fetch_vol_surface(self, marquee_id: str, begin: str, end: str) -> pd.DataFrame:
        """Fetch volatility surface from GS Marquee"""
        if not PSC_AVAILABLE:
            # Return mock vol surface
            return self._generate_mock_vol_surface(begin, end)
        
        strikes = [i / 100 for i in range(60, 160, 10)]
        tenors = ['1m', '2m', '3m', '6m', '9m', '1y', '18m', '2y', '3y', '4y', '5y']
        all_data = []

        for strike, tenor in [(s, t) for s in strikes for t in tenors]:
            data = await self.marquee_client.get_data(
                'EDRVOL_PERCENT_PREMIUM_PTS_EXT',
                asset_id=marquee_id,
                start_date=begin,
                end_date=end,
                extra_params={
                    'tenor': tenor,
                    'relativeStrike': strike,
                    'strikeReference': 'forward'
                }
            )
            if data:
                df = pd.DataFrame(data)
                # Keep only the columns we need and rename for consistency
                if 'impliedVolatility' in df.columns:
                    df = df[['date', 'impliedVolatility']].copy()
                    df.rename(columns={'impliedVolatility': 'implied_vol'}, inplace=True)
                else:
                    # Handle case where column might already be named differently
                    df = df[['date', 'implied_vol']].copy() if 'implied_vol' in df.columns else df
                df['strike'] = strike
                df['tenor'] = tenor
                all_data.append(df)

        return pd.concat(all_data, axis=0) if all_data else pd.DataFrame()
    
    async def fetch_ssvi_parameters(self, ticker: str, begin: str, end: str) -> pd.DataFrame:
        """Fetch SSVI parameters from PSC database"""
        if not PSC_AVAILABLE:
            dates = pd.date_range(begin, end, freq='D')
            return pd.DataFrame({
                'date': dates,
                'theta': np.random.uniform(0.02, 0.05, len(dates)),
                'rho': np.random.uniform(-0.7, -0.3, len(dates)),
                'beta': np.random.uniform(0.1, 0.3, len(dates))
            })
        
        db = await get_db_async(self.config.psc_env)
        sql = f"""
            SELECT cob_date as date, theta, rho, beta 
            FROM tbl_market_data_ssvi_parameters_eod 
            WHERE underlying = '{ticker}' 
            AND cob_date BETWEEN '{begin}' AND '{end}'
        """
        data = await db.fetch(sql)
        df = pd.DataFrame([dict(d) for d in data])
        df.sort_values('date', inplace=True)
        return df
    
    def fetch_dividend_yield(self, ticker: str, begin: str, end: str) -> pd.DataFrame:
        """Fetch dividend yield from Bloomberg"""
        if not PSC_AVAILABLE:
            dates = pd.date_range(begin, end, freq='D')
            return pd.DataFrame({
                'date': dates,
                'dividend_yield': np.random.uniform(0.01, 0.02, len(dates))
            })
        
        try:
            df = blp.bdh(
                tickers=ticker,
                flds='BEST_DIV_YLD',
                start_date=begin,
                end_date=end,
                Per='D'
            )
            df.columns = ['dividend_yield']
            df = df.reset_index().rename(columns={'index': 'date'})
            df['date'] = pd.to_datetime(df['date'])
            
            # Sanity check - if dividend yield > 10%, it's likely in basis points
            if (df['dividend_yield'] > 10).any():
                print("Dividend yield appears to be in basis points, converting to decimal")
                df['dividend_yield'] = df['dividend_yield'] / 10000
            elif (df['dividend_yield'] > 0.1).any():
                print("Dividend yield appears to be in percentage, converting to decimal")
                df['dividend_yield'] = df['dividend_yield'] / 100
            
            # Additional sanity check - SPX dividend yield should be between 0.5% and 3%
            df['dividend_yield'] = df['dividend_yield'].clip(lower=0.005, upper=0.03)
            
            return df[['date', 'dividend_yield']]
        except Exception as e:
            print(f"Bloomberg dividend pull failed: {e}")
            return pd.DataFrame()
    
    def fetch_ois_curve(self, currency: str, begin: str, end: str) -> pd.DataFrame:
        """Fetch OIS curve from Bloomberg - returns wide format with rf_* columns"""
        if not PSC_AVAILABLE:
            # Mock data for testing
            dates = pd.date_range(begin, end, freq='D')
            data = {'date': dates}
            tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
            for tenor in tenors:
                data[f'rf_{tenor}'] = np.random.uniform(0.03, 0.05, len(dates))
            return pd.DataFrame(data)
        
        tickers = [
            "SOFRRATE INDEX", "USOSFR1Z Curncy", "USOSFR2Z Curncy", "USOSFR3Z Curncy",
            "USOSFRA Curncy", "USOSFRB Curncy", "USOSFRC Curncy", "USOSFRD Curncy",
            "USOSFRE Curncy", "USOSFRF Curncy", "USOSFRG Curncy", "USOSFRH Curncy",
            "USOSFRI Curncy", "USOSFRJ Curncy", "USOSFRK Curncy",
            "USOSFR1 Curncy", "USOSFR1F Curncy", "USOSFR2 Curncy", "USOSFR3 Curncy",
            "USOSFR4 Curncy", "USOSFR5 Curncy", "USOSFR6 Curncy", "USOSFR7 Curncy",
            "USOSFR8 Curncy", "USOSFR9 Curncy", "USOSFR10 Curncy", "USOSFR12 Curncy",
            "USOSFR15 Curncy", "USOSFR20 Curncy", "USOSFR25 Curncy", "USOSFR30 Curncy",
            "USOSFR40 Curncy", "USOSFR50 Curncy"
        ]
        
        print(f"Fetching OIS data from {begin} to {end}...")
        try:
            df = blp.bdh(tickers=tickers, flds='PX_LAST', start_date=begin, end_date=end)
            if df.empty:
                print("Bloomberg returned empty OIS data")
                return pd.DataFrame()
            
            # Get maturity mappings
            ticker_to_tau = self._get_ticker_year_fractions(tickers)
            if not ticker_to_tau:
                print("No maturities found")
                return pd.DataFrame()
            
            # Rename columns using the rf_τ format (keeping wide format like your code)
            df.columns = [ticker_to_tau.get(col[0], None) for col in df.columns]
            df = df.loc[:, ~pd.isna(df.columns)]
            df.columns.name = None
            
            # Convert rates to decimal
            for col in df.columns:
                df[col] = df[col] / 100.0
            
            df.index.name = "date"
            df.reset_index(inplace=True)
            print(f"OIS curve shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error fetching OIS curve: {e}")
            return pd.DataFrame()
    
    def _get_ticker_year_fractions(self, tickers: list) -> dict:
        """Convert Bloomberg tickers to rf_year format for column names"""
        if not PSC_AVAILABLE:
            return {}
        
        today = pd.Timestamp(dt.datetime.today().date())
        meta = blp.bdp(tickers=tickers, flds='MATURITY')
        meta = meta.dropna()
        
        ticker_to_rf = {}
        for t in meta.index:
            mat_date = pd.to_datetime(meta.loc[t, 'maturity'])
            years = (mat_date - today).days / 365.0
            if years > 0:
                # Return rf_years format to match your working code
                ticker_to_rf[t] = f"rf_{round(years, 6)}"
        
        print(f"Retrieved {len(ticker_to_rf)} ticker → year-fraction mappings")
        return ticker_to_rf
    
    # =================== Database Save Methods ===================
    
    def save_spot_data(self, df: pd.DataFrame):
        """Save spot price data to database"""
        with self._get_db_connection() as conn:
            # Convert dates to strings for SQLite
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df.to_sql('spx_spot_prices', conn, if_exists='replace', index=False)
            print(f"Saved {len(df)} spot price records")
    
    def save_vol_surface(self, df: pd.DataFrame):
        """Save volatility surface to database"""
        with self._get_db_connection() as conn:
            # Prepare dataframe for database
            df = df.copy()
            
            # Rename column to match database schema
            if 'impliedVolatility' in df.columns:
                df.rename(columns={'impliedVolatility': 'implied_vol'}, inplace=True)
            
            # Convert dates to strings for SQLite
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Clear existing data for the date range
            dates = df['date'].unique()
            cursor = conn.cursor()
            for date in dates:
                cursor.execute("DELETE FROM spx_vol_surface WHERE date = ?", (date,))
            
            # Insert new data
            df.to_sql('spx_vol_surface', conn, if_exists='append', index=False)
            print(f"Saved {len(df)} vol surface records")
    
    def save_ssvi_parameters(self, df: pd.DataFrame):
        """Save SSVI parameters to database"""
        with self._get_db_connection() as conn:
            # Convert dates to strings for SQLite
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df.to_sql('spx_ssvi_parameters', conn, if_exists='replace', index=False)
            print(f"Saved {len(df)} SSVI parameter records")
    
    def save_dividend_yield(self, df: pd.DataFrame):
        """Save dividend yield to database"""
        with self._get_db_connection() as conn:
            # Convert dates to strings for SQLite
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df.to_sql('spx_dividend_yield', conn, if_exists='replace', index=False)
            print(f"Saved {len(df)} dividend yield records")
    
    def save_ois_curve(self, df: pd.DataFrame):
            """Save OIS curve to database - handles wide format with rf_* columns"""
            if df.empty:
                print("No OIS curve data to save")
                return
                
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # First ensure the table exists with the correct schema
                cursor.execute("DROP TABLE IF EXISTS ois_curve")
                cursor.execute('''
                    CREATE TABLE ois_curve (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        tenor_years REAL NOT NULL,
                        rate REAL NOT NULL,
                        currency TEXT DEFAULT 'USD',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ois_date ON ois_curve(date)')
                conn.commit()
                
                # Convert wide format to long format
                df = df.copy()
                
                # Get all rf_* columns
                rf_columns = [col for col in df.columns if col.startswith('rf_')]
                if not rf_columns:
                    print("No rf_* columns found in OIS data")
                    return
                
                print(f"Found {len(rf_columns)} tenor columns in OIS data")
                
                # Prepare data for insertion
                records = []
                for _, row in df.iterrows():
                    date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                    for col in rf_columns:
                        if pd.notna(row[col]):
                            # Extract tenor from column name (rf_0.024658 -> 0.024658)
                            tenor_years = float(col.replace('rf_', ''))
                            rate = float(row[col])
                            records.append((date_str, tenor_years, rate, 'USD'))
                
                if records:
                    # Bulk insert
                    cursor.executemany(
                        "INSERT INTO ois_curve (date, tenor_years, rate, currency) VALUES (?, ?, ?, ?)",
                        records
                    )
                    conn.commit()
                    print(f"Saved {len(records)} OIS curve records")
                else:
                    print("No valid OIS records to save")
    def get_risk_free_rate(self, date: str, tenor_years: float) -> float:
        """
        Get risk-free rate for a specific tenor from OIS curve.
        Interpolates between available tenors if needed.
        
        Args:
            date: Valuation date
            tenor_years: Time to maturity in years
            
        Returns:
            Risk-free rate as decimal (0.05 = 5%)
        """
        with self._get_db_connection() as conn:
            # Get OIS curve for the date
            ois_df = pd.read_sql_query(
                """
                SELECT tenor_years, rate 
                FROM ois_curve 
                WHERE date = (
                    SELECT MAX(date) FROM ois_curve WHERE date <= ?
                )
                ORDER BY tenor_years
                """, 
                conn, params=(date,)
            )
            
            if ois_df.empty:
                print(f"Warning: No OIS curve data available, using default rate of 5%")
                return 0.05
            
            # If exact tenor exists, return it
            exact_match = ois_df[ois_df['tenor_years'] == tenor_years]
            if not exact_match.empty:
                return exact_match.iloc[0]['rate']
            
            # Otherwise interpolate
            from scipy import interpolate
            
            # Get unique tenors and rates
            tenors = ois_df['tenor_years'].values
            rates = ois_df['rate'].values
            
            # Handle edge cases
            if tenor_years <= tenors.min():
                return rates[0]
            if tenor_years >= tenors.max():
                return rates[-1]
            
            # Linear interpolation
            interp_func = interpolate.interp1d(tenors, rates, kind='linear', fill_value='extrapolate')
            rate = float(interp_func(tenor_years))
            
            return rate
    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor string to years"""
        if isinstance(tenor, (int, float)):
            return float(tenor)
        
        tenor = tenor.lower()
        if tenor.endswith('d'):
            return int(tenor[:-1]) / 365.25
        elif tenor.endswith('w'):
            return int(tenor[:-1]) / 52
        elif tenor.endswith('m'):
            return int(tenor[:-1]) / 12
        elif tenor.endswith('y'):
            return int(tenor[:-1])
        else:
            raise ValueError(f"Unknown tenor format: {tenor}")
    
    # =================== Main Data Collection Method ===================
    
    async def collect_and_save_all_data(self, ticker: str = "SPX Index", 
                                       begin: str = "2025-09-25", 
                                       end: str = "2025-10-03",
                                       currency: str = "USD"):
        """
        Main method to collect all data and save to database.
        This replaces the CSV export in your original code.
        """
        print(f"Starting data collection for {ticker} from {begin} to {end}")
        
        # Get Marquee ID
        print("Getting Marquee ID...")
        marquee_id = await self.get_marquee_id_from_stock_ticker("SPX")
        print(f"Marquee ID: {marquee_id}")
        
        # Fetch and save spot data
        print("Fetching spot data...")
        spot_df = await self.fetch_spot_data(ticker, begin, end)
        spot_df['date'] = pd.to_datetime(spot_df['date'])
        self.save_spot_data(spot_df)
        
        # Fetch and save SSVI parameters
        print("Fetching SSVI parameters...")
        ssvi_df = await self.fetch_ssvi_parameters(ticker, begin, end)
        ssvi_df['date'] = pd.to_datetime(ssvi_df['date'])
        ssvi_df = ssvi_df.groupby('date')[['theta', 'rho', 'beta']].mean().reset_index()
        self.save_ssvi_parameters(ssvi_df)
        
        # Fetch and save vol surface
        print("Fetching vol surface...")
        vol_df = await self.fetch_vol_surface(marquee_id, begin, end)
        if not vol_df.empty:
            vol_df['date'] = pd.to_datetime(vol_df['date'])
            self.save_vol_surface(vol_df)
        
        # Fetch and save dividend yield
        print("Fetching dividend yield...")
        div_df = self.fetch_dividend_yield(ticker, begin, end)
        if not div_df.empty:
            div_df['date'] = pd.to_datetime(div_df['date'])
            self.save_dividend_yield(div_df)
        
        # Fetch and save OIS curve
        print("Fetching OIS curve...")
        try:
            ois_df = self.fetch_ois_curve(currency, begin, end)
            if not ois_df.empty:
                ois_df['date'] = pd.to_datetime(ois_df['date'])
                self.save_ois_curve(ois_df)
            else:
                print("No OIS curve data retrieved")
        except Exception as e:
            print(f"Warning: OIS curve fetch/save failed: {e}")
            ois_df = pd.DataFrame()  # Continue with empty dataframe
        
        print("All data collection and saving completed!")
        return {
            'spot_records': len(spot_df),
            'ssvi_records': len(ssvi_df),
            'vol_records': len(vol_df) if not vol_df.empty else 0,
            'dividend_records': len(div_df) if not div_df.empty else 0,
            'ois_records': len(ois_df) if not ois_df.empty else 0
        }
    
    # =================== Data Retrieval Methods ===================
    
    def load_data_for_date(self, date: str) -> Dict[str, Any]:
        """Load all market data for a specific date"""
        with self._get_db_connection() as conn:
            # Load spot price
            spot_df = pd.read_sql_query(
                "SELECT * FROM spx_spot_prices WHERE date = ?", 
                conn, params=(date,)
            )
            
            # Load SSVI parameters
            ssvi_df = pd.read_sql_query(
                "SELECT * FROM spx_ssvi_parameters WHERE date = ?", 
                conn, params=(date,)
            )
            
            # Load vol surface
            vol_df = pd.read_sql_query(
                "SELECT * FROM spx_vol_surface WHERE date = ?", 
                conn, params=(date,)
            )
            
            # Load dividend yield (get most recent if not available for exact date)
            div_df = pd.read_sql_query(
                "SELECT * FROM spx_dividend_yield WHERE date <= ? ORDER BY date DESC LIMIT 1", 
                conn, params=(date,)
            )
            
            # Load OIS curve (get most recent if not available for exact date)
            ois_df = pd.read_sql_query(
                """
                SELECT date, tenor_years, rate 
                FROM ois_curve 
                WHERE date = (
                    SELECT MAX(date) FROM ois_curve WHERE date <= ?
                )
                ORDER BY tenor_years
                """, 
                conn, params=(date,)
            )
            
            # If no OIS data, try without date filter to see if any exists
            if ois_df.empty:
                ois_df = pd.read_sql_query(
                    "SELECT date, tenor_years, rate FROM ois_curve ORDER BY date DESC, tenor_years LIMIT 50",
                    conn
                )
                if not ois_df.empty:
                    print(f"Note: Using OIS curve from {ois_df.iloc[0]['date']} (no data for {date})")
                    # Filter to unique date
                    latest_date = ois_df.iloc[0]['date']
                    ois_df = ois_df[ois_df['date'] == latest_date]
            
            return {
                'spot': spot_df.iloc[0]['last'] if not spot_df.empty else None,
                'ssvi': ssvi_df.iloc[0].to_dict() if not ssvi_df.empty else None,
                'vol_surface': vol_df,
                'dividend_yield': div_df.iloc[0]['dividend_yield'] if not div_df.empty else None,
                'ois_curve': ois_df
            }
    
    def _generate_mock_vol_surface(self, begin: str, end: str) -> pd.DataFrame:
        """Generate mock vol surface for testing"""
        dates = pd.date_range(begin, end, freq='D')
        strikes = [i / 100 for i in range(60, 160, 10)]
        tenors = ['1m', '2m', '3m', '6m', '9m', '1y', '2y', '3y', '5y']
        
        data = []
        for date in dates[:10]:  # Just 10 days for testing
            for strike in strikes:
                for tenor in tenors:
                    # Simple vol surface model
                    moneyness = np.log(strike)
                    tenor_years = self._tenor_to_years(tenor)
                    base_vol = 0.15 + 0.1 * abs(moneyness) + 0.05 * np.sqrt(tenor_years)
                    vol = base_vol + np.random.normal(0, 0.01)
                    
                    data.append({
                        'date': date,
                        'strike': strike,
                        'tenor': tenor,
                        'implied_vol': vol  # Changed from 'impliedVolatility'
                    })
        
        return pd.DataFrame(data)
    
    async def update_recent_data(self, days_back: int = 5) -> Dict[str, Any]:
        """
        Update database with recent data only.
        More efficient than full reload - just gets last few days.
        
        Args:
            days_back: Number of days to look back from today
            
        Returns:
            Statistics about what was updated
        """
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates
        begin = start_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
        
        print(f"Updating database with data from {begin} to {end}")
        
        # Run the standard collection for just these dates
        return await self.collect_and_save_all_data(
            ticker="SPX Index",
            begin=begin,
            end=end,
            currency="USD"
        )
    
    def _has_table(self, conn, table_name: str) -> bool:
        """Check if a table exists in the database."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def get_latest_data(self) -> Dict[str, Any]:
        """
        Get the most recent data available in the database.
        Reads from standard schema tables (time_series, vol_surfaces, etc.)
        with fallback to legacy tables for backward compatibility.

        Returns:
            Dictionary with the latest data for each type
        """
        with self._get_db_connection() as conn:
            use_standard = self._has_table(conn, "time_series")

            # --- Spot price ---
            if use_standard:
                spot_df = pd.read_sql_query(
                    "SELECT date, value AS last FROM time_series "
                    "WHERE symbol = 'SPX' AND field = 'spot' ORDER BY date DESC LIMIT 1",
                    conn,
                )
                ret_df = pd.read_sql_query(
                    "SELECT value AS daily_return FROM time_series "
                    "WHERE symbol = 'SPX' AND field = 'daily_return' ORDER BY date DESC LIMIT 1",
                    conn,
                )
            else:
                spot_df = pd.read_sql_query(
                    "SELECT * FROM spx_spot_prices ORDER BY date DESC LIMIT 1", conn
                )
                ret_df = spot_df

            # --- SSVI parameters ---
            if use_standard and self._has_table(conn, "ssvi_parameters"):
                ssvi_df = pd.read_sql_query(
                    "SELECT date, theta, rho, beta FROM ssvi_parameters "
                    "WHERE symbol = 'SPX' ORDER BY date DESC LIMIT 1",
                    conn,
                )
            else:
                ssvi_df = pd.read_sql_query(
                    "SELECT * FROM spx_ssvi_parameters ORDER BY date DESC LIMIT 1", conn
                )

            # --- Vol surface ---
            if use_standard:
                latest_vol_date = pd.read_sql_query(
                    "SELECT MAX(date) as max_date FROM vol_surfaces WHERE symbol = 'SPX'",
                    conn,
                )
                vol_df = pd.DataFrame()
                if not latest_vol_date.empty and latest_vol_date.iloc[0]["max_date"]:
                    vol_df = pd.read_sql_query(
                        "SELECT date, strike, expiry AS tenor, iv AS implied_vol "
                        "FROM vol_surfaces WHERE symbol = 'SPX' AND date = ?",
                        conn,
                        params=(latest_vol_date.iloc[0]["max_date"],),
                    )
            else:
                latest_vol_date = pd.read_sql_query(
                    "SELECT MAX(date) as max_date FROM spx_vol_surface", conn
                )
                vol_df = pd.DataFrame()
                if not latest_vol_date.empty and latest_vol_date.iloc[0]["max_date"]:
                    vol_df = pd.read_sql_query(
                        "SELECT * FROM spx_vol_surface WHERE date = ?",
                        conn,
                        params=(latest_vol_date.iloc[0]["max_date"],),
                    )

            # --- Dividend yield ---
            if use_standard:
                div_df = pd.read_sql_query(
                    "SELECT date, value AS dividend_yield FROM time_series "
                    "WHERE symbol = 'SPX' AND field = 'dividend_yield' ORDER BY date DESC LIMIT 1",
                    conn,
                )
            else:
                div_df = pd.read_sql_query(
                    "SELECT * FROM spx_dividend_yield ORDER BY date DESC LIMIT 1", conn
                )

            # --- OIS curve ---
            if use_standard:
                latest_ois_date = pd.read_sql_query(
                    "SELECT MAX(date) as max_date FROM term_structures WHERE curve_id = 'USD_OIS'",
                    conn,
                )
                ois_df = pd.DataFrame()
                if not latest_ois_date.empty and latest_ois_date.iloc[0]["max_date"]:
                    ois_df = pd.read_sql_query(
                        "SELECT date, tenor, value AS rate FROM term_structures "
                        "WHERE curve_id = 'USD_OIS' AND date = ? ORDER BY tenor",
                        conn,
                        params=(latest_ois_date.iloc[0]["max_date"],),
                    )
            else:
                latest_ois_date = pd.read_sql_query(
                    "SELECT MAX(date) as max_date FROM ois_curve", conn
                )
                ois_df = pd.DataFrame()
                if not latest_ois_date.empty and latest_ois_date.iloc[0]["max_date"]:
                    ois_df = pd.read_sql_query(
                        "SELECT * FROM ois_curve WHERE date = ? ORDER BY tenor_years",
                        conn,
                        params=(latest_ois_date.iloc[0]["max_date"],),
                    )

            # Compile results — same structure as before
            latest_data = {
                'spot': {
                    'date': spot_df.iloc[0]['date'] if not spot_df.empty else None,
                    'price': spot_df.iloc[0]['last'] if not spot_df.empty else None,
                    'return': (ret_df.iloc[0]['daily_return']
                               if not ret_df.empty and 'daily_return' in ret_df.columns
                               else None),
                },
                'ssvi': {
                    'date': ssvi_df.iloc[0]['date'] if not ssvi_df.empty else None,
                    'params': ssvi_df.iloc[0].to_dict() if not ssvi_df.empty else None,
                },
                'vol_surface': {
                    'date': latest_vol_date.iloc[0]['max_date'] if not latest_vol_date.empty else None,
                    'points': len(vol_df),
                    'data': vol_df,
                },
                'dividend': {
                    'date': div_df.iloc[0]['date'] if not div_df.empty else None,
                    'yield': div_df.iloc[0]['dividend_yield'] if not div_df.empty else None,
                },
                'ois_curve': {
                    'date': latest_ois_date.iloc[0]['max_date'] if not latest_ois_date.empty else None,
                    'points': len(ois_df),
                    'data': ois_df,
                },
            }

            # Print summary
            print("\nLatest data available:")
            print(f"  Spot: {latest_data['spot']['date']} - ${latest_data['spot']['price']:,.2f}"
                  if latest_data['spot']['price'] else "  Spot: No data")
            print(f"  SSVI: {latest_data['ssvi']['date']}"
                  if latest_data['ssvi']['date'] else "  SSVI: No data")
            print(f"  Vol Surface: {latest_data['vol_surface']['date']} ({latest_data['vol_surface']['points']} points)"
                  if latest_data['vol_surface']['date'] else "  Vol Surface: No data")
            print(f"  Dividend: {latest_data['dividend']['date']} - {latest_data['dividend']['yield']:.3%}"
                  if latest_data['dividend']['yield'] else "  Dividend: No data")
            print(f"  OIS Curve: {latest_data['ois_curve']['date']} ({latest_data['ois_curve']['points']} points)"
                  if latest_data['ois_curve']['date'] else "  OIS Curve: No data")

            return latest_data
        """Convert tenor string to years"""
        if tenor.endswith('m'):
            return int(tenor[:-1]) / 12
        elif tenor.endswith('y'):
            return int(tenor[:-1])
        else:
            raise ValueError(f"Unknown tenor format: {tenor}")


# =================== Standalone Execution ===================

async def main():
    """Main function to run data collection"""
    loader = SPXDataLoader()
    
    # For the latest 5 days from Oct 3, 2025
    stats = await loader.collect_and_save_all_data(
        ticker="SPX Index",
        begin="2025-09-29",  # 5 days back from Oct 3
        end="2025-10-03",
        currency="USD"
    )
    
    print("\nData collection summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get the latest data for pricing
    latest = loader.get_latest_data()
    print(f"\nLatest spot for pricing: ${latest['spot']['price']:,.2f} on {latest['spot']['date']}")

if __name__ == "__main__":
    asyncio.run(main())