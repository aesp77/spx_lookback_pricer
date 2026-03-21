"""SPX Lookback Option Pricer — Streamlit Dashboard."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
from pathlib import Path

from spx_lookback_pricer.data.market_data import SPXDataLoader, DatabaseConfig
from spx_lookback_pricer.data.vol_surface import VolatilitySurface
from spx_lookback_pricer.instruments.lookback import (
    PercentageLookbackPut,
    PercentageLookbackCall,
    RatchetingLookbackPut,
    RatchetingLookbackCall,
)

st.set_page_config(
    page_title="SPX Lookback Option Pricer",
    page_icon="",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "spx_lookback_pricer.db"


@st.cache_data
def load_market_data():
    config = DatabaseConfig(
        db_type='sqlite',
        db_path=str(DB_PATH),
    )
    loader = SPXDataLoader(config)
    return loader.get_latest_data(), loader

def calculate_greeks(payoff_spec, spot, protection_level, expiry_years, 
                    vol, rate, div_yield, notional, option_type="Put", n_paths=100000):
    """
    Calculate Greeks using PATHWISE DERIVATIVE method - matches lookback.py
    """
    n_paths_greeks = max(n_paths, 200000)  # Increased for accuracy
    n_steps = int(expiry_years * 252)
    dt = expiry_years / n_steps
    
    # Generate common random numbers ONCE with fixed seed
    np.random.seed(42)
    Z = np.random.standard_normal((n_paths_greeks, n_steps))
    
    def generate_paths_from_spot(initial_spot, vol_val, rate_val, use_fresh_random=False):
        """Generate paths from given initial spot 
        
        Args:
            use_fresh_random: If True, use fresh random numbers instead of common Z
        """
        paths = np.zeros((n_paths_greeks, n_steps + 1))
        paths[:, 0] = initial_spot
        
        drift = (rate_val - div_yield - 0.5 * vol_val**2) * dt
        diffusion = vol_val * np.sqrt(dt)
        
        # Use fresh random numbers for gamma calculation to avoid homogeneity issue
        Z_local = np.random.standard_normal((n_paths_greeks, n_steps)) if use_fresh_random else Z
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(drift + diffusion * Z_local[:, i])
        
        return paths
    
    # Generate base paths
    paths_base = generate_paths_from_spot(spot, vol, rate)
    
    # Calculate base price and payoffs
    final_prices = paths_base[:, -1]
    initial_prices = paths_base[:, 0]
    
    if option_type == "Put":
        max_prices = np.max(paths_base, axis=1)
        
        if payoff_spec == "Dollar Ratcheting":
            # Dollar payoff
            strikes = protection_level * max_prices
            payoffs = np.maximum(strikes - final_prices, 0)
        else:  # Floating Notional %
            strike_prices = protection_level * max_prices
            payoffs = np.maximum(0, (strike_prices - final_prices) / max_prices)
    else:  # Call
        min_prices = np.min(paths_base, axis=1)
        
        if payoff_spec == "Dollar Ratcheting":
            # Dollar payoff
            strikes = protection_level * min_prices
            payoffs = np.maximum(final_prices - strikes, 0)
        else:  # Floating Notional %
            strike_prices = protection_level * min_prices
            payoffs = np.where(min_prices > 0,
                             np.maximum(0, (final_prices - strike_prices) / min_prices),
                             0)
    
    base_price = np.exp(-rate * expiry_years) * np.mean(payoffs)
    
    # PATHWISE DERIVATIVE for Delta
    in_money = payoffs > 0
    
    if payoff_spec == "Dollar Ratcheting":
        # For dollar payoffs, use finite differences with independent random numbers for gamma
        bump_pct = 0.01
        paths_up = generate_paths_from_spot(spot * (1 + bump_pct), vol, rate, use_fresh_random=True)
        paths_down = generate_paths_from_spot(spot * (1 - bump_pct), vol, rate, use_fresh_random=True)
        
        # Recalculate payoffs for bumped paths
        if option_type == "Put":
            max_up = np.max(paths_up, axis=1)
            max_down = np.max(paths_down, axis=1)
            payoffs_up = np.maximum(protection_level * max_up - paths_up[:, -1], 0)
            payoffs_down = np.maximum(protection_level * max_down - paths_down[:, -1], 0)
        else:
            min_up = np.min(paths_up, axis=1)
            min_down = np.min(paths_down, axis=1)
            payoffs_up = np.maximum(paths_up[:, -1] - protection_level * min_up, 0)
            payoffs_down = np.maximum(paths_down[:, -1] - protection_level * min_down, 0)
        
        price_up = np.exp(-rate * expiry_years) * np.mean(payoffs_up)
        price_down = np.exp(-rate * expiry_years) * np.mean(payoffs_down)
        
        delta_dollar = (price_up - price_down) / (2 * spot * bump_pct)
        
        # Convert to percentage elasticity
        if base_price > 0:
            delta_pct = delta_dollar * spot / base_price
        else:
            delta_pct = 0
            
        # Gamma 1% in dollar terms
        dollar_base = base_price * 1.0  # notional = 1 for dollar ratcheting
        dollar_up = price_up * 1.0
        dollar_down = price_down * 1.0
        
        # Dollar delta at each level
        delta_at_up = (dollar_up - dollar_base) / (spot * bump_pct)
        delta_at_down = (dollar_base - dollar_down) / (spot * bump_pct)
        
        # Gamma 1% = change in dollar delta per 1% spot move
        gamma_1pct = (delta_at_up - delta_at_down) / (2 * bump_pct)
        
    else:  # Percentage payoffs - use pathwise derivative
        if option_type == "Put":
            # dPayoff/dS0 = -S_T / (S_0 * S_max) when in the money
            dpayoff_ds0 = np.where(in_money, -final_prices / (spot * max_prices), 0.0)
        else:  # Call
            # dPayoff/dS0 = +S_T / (S_0 * S_min) when in the money
            dpayoff_ds0 = np.where(in_money & (min_prices > 0), 
                                   final_prices / (spot * min_prices), 0.0)
        
        # Raw delta from pathwise derivative
        delta_raw = np.exp(-rate * expiry_years) * np.mean(dpayoff_ds0)
        
        # Convert to elasticity: Delta = (delta_raw * S) / P (no × 100!)
        if base_price > 0:
            delta_pct = (delta_raw * spot / base_price)
        else:
            delta_pct = 0
        
        # Gamma using finite differences with independent random numbers for each scenario
        bump_pct = 0.01
        paths_up = generate_paths_from_spot(spot * (1 + bump_pct), vol, rate, use_fresh_random=True)
        paths_down = generate_paths_from_spot(spot * (1 - bump_pct), vol, rate, use_fresh_random=True)
        
        # Recalculate payoffs
        if option_type == "Put":
            max_up = np.max(paths_up, axis=1)
            max_down = np.max(paths_down, axis=1)
            strike_up = protection_level * max_up
            strike_down = protection_level * max_down
            payoffs_up = np.maximum(0, (strike_up - paths_up[:, -1]) / max_up)
            payoffs_down = np.maximum(0, (strike_down - paths_down[:, -1]) / max_down)
        else:
            min_up = np.min(paths_up, axis=1)
            min_down = np.min(paths_down, axis=1)
            strike_up = protection_level * min_up
            strike_down = protection_level * min_down
            payoffs_up = np.where(min_up > 0,
                                 np.maximum(0, (paths_up[:, -1] - strike_up) / min_up),
                                 0)
            payoffs_down = np.where(min_down > 0,
                                   np.maximum(0, (paths_down[:, -1] - strike_down) / min_down),
                                   0)
        
        price_up = np.exp(-rate * expiry_years) * np.mean(payoffs_up)
        price_down = np.exp(-rate * expiry_years) * np.mean(payoffs_down)
        
        # Gamma 1% - change in dollar delta for a 1% spot move
        dollar_base = base_price * notional
        dollar_up = price_up * notional
        dollar_down = price_down * notional
        
        # Dollar delta at each level
        delta_at_up = (dollar_up - dollar_base) / (spot * bump_pct)
        delta_at_down = (dollar_base - dollar_down) / (spot * bump_pct)
        
        # Gamma 1% = change in dollar delta per 1% spot move
        gamma_1pct = (delta_at_up - delta_at_down) / (2 * bump_pct)
    
    # Vega - use common random numbers
    vol_bump = 0.01
    paths_vega = generate_paths_from_spot(spot, vol + vol_bump, rate)
    
    if payoff_spec == "Dollar Ratcheting":
        if option_type == "Put":
            max_vega = np.max(paths_vega, axis=1)
            payoffs_vega = np.maximum(protection_level * max_vega - paths_vega[:, -1], 0)
        else:
            min_vega = np.min(paths_vega, axis=1)
            payoffs_vega = np.maximum(paths_vega[:, -1] - protection_level * min_vega, 0)
        price_vega = np.exp(-rate * expiry_years) * np.mean(payoffs_vega)
        vega = price_vega - base_price
    else:  # Percentage payoffs
        if option_type == "Put":
            max_vega = np.max(paths_vega, axis=1)
            strike_vega = protection_level * max_vega
            payoffs_vega = np.maximum(0, (strike_vega - paths_vega[:, -1]) / max_vega)
        else:
            min_vega = np.min(paths_vega, axis=1)
            strike_vega = protection_level * min_vega
            payoffs_vega = np.where(min_vega > 0,
                                   np.maximum(0, (paths_vega[:, -1] - strike_vega) / min_vega),
                                   0)
        price_vega = np.exp(-rate * expiry_years) * np.mean(payoffs_vega)
        vega = (price_vega - base_price) * notional
    
    # Theta - approximate
    theta = -base_price * rate / 365.25
    if payoff_spec != "Dollar Ratcheting":
        theta *= notional
    
    # Rho - use common random numbers
    rate_bump = 0.0001
    paths_rho = generate_paths_from_spot(spot, vol, rate + rate_bump)
    
    if payoff_spec == "Dollar Ratcheting":
        if option_type == "Put":
            max_rho = np.max(paths_rho, axis=1)
            payoffs_rho = np.maximum(protection_level * max_rho - paths_rho[:, -1], 0)
        else:
            min_rho = np.min(paths_rho, axis=1)
            payoffs_rho = np.maximum(paths_rho[:, -1] - protection_level * min_rho, 0)
        price_rho = np.exp(-(rate + rate_bump) * expiry_years) * np.mean(payoffs_rho)
        rho = (price_rho - base_price) * 100
    else:  # Percentage payoffs
        if option_type == "Put":
            max_rho = np.max(paths_rho, axis=1)
            strike_rho = protection_level * max_rho
            payoffs_rho = np.maximum(0, (strike_rho - paths_rho[:, -1]) / max_rho)
        else:
            min_rho = np.min(paths_rho, axis=1)
            strike_rho = protection_level * min_rho
            payoffs_rho = np.where(min_rho > 0,
                                  np.maximum(0, (paths_rho[:, -1] - strike_rho) / min_rho),
                                  0)
        price_rho = np.exp(-(rate + rate_bump) * expiry_years) * np.mean(payoffs_rho)
        rho = (price_rho - base_price) * 100 * notional
    
    # Return results
    return {
        'delta': delta_pct,  # Already in percentage form
        'gamma': gamma_1pct,  # Gamma 1% in dollar terms
        'vega': vega,
        'theta': theta,
        'rho': rho,
        'base_price': base_price * (notional if payoff_spec != "Dollar Ratcheting" else 1.0),
        'base_price_pct': base_price * 100 if payoff_spec != "Dollar Ratcheting" else base_price,
        'diagnostics': {
            'paths_used': n_paths_greeks,
            'price_up': price_up,
            'price_down': price_down,
            'price_base': base_price,
            'spot_bump_pct': bump_pct,
            'delta_raw': delta_raw if payoff_spec != "Dollar Ratcheting" else delta_dollar
        }
    }

def calculate_vanilla_greeks(spot, strike, expiry_years, vol, rate, div_yield, option_type="Put"):
    """
    Calculate analytical Greeks for vanilla European option
    """
    T = expiry_years
    
    if T <= 0:
        return {
            'delta': 0,
            'gamma': 0,
            'vega': 0,
            'theta': 0,
            'rho': 0
        }
    
    d1 = (np.log(spot/strike) + (rate - div_yield + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    
    # Calculate price first
    if option_type == "Put":
        price = strike*np.exp(-rate*T)*norm.cdf(-d2) - spot*np.exp(-div_yield*T)*norm.cdf(-d1)
        delta_raw = -np.exp(-div_yield*T) * norm.cdf(-d1)
    else:  # Call
        price = spot*np.exp(-div_yield*T)*norm.cdf(d1) - strike*np.exp(-rate*T)*norm.cdf(d2)
        delta_raw = np.exp(-div_yield*T) * norm.cdf(d1)
    
    # Delta as percentage (standard convention)
    delta_pct = delta_raw * 100
    
    # Gamma (raw)
    gamma_raw = np.exp(-div_yield*T) * norm.pdf(d1) / (spot * vol * np.sqrt(T))
    # Gamma as percentage (percent per percent)
    gamma_pct = gamma_raw * spot * 100
    
    # Vega (per 1% vol move)
    vega = spot * np.exp(-div_yield*T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Theta (daily)
    if option_type == "Put":
        theta = (-spot * np.exp(-div_yield*T) * norm.pdf(d1) * vol / (2*np.sqrt(T))
                + rate * strike * np.exp(-rate*T) * norm.cdf(-d2)
                - div_yield * spot * np.exp(-div_yield*T) * norm.cdf(-d1)) / 365.25
    else:
        theta = (-spot * np.exp(-div_yield*T) * norm.pdf(d1) * vol / (2*np.sqrt(T))
                - rate * strike * np.exp(-rate*T) * norm.cdf(d2)
                + div_yield * spot * np.exp(-div_yield*T) * norm.cdf(d1)) / 365.25
    
    # Rho (per 1% rate move)
    if option_type == "Put":
        rho = -strike * T * np.exp(-rate*T) * norm.cdf(-d2)
    else:
        rho = strike * T * np.exp(-rate*T) * norm.cdf(d2)
    
    return {
        'delta': delta_pct,
        'gamma': gamma_pct,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def main():
    st.title("SPX Lookback Option Pricer")
    st.markdown("### Lookback options with correct Greeks calculation")
    
    # Load market data
    with st.spinner("Loading market data..."):
        latest_data, loader = load_market_data()
    
    if latest_data['spot']['price'] is None:
        st.error("No market data found. Please run update_database.py first.")
        st.stop()
    
    # Sidebar inputs
    st.sidebar.header("Option Parameters")
    
    # Option type
    option_type = st.sidebar.radio(
        "Option Type",
        ["Put", "Call"],
        index=0
    )
    
    # Payoff type - REMOVING Fixed Notional due to mathematical issues
    payoff_spec = st.sidebar.selectbox(
        "Payoff Type",
        [
            "Floating Notional %",
            "Dollar Ratcheting"
        ],
        index=0,
        help="""
        **Floating Notional %**: Payoff = Notional × (α×Max - Final) / Max
        **Dollar Ratcheting**: Payoff = α×Max - Final (in dollars)
        """
    )
    
    # Show payoff formula
    if option_type == "Put":
        if payoff_spec == "Floating Notional %":
            st.sidebar.latex(r"\text{Payoff} = N \cdot \frac{\alpha \cdot S_{max} - S_T}{S_{max}}")
        else:
            st.sidebar.latex(r"\text{Payoff} = \alpha \cdot S_{max} - S_T")
    else:  # Call
        if payoff_spec == "Floating Notional %":
            st.sidebar.latex(r"\text{Payoff} = N \cdot \frac{S_T - \alpha \cdot S_{min}}{S_{min}}")
        else:
            st.sidebar.latex(r"\text{Payoff} = S_T - \alpha \cdot S_{min}")
    
    # Market data
    spot_default = latest_data['spot']['price']
    spot_date = datetime.strptime(latest_data['spot']['date'], '%Y-%m-%d')
    
    st.sidebar.markdown("### Market Data")
    
    spot = st.sidebar.number_input(
        "Spot Price ($)",
        value=float(spot_default),
        min_value=100.0,
        max_value=20000.0,
        step=10.0,
        format="%.2f"
    )
    
    st.sidebar.info(f"""
    **Market Date:** {latest_data['spot']['date']}
    **Using Spot:** ${spot:,.2f}
    """)
    
    # Protection level
    if option_type == "Put":
        protection_default = 95.0
        protection_min = 70.0
        protection_max = 100.0
    else:  # Call
        protection_default = 105.0
        protection_min = 100.0
        protection_max = 130.0
    
    protection_level = st.sidebar.slider(
        "Protection Level (α)",
        min_value=protection_min,
        max_value=protection_max,
        value=protection_default,
        step=1.0
    ) / 100
    
    # Notional
    if payoff_spec != "Dollar Ratcheting":
        notional = st.sidebar.number_input(
            "Notional Amount ($)",
            value=10_000_000.0,
            min_value=1_000.0,
            max_value=1_000_000_000.0,
            step=100_000.0,
            format="%.0f"
        )
        num_units = notional / spot
        
        st.sidebar.success(f"""
        **Position Details:**
        • Notional: ${notional:,.0f}
        • Units: {num_units:,.4f}
        • Value per Unit: ${spot:,.2f}
        """)
    else:
        notional = 1.0
        num_units = 1.0
    
    # Expiry
    default_expiry = spot_date + timedelta(days=180)
    expiry_date = st.sidebar.date_input(
        "Expiry Date",
        value=default_expiry,
        min_value=spot_date + timedelta(days=1),
        max_value=spot_date + timedelta(days=730)
    )
    
    days_to_expiry = (expiry_date - spot_date.date()).days
    time_to_maturity = days_to_expiry / 365.25
    
    st.sidebar.info(f"Days to expiry: {days_to_expiry}")
    
    # Market parameters
    st.sidebar.markdown("### Market Parameters")
    
    risk_free_rate = loader.get_risk_free_rate(latest_data['spot']['date'], time_to_maturity)
    rate_input = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        value=risk_free_rate * 100,
        min_value=0.0,
        max_value=10.0,
        step=0.1
    ) / 100
    
    div_yield = latest_data['dividend']['yield'] if latest_data['dividend']['yield'] else 0.012
    div_input = st.sidebar.number_input(
        "Dividend Yield (%)",
        value=div_yield * 100,
        min_value=0.0,
        max_value=5.0,
        step=0.1
    ) / 100
    
    # Volatility
    st.sidebar.markdown("### Volatility Settings")
    
    # Get implied volatility for the vanilla option
    strike = protection_level * spot
    atm_vol = 0.18  # Default
    implied_vol = None
    
    if latest_data['vol_surface']['data'] is not None:
        vol_surface = VolatilitySurface()
        if vol_surface.load_surface(latest_data['spot']['date']):
            atm_vol = vol_surface.get_vol(spot, time_to_maturity, spot)
            # Get implied vol at the strike
            implied_vol = vol_surface.get_vol(strike, time_to_maturity, spot)
    
    # Lookback volatility input
    vol_input = st.sidebar.number_input(
        "Lookback Volatility (%)",
        value=atm_vol * 100,
        min_value=5.0,
        max_value=50.0,
        step=0.5,
        help="Volatility used for pricing the lookback option"
    ) / 100
    
    # NEW: Display implied volatility information
    st.sidebar.markdown("### Vanilla Option Volatility")
    if implied_vol is not None:
        moneyness = strike / spot
        st.sidebar.info(f"""
        **Market Implied Vol:**
        • Strike: ${strike:,.2f}
        • Moneyness: {moneyness:.2%}
        • Implied Vol: {implied_vol*100:.2f}%
        • ATM Vol: {atm_vol*100:.2f}%
        • Vol Skew: {(implied_vol - atm_vol)*100:.2f}%
        """)
        default_vanilla_vol = implied_vol * 100
    else:
        st.sidebar.warning("Vol surface not available - using ATM vol")
        default_vanilla_vol = atm_vol * 100
    
    # NEW: Editable vanilla volatility
    vanilla_vol = st.sidebar.number_input(
        "Vanilla Volatility (%) - Override",
        value=default_vanilla_vol,
        min_value=5.0,
        max_value=50.0,
        step=0.5,
        help="Override the implied vol for vanilla option pricing"
    ) / 100
    
    # Monte Carlo settings
    st.sidebar.markdown("### Simulation Settings")
    n_paths = st.sidebar.select_slider(
        "Number of Paths",
        options=[10000, 25000, 50000, 100000, 200000],
        value=50000
    )
    
    show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Pricing", "Greeks", "Sensitivity"])
    
    with tab1:
        st.header("Option Pricing")
        
        with st.spinner("Calculating prices..."):
            # Vanilla option price - Using selected vanilla vol
            strike = protection_level * spot
            
            if time_to_maturity <= 0:
                vanilla_price = max(strike - spot, 0) if option_type == "Put" else max(spot - strike, 0)
            else:
                d1 = (np.log(spot/strike) + (rate_input - div_input + 0.5*vanilla_vol**2)*time_to_maturity) / (vanilla_vol*np.sqrt(time_to_maturity))
                d2 = d1 - vanilla_vol*np.sqrt(time_to_maturity)
                
                if option_type == "Put":
                    vanilla_price = strike*np.exp(-rate_input*time_to_maturity)*norm.cdf(-d2) - spot*np.exp(-div_input*time_to_maturity)*norm.cdf(-d1)
                else:
                    vanilla_price = spot*np.exp(-div_input*time_to_maturity)*norm.cdf(d1) - strike*np.exp(-rate_input*time_to_maturity)*norm.cdf(d2)
            
            # Lookback price (Monte Carlo)
            if option_type == "Put":
                if payoff_spec == "Dollar Ratcheting":
                    option = RatchetingLookbackPut(protection_level, time_to_maturity, spot)
                    mc_result = option.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths)
                else:  # Floating Notional %
                    option = PercentageLookbackPut(time_to_maturity, "floating_notional", protection_level)
                    mc_result = option.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths=n_paths, notional=1.0)
            else:  # Call
                if payoff_spec == "Dollar Ratcheting":
                    option = RatchetingLookbackCall(protection_level, time_to_maturity, spot)
                    mc_result = option.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths)
                else:  # Floating Notional %
                    option = PercentageLookbackCall(time_to_maturity, "floating_notional", protection_level)
                    mc_result = option.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths=n_paths, notional=1.0)
            
            lookback_price_unit = mc_result['price']
            lookback_std_error = mc_result.get('std_error', 0)
            
            # Analytical price for standard fixed-strike lookback (if applicable)
            analytical_price = None
            if payoff_spec == "Dollar Ratcheting" and protection_level == 1.0:
                # This is equivalent to a standard fixed-strike lookback
                from spx_lookback_pricer.instruments.lookback import StandardFixedStrikeLookbackPut, StandardFixedStrikeLookbackCall
                if option_type == "Put":
                    standard_lb = StandardFixedStrikeLookbackPut(strike, time_to_maturity)
                    analytical_price = standard_lb.analytical_price(spot, vol_input, rate_input, div_input)
                else:
                    standard_lb = StandardFixedStrikeLookbackCall(strike, time_to_maturity)
                    analytical_price = standard_lb.analytical_price(spot, vol_input, rate_input, div_input)
        
        # Display pricing comparison
        st.subheader("Unit Pricing Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Display format based on payoff type
        if payoff_spec == "Floating Notional %":
            # For percentage payoffs, show as percentage AND dollar value
            vanilla_pct = (vanilla_price / spot) * 100
            lookback_pct = lookback_price_unit * 100
            
            with col1:
                st.metric(
                    f"Vanilla {option_type}",
                    f"{vanilla_pct:.2f}%",
                    f"${vanilla_price:.2f} per unit"
                )
            
            with col2:
                st.metric(
                    f"Lookback {option_type} (MC)",
                    f"{lookback_pct:.2f}%",
                    f"${lookback_pct * spot / 100:.2f} per unit"
                )
            
            with col3:
                st.metric(
                    "Premium Ratio",
                    f"{lookback_pct/vanilla_pct:.2f}x" if vanilla_pct > 0 else "N/A",
                    f"{lookback_pct - vanilla_pct:.2f}% extra"
                )
            
            with col4:
                st.metric(
                    "MC Convergence",
                    f"{n_paths:,} paths",
                    f"95% CI: ±{1.96*lookback_std_error*100:.4f}%"
                )
        else:
            # For dollar payoffs, show in dollars
            with col1:
                st.metric(
                    f"Vanilla {option_type}",
                    f"${vanilla_price:.4f}",
                    f"Vol: {vanilla_vol*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    f"Lookback {option_type} (MC)",
                    f"${lookback_price_unit:.4f}",
                    f"Std Error: ±${lookback_std_error:.6f}"
                )
            
            with col3:
                if analytical_price is not None:
                    st.metric(
                        f"Lookback {option_type} (Analytical)",
                        f"${analytical_price:.4f}",
                        f"Diff from MC: ${analytical_price - lookback_price_unit:.6f}"
                    )
                else:
                    st.metric(
                        "Premium Ratio",
                        f"{lookback_price_unit/vanilla_price:.2f}x" if vanilla_price > 0 else "N/A",
                        f"${lookback_price_unit - vanilla_price:.4f} extra"
                    )
            
            with col4:
                st.metric(
                    "MC Convergence",
                    f"{n_paths:,} paths",
                    f"95% CI: ±${1.96*lookback_std_error:.6f}"
                )
        
        # Position-level pricing
        if payoff_spec == "Floating Notional %":
            st.subheader("Position Pricing")
            
            col1, col2, col3 = st.columns(3)
            
            vanilla_position = vanilla_price * num_units
            lookback_position = (lookback_price_unit * notional)  # lookback_price_unit is already a decimal percentage
            
            with col1:
                st.metric(
                    "Vanilla Position Value",
                    f"${vanilla_position:,.2f}",
                    f"{num_units:,.4f} units × ${vanilla_price:.2f}"
                )
            
            with col2:
                st.metric(
                    "Lookback Position Value", 
                    f"${lookback_position:,.2f}",
                    f"{lookback_price_unit*100:.2f}% × ${notional:,.0f}"
                )
            
            with col3:
                st.metric(
                    "Position Premium",
                    f"{lookback_position/vanilla_position:.2f}x" if vanilla_position > 0 else "N/A",
                    f"${lookback_position - vanilla_position:,.2f}"
                )
        
        # Sensitivity table for different protection levels
        st.subheader("Protection Level Sensitivity")
        
        protection_levels = [0.90, 0.92, 0.95, 0.97, 0.99, 1.00] if option_type == "Put" else [1.00, 1.02, 1.05, 1.07, 1.10, 1.15]
        
        sensitivity_data = []
        for prot_level in protection_levels:
            # Vanilla at this protection
            strike_temp = prot_level * spot
            if time_to_maturity > 0:
                d1_temp = (np.log(spot/strike_temp) + (rate_input - div_input + 0.5*vanilla_vol**2)*time_to_maturity) / (vanilla_vol*np.sqrt(time_to_maturity))
                d2_temp = d1_temp - vanilla_vol*np.sqrt(time_to_maturity)
                
                if option_type == "Put":
                    vanilla_temp = strike_temp*np.exp(-rate_input*time_to_maturity)*norm.cdf(-d2_temp) - spot*np.exp(-div_input*time_to_maturity)*norm.cdf(-d1_temp)
                else:
                    vanilla_temp = spot*np.exp(-div_input*time_to_maturity)*norm.cdf(d1_temp) - strike_temp*np.exp(-rate_input*time_to_maturity)*norm.cdf(d2_temp)
            else:
                vanilla_temp = max(strike_temp - spot, 0) if option_type == "Put" else max(spot - strike_temp, 0)
            
            # Quick MC for lookback
            if option_type == "Put":
                if payoff_spec == "Dollar Ratcheting":
                    opt_temp = RatchetingLookbackPut(prot_level, time_to_maturity, spot)
                    res_temp = opt_temp.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths=10000)
                else:
                    opt_temp = PercentageLookbackPut(time_to_maturity, "floating_notional", prot_level)
                    res_temp = opt_temp.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths=10000, notional=1.0)
            else:
                if payoff_spec == "Dollar Ratcheting":
                    opt_temp = RatchetingLookbackCall(prot_level, time_to_maturity, spot)
                    res_temp = opt_temp.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths=10000)
                else:
                    opt_temp = PercentageLookbackCall(time_to_maturity, "floating_notional", prot_level)
                    res_temp = opt_temp.monte_carlo_price(spot, vol_input, rate_input, div_input, n_paths=10000, notional=1.0)
            
            lookback_temp = res_temp['price']
            
            # Format based on payoff type
            if payoff_spec == "Floating Notional %":
                vanilla_pct = (vanilla_temp / spot) * 100
                lookback_pct = lookback_temp * 100
                
                sensitivity_data.append({
                    'Protection': f"{prot_level*100:.0f}%",
                    'Strike': f"${strike_temp:.2f}",
                    'Vanilla': f"{vanilla_pct:.4f}%",
                    'Lookback': f"{lookback_pct:.4f}%",
                    'Premium': f"{lookback_pct/vanilla_pct:.2f}x" if vanilla_pct > 0 else "N/A",
                    'Dollar Value': f"${lookback_pct * spot / 100:.2f}"
                })
            else:
                sensitivity_data.append({
                    'Protection': f"{prot_level*100:.0f}%",
                    'Strike': f"${strike_temp:.2f}",
                    'Vanilla': f"${vanilla_temp:.4f}",
                    'Lookback': f"${lookback_temp:.4f}",
                    'Premium': f"{lookback_temp/vanilla_temp:.2f}x" if vanilla_temp > 0 else "N/A"
                })
        
        df_sensitivity = pd.DataFrame(sensitivity_data)
        st.dataframe(df_sensitivity, use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("Greeks Analysis")
        
        with st.spinner("Calculating Greeks..."):
            # Calculate lookback Greeks using improved method
            greeks = calculate_greeks(
                payoff_spec, spot, protection_level, time_to_maturity,
                vol_input, rate_input, div_input, notional, option_type, n_paths
            )
            
            # Calculate vanilla Greeks analytically
            vanilla_greeks = calculate_vanilla_greeks(
                spot, strike, time_to_maturity, vanilla_vol, 
                rate_input, div_input, option_type
            )
        
        # Display comparison
        st.subheader("Greeks Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Lookback {option_type}")
            col1_1, col1_2, col1_3, col1_4 = st.columns(4)
            
            with col1_1:
                st.metric("Delta", f"{greeks['delta']:.2f}%")
                st.caption("% per 1% spot")
            
            with col1_2:
                gamma_display = greeks['gamma'] / 1000 if abs(greeks['gamma']) > 10000 else greeks['gamma']
                gamma_unit = "k" if abs(greeks['gamma']) > 10000 else ""
                st.metric("Gamma 1%", f"${gamma_display:.1f}{gamma_unit}")
                st.caption("$ P&L per 1% move")
            
            with col1_3:
                vega_display = greeks['vega'] / 1000 if abs(greeks['vega']) > 10000 else greeks['vega']
                vega_unit = "k" if abs(greeks['vega']) > 10000 else ""
                st.metric("Vega", f"${vega_display:.1f}{vega_unit}")
                st.caption("$ per vol pt")
            
            with col1_4:
                theta_display = greeks['theta'] / 1000 if abs(greeks['theta']) > 10000 else greeks['theta']
                theta_unit = "k" if abs(greeks['theta']) > 10000 else ""
                st.metric("Theta", f"${theta_display:.1f}{theta_unit}")
                st.caption("Daily decay")
        
        with col2:
            st.markdown(f"### Vanilla {option_type}")
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)
            
            with col2_1:
                st.metric("Delta", f"{vanilla_greeks['delta']:.2f}%")
                st.caption("% per 1% spot")
            
            with col2_2:
                st.metric("Gamma", f"{vanilla_greeks['gamma']:.2f}%")
                st.caption("Delta chg per 1%")
            
            with col2_3:
                vanilla_vega_position = vanilla_greeks['vega'] * (num_units if payoff_spec != "Dollar Ratcheting" else 1.0)
                vega_display = vanilla_vega_position / 1000 if abs(vanilla_vega_position) > 10000 else vanilla_vega_position
                vega_unit = "k" if abs(vanilla_vega_position) > 10000 else ""
                st.metric("Vega", f"${vega_display:.1f}{vega_unit}")
                st.caption("$ per vol pt")
            
            with col2_4:
                vanilla_theta_position = vanilla_greeks['theta'] * (num_units if payoff_spec != "Dollar Ratcheting" else 1.0)
                theta_display = vanilla_theta_position / 1000 if abs(vanilla_theta_position) > 10000 else vanilla_theta_position
                theta_unit = "k" if abs(vanilla_theta_position) > 10000 else ""
                st.metric("Theta", f"${theta_display:.1f}{theta_unit}")
                st.caption("Daily decay")
        
        # Greeks ratios
        st.subheader("Greeks Ratios (Lookback / Vanilla)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_ratio = greeks['delta'] / vanilla_greeks['delta'] if vanilla_greeks['delta'] != 0 else 0
            st.metric("Delta Ratio", f"{delta_ratio:.2f}x")
        
        with col2:
            gamma_ratio = greeks['gamma'] / vanilla_greeks['gamma'] if vanilla_greeks['gamma'] != 0 else 0
            st.metric("Gamma Ratio", f"{gamma_ratio:.2f}x")
        
        with col3:
            vanilla_vega_pos = vanilla_greeks['vega'] * (num_units if payoff_spec != "Dollar Ratcheting" else 1.0)
            vega_ratio = greeks['vega'] / vanilla_vega_pos if vanilla_vega_pos != 0 else 0
            st.metric("Vega Ratio", f"{vega_ratio:.2f}x")
        
        with col4:
            vanilla_theta_pos = vanilla_greeks['theta'] * (num_units if payoff_spec != "Dollar Ratcheting" else 1.0)
            theta_ratio = greeks['theta'] / vanilla_theta_pos if vanilla_theta_pos != 0 else 0
            st.metric("Theta Ratio", f"{theta_ratio:.2f}x")
        
        # Sanity checks
        if option_type == "Put":
            if greeks['delta'] > 0:
                st.error("Warning: Lookback Put showing positive delta")
            if vanilla_greeks['delta'] > 0:
                st.error("Warning: Vanilla Put showing positive delta")
        else:  # Call
            if greeks['delta'] < 0:
                st.error("Warning: Lookback Call showing negative delta")
            if vanilla_greeks['delta'] < 0:
                st.error("Warning: Vanilla Call showing negative delta")
        
        # Delta Profile Chart
        st.subheader("Delta Profile by Protection Level")
        
        with st.spinner("Calculating delta sensitivity..."):
            # Protection levels from 85% to 100% for puts, 100% to 115% for calls
            if option_type == "Put":
                protection_range = np.linspace(0.85, 1.00, 16)
            else:
                protection_range = np.linspace(1.00, 1.15, 16)
            
            delta_values = []
            protection_labels = []
            
            progress_bar = st.progress(0)
            for i, prot in enumerate(protection_range):
                # Calculate greeks for this protection level
                greeks_temp = calculate_greeks(
                    payoff_spec, spot, prot, time_to_maturity,
                    vol_input, rate_input, div_input, notional, option_type, 
                    n_paths=50000  # Use fewer paths for speed
                )
                delta_values.append(greeks_temp['delta'])
                protection_labels.append(f"{prot*100:.0f}%")
                progress_bar.progress((i + 1) / len(protection_range))
            
            progress_bar.empty()
            
            # Create chart
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=protection_range * 100,
                y=delta_values,
                mode='lines+markers',
                name='Delta',
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=8)
            ))
            
            # Add current protection level marker
            fig_delta.add_vline(
                x=protection_level*100, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Current: {protection_level*100:.0f}%",
                annotation_position="top"
            )
            
            # Add zero line
            fig_delta.add_hline(y=0, line_dash="dot", line_color="gray")
            
            fig_delta.update_layout(
                xaxis_title="Protection Level (%)",
                yaxis_title="Delta (%)",
                height=400,
                hovermode='x unified',
                showlegend=False
            )
            
            st.plotly_chart(fig_delta, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Delta", f"{min(delta_values):.2f}%", f"at {protection_range[np.argmin(delta_values)]*100:.0f}%")
            with col2:
                st.metric("Max Delta", f"{max(delta_values):.2f}%", f"at {protection_range[np.argmax(delta_values)]*100:.0f}%")
            with col3:
                st.metric("Current Delta", f"{greeks['delta']:.2f}%", f"at {protection_level*100:.0f}%")
        
        st.info("""
        **Greeks Notes:**
        - Delta shown as percentage elasticity (% price change per 1% spot move)
        - Gamma 1% shows dollar P&L change for a 1% spot move (convexity risk)
        - Greeks calculated using pathwise derivative method for lookback options
        - Vanilla Greeks calculated analytically using Black-Scholes
        - Position Greeks scaled to actual notional amount
        """)
        
        # Detailed diagnostics
        if show_debug:
            with st.expander("Debug Information"):
                st.write("### Lookback Greeks Diagnostics")
                if 'diagnostics' in greeks:
                    diag = greeks['diagnostics']
                    st.write(f"- Paths used: {diag.get('paths_used', 'N/A'):,}")
                    st.write(f"- Base price: {diag.get('price_base', 0):.6f}")
                    st.write(f"- Price up: {diag.get('price_up', 0):.6f}")
                    st.write(f"- Price down: {diag.get('price_down', 0):.6f}")
                    st.write(f"- Spot bump: {diag.get('spot_bump_pct', 0)*100:.2f}%")
                
                st.write("### Position Information")
                st.write(f"- Notional: ${notional:,.0f}")
                if payoff_spec != "Dollar Ratcheting":
                    st.write(f"- Number of units: {num_units:.4f}")
                    st.write(f"- Spot price: ${spot:.2f}")
                
                st.write("### Raw Greeks Values")
                st.write("**Lookback:**")
                st.write(f"- Delta: {greeks['delta']:.6f}%")
                st.write(f"- Gamma: {greeks['gamma']:.6f}%")
                st.write(f"- Vega: ${greeks['vega']:.2f}")
                st.write(f"- Theta: ${greeks['theta']:.2f}")
                
                st.write("**Vanilla:**")
                st.write(f"- Delta: {vanilla_greeks['delta']:.6f}%")
                st.write(f"- Gamma: {vanilla_greeks['gamma']:.6f}%")
                st.write(f"- Vega: ${vanilla_greeks['vega']:.2f} per unit")
                st.write(f"- Theta: ${vanilla_greeks['theta']:.2f} per unit")
    
    with tab3:
        st.header("Sensitivity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Volatility Sensitivity")
            
            vol_range = np.linspace(0.10, 0.40, 15)
            prices_vol = []
            
            progress_bar = st.progress(0)
            for i, v in enumerate(vol_range):
                if payoff_spec == "Dollar Ratcheting":
                    if option_type == "Put":
                        opt = RatchetingLookbackPut(protection_level, time_to_maturity, spot)
                    else:
                        opt = RatchetingLookbackCall(protection_level, time_to_maturity, spot)
                    res = opt.monte_carlo_price(spot, v, rate_input, div_input, 10000)
                else:
                    if option_type == "Put":
                        opt = PercentageLookbackPut(time_to_maturity, "floating_notional", protection_level)
                    else:
                        opt = PercentageLookbackCall(time_to_maturity, "floating_notional", protection_level)
                    # Use UNIT notional for sensitivity analysis
                    res = opt.monte_carlo_price(spot, v, rate_input, div_input, n_paths=10000, notional=1.0)
                    res['price'] *= notional  # Scale to position
                
                prices_vol.append(res['price'])
                progress_bar.progress((i + 1) / len(vol_range))
            
            progress_bar.empty()
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_range * 100,
                y=prices_vol,
                mode='lines+markers',
                name='Price'
            ))
            fig_vol.add_vline(x=vol_input*100, line_dash="dash", line_color="red")
            fig_vol.update_layout(
                xaxis_title="Volatility (%)",
                yaxis_title="Option Price ($)",
                height=400
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            st.subheader("Spot Sensitivity")
            
            spot_range = np.linspace(0.85 * spot, 1.15 * spot, 15)
            prices_spot = []
            
            progress_bar = st.progress(0)
            for i, s in enumerate(spot_range):
                if payoff_spec == "Dollar Ratcheting":
                    if option_type == "Put":
                        opt = RatchetingLookbackPut(protection_level, time_to_maturity, s)
                    else:
                        opt = RatchetingLookbackCall(protection_level, time_to_maturity, s)
                    res = opt.monte_carlo_price(s, vol_input, rate_input, div_input, 10000)
                else:
                    if option_type == "Put":
                        opt = PercentageLookbackPut(time_to_maturity, "floating_notional", protection_level)
                    else:
                        opt = PercentageLookbackCall(time_to_maturity, "floating_notional", protection_level)
                    # Use UNIT notional for consistent sensitivity
                    res = opt.monte_carlo_price(s, vol_input, rate_input, div_input, n_paths=10000, notional=1.0)
                    res['price'] *= notional  # Scale to position
                
                prices_spot.append(res['price'])
                progress_bar.progress((i + 1) / len(spot_range))
            
            progress_bar.empty()
            
            fig_spot = go.Figure()
            fig_spot.add_trace(go.Scatter(
                x=spot_range,
                y=prices_spot,
                mode='lines+markers',
                name='Price'
            ))
            fig_spot.add_vline(x=spot, line_dash="dash", line_color="red")
            fig_spot.update_layout(
                xaxis_title="Spot Price ($)",
                yaxis_title="Option Price ($)",
                height=400
            )
            st.plotly_chart(fig_spot, use_container_width=True)

if __name__ == "__main__":
    main()