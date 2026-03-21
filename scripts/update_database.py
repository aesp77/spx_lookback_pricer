#!/usr/bin/env python
"""
Update database with recent market data.
Run this daily or as needed to keep data current.
"""

import sys
import asyncio
from datetime import datetime
from pathlib import Path

from spx_lookback_pricer.data.market_data import SPXDataLoader, DatabaseConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "spx_lookback_pricer.db"

async def update_database(days_back: int = 5):
    """
    Update the database with recent data.
    
    Args:
        days_back: Number of days to look back (default 5 to cover weekends)
    """
    print("="*70)
    print("SPX LOOKBACK PRICER - DATABASE UPDATE")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Looking back {days_back} days")
    print("-"*70)
    
    # Use production database
    config = DatabaseConfig(
        db_type='sqlite',
        db_path=str(DB_PATH)
    )
    loader = SPXDataLoader(config)
    
    # Update recent data
    stats = await loader.update_recent_data(days_back=days_back)
    
    print("\n✓ Update completed!")
    print("\nRecords updated:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Show latest data available
    print("\n" + "="*70)
    print("LATEST DATA IN DATABASE")
    print("="*70)
    
    latest = loader.get_latest_data()
    
    # Display summary
    if latest['spot']['price']:
        print(f"\nLatest Spot Price: ${latest['spot']['price']:,.2f} on {latest['spot']['date']}")
    
    if latest['vol_surface']['points'] > 0:
        print(f"Latest Vol Surface: {latest['vol_surface']['points']} points on {latest['vol_surface']['date']}")
    
    return latest

async def check_data_for_pricing():
    """
    Check if we have all necessary data for pricing today.
    """
    print("="*70)
    print("CHECKING DATA READINESS FOR PRICING")
    print("="*70)
    
    config = DatabaseConfig(
        db_type='sqlite',
        db_path=str(DB_PATH)
    )
    loader = SPXDataLoader(config)
    
    # Get latest data
    latest = loader.get_latest_data()
    
    # Check data completeness
    ready = True
    issues = []
    
    if not latest['spot']['price']:
        ready = False
        issues.append("No spot price data")
    
    if not latest['ssvi']['params']:
        ready = False
        issues.append("No SSVI parameters")
    
    if latest['vol_surface']['points'] == 0:
        ready = False
        issues.append("No volatility surface data")
    
    if not latest['dividend']['yield']:
        print("⚠ Warning: No dividend yield data (will use 0%)")
    
    if latest['ois_curve']['points'] == 0:
        print("⚠ Warning: No OIS curve data (will use flat rate)")
    
    if ready:
        print("\n✓ All required data is available for pricing!")
        print(f"\nLatest data dates:")
        print(f"  Spot: {latest['spot']['date']}")
        print(f"  Vol Surface: {latest['vol_surface']['date']}")
        print(f"  SSVI: {latest['ssvi']['date']}")
    else:
        print("\n✗ Missing required data for pricing:")
        for issue in issues:
            print(f"  - {issue}")
    
    return ready, latest

def main():
    """Main function with menu"""
    
    print("\n" + "="*70)
    print("DATABASE UPDATE MENU")
    print("="*70)
    print("1. Quick update (last 5 days)")
    print("2. Weekly update (last 7 days)")
    print("3. Custom update (specify days)")
    print("4. Check data readiness for pricing")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ")
    
    if choice == '1':
        asyncio.run(update_database(days_back=5))
    elif choice == '2':
        asyncio.run(update_database(days_back=7))
    elif choice == '3':
        days = int(input("How many days back? "))
        asyncio.run(update_database(days_back=days))
    elif choice == '4':
        ready, data = asyncio.run(check_data_for_pricing())
        if ready:
            response = input("\nData is ready. Do you want to proceed with pricing? (y/n): ")
            if response.lower() == 'y':
                print("\nYou can now run the pricing module with this data.")
                print(f"Latest spot: ${data['spot']['price']:,.2f}")
    elif choice == '5':
        print("Exiting...")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Auto mode for scheduled tasks
        asyncio.run(update_database(days_back=5))
    else:
        main()