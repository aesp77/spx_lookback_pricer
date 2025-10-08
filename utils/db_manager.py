# utils/db_manager.py
"""
Database management utilities for SPX lookback pricer.
Handles table creation, validation, and maintenance.
"""

import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os

class DatabaseManager:
    """Manage database operations for SPX lookback pricer"""
    
    def __init__(self, db_path: str = 'spx_lookback_data.db'):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database"""
        info = {
            'db_path': self.db_path,
            'db_exists': os.path.exists(self.db_path),
            'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
            'tables': {}
        }
        
        if info['db_exists']:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                # Check if date column exists
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [column[1] for column in cursor.fetchall()]
                
                date_range = (None, None)
                if 'date' in columns and count > 0:
                    try:
                        cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table_name} WHERE date IS NOT NULL")
                        date_range = cursor.fetchone()
                    except Exception as e:
                        # Skip if there's any issue
                        pass
                
                info['tables'][table_name] = {
                    'row_count': count,
                    'min_date': date_range[0] if date_range and date_range[0] else None,
                    'max_date': date_range[1] if date_range and date_range[1] else None
                }
            
            self.disconnect()
        
        return info
    
    def validate_data_integrity(self) -> List[str]:
        """Check data integrity and return list of issues"""
        issues = []
        conn = self.connect()
        
        try:
            # Check for missing spot prices
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date FROM spx_spot_prices 
                ORDER BY date
            """)
            dates = [row[0] for row in cursor.fetchall()]
            
            if dates:
                # Check for gaps in dates
                date_objs = [pd.to_datetime(d) for d in dates]
                for i in range(1, len(date_objs)):
                    diff = (date_objs[i] - date_objs[i-1]).days
                    if diff > 5:  # More than 5 days gap (accounting for weekends)
                        issues.append(f"Large gap in spot prices between {dates[i-1]} and {dates[i]}")
            
            # Check for null values in critical fields
            cursor.execute("SELECT COUNT(*) FROM spx_spot_prices WHERE last IS NULL")
            null_count = cursor.fetchone()[0]
            if null_count > 0:
                issues.append(f"Found {null_count} null spot prices")
            
            # Check SSVI parameters range
            cursor.execute("""
                SELECT COUNT(*) FROM spx_ssvi_parameters 
                WHERE theta < 0 OR theta > 1 
                   OR rho < -1 OR rho > 1 
                   OR beta < 0 OR beta > 10
            """)
            invalid_ssvi = cursor.fetchone()[0]
            if invalid_ssvi > 0:
                issues.append(f"Found {invalid_ssvi} SSVI parameters out of reasonable range")
            
            # Check vol surface values
            cursor.execute("""
                SELECT COUNT(*) FROM spx_vol_surface 
                WHERE implied_vol <= 0 OR implied_vol > 5
            """)
            invalid_vols = cursor.fetchone()[0]
            if invalid_vols > 0:
                issues.append(f"Found {invalid_vols} implied vols out of reasonable range")
            
        finally:
            self.disconnect()
        
        return issues
    
    def get_date_coverage(self) -> pd.DataFrame:
        """Get coverage statistics for each data type"""
        conn = self.connect()
        
        # Get date ranges for each table
        tables = ['spx_spot_prices', 'spx_ssvi_parameters', 'spx_vol_surface', 
                  'spx_dividend_yield', 'ois_curve']
        
        coverage_data = []
        for table in tables:
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    SELECT 
                        MIN(date) as min_date,
                        MAX(date) as max_date,
                        COUNT(DISTINCT date) as unique_dates
                    FROM {table}
                """)
                result = cursor.fetchone()
                if result and result[0]:
                    coverage_data.append({
                        'table': table,
                        'min_date': result[0],
                        'max_date': result[1],
                        'unique_dates': result[2]
                    })
            except sqlite3.OperationalError:
                # Table doesn't exist
                coverage_data.append({
                    'table': table,
                    'min_date': None,
                    'max_date': None,
                    'unique_dates': 0
                })
        
        self.disconnect()
        return pd.DataFrame(coverage_data)
    
    def clean_duplicate_data(self):
        """Remove duplicate entries from tables"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Clean vol surface duplicates
        cursor.execute("""
            DELETE FROM spx_vol_surface 
            WHERE rowid NOT IN (
                SELECT MIN(rowid) 
                FROM spx_vol_surface 
                GROUP BY date, strike, tenor
            )
        """)
        vol_deleted = cursor.rowcount
        
        # Clean OIS curve duplicates
        cursor.execute("""
            DELETE FROM ois_curve 
            WHERE rowid NOT IN (
                SELECT MIN(rowid) 
                FROM ois_curve 
                GROUP BY date, tenor_years
            )
        """)
        ois_deleted = cursor.rowcount
        
        conn.commit()
        self.disconnect()
        
        print(f"Removed {vol_deleted} duplicate vol surface entries")
        print(f"Removed {ois_deleted} duplicate OIS curve entries")
    
    def export_to_parquet(self, output_dir: str = './data_export'):
        """Export all tables to parquet files for backup or analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        conn = self.connect()
        tables = ['spx_spot_prices', 'spx_ssvi_parameters', 'spx_vol_surface', 
                  'spx_dividend_yield', 'ois_curve']
        
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                output_path = os.path.join(output_dir, f"{table}.parquet")
                df.to_parquet(output_path)
                print(f"Exported {table} to {output_path} ({len(df)} rows)")
            except Exception as e:
                print(f"Failed to export {table}: {e}")
        
        self.disconnect()
    
    def create_materialized_view(self):
        """Create a denormalized view for faster querying"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Drop existing view if it exists
        cursor.execute("DROP TABLE IF EXISTS spx_master_view")
        
        # Create materialized view with all data joined
        cursor.execute("""
            CREATE TABLE spx_master_view AS
            SELECT 
                sp.date,
                sp.last as spot_price,
                sp.daily_return,
                ssvi.theta,
                ssvi.rho,
                ssvi.beta,
                div.dividend_yield
            FROM spx_spot_prices sp
            LEFT JOIN spx_ssvi_parameters ssvi ON sp.date = ssvi.date
            LEFT JOIN (
                SELECT date, dividend_yield 
                FROM spx_dividend_yield
            ) div ON sp.date = div.date
            ORDER BY sp.date
        """)
        
        conn.commit()
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM spx_master_view")
        count = cursor.fetchone()[0]
        
        self.disconnect()
        print(f"Created materialized view with {count} rows")
    
    def get_data_summary(self) -> str:
        """Get a formatted summary of the database"""
        info = self.get_database_info()
        coverage = self.get_date_coverage()
        issues = self.validate_data_integrity()
        
        summary = []
        summary.append("=" * 60)
        summary.append("SPX LOOKBACK PRICER DATABASE SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Database: {info['db_path']}")
        summary.append(f"Size: {info['db_size_mb']:.2f} MB")
        summary.append("")
        
        summary.append("TABLE STATISTICS:")
        summary.append("-" * 40)
        for table, stats in info['tables'].items():
            summary.append(f"{table}:")
            summary.append(f"  Records: {stats['row_count']:,}")
            if stats['min_date']:
                summary.append(f"  Date range: {stats['min_date']} to {stats['max_date']}")
        
        summary.append("")
        summary.append("DATA COVERAGE:")
        summary.append("-" * 40)
        for _, row in coverage.iterrows():
            if row['unique_dates'] > 0:
                summary.append(f"{row['table']}: {row['unique_dates']} unique dates")
        
        if issues:
            summary.append("")
            summary.append("DATA INTEGRITY ISSUES:")
            summary.append("-" * 40)
            for issue in issues:
                summary.append(f"⚠ {issue}")
        else:
            summary.append("")
            summary.append("✓ No data integrity issues found")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


# Example usage
if __name__ == "__main__":
    db_manager = DatabaseManager()
    
    # Print database summary
    print(db_manager.get_data_summary())
    
    # Clean duplicates
    db_manager.clean_duplicate_data()
    
    # Export to parquet
    # db_manager.export_to_parquet()
    
    # Create materialized view
    # db_manager.create_materialized_view()