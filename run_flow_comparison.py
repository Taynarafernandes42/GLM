#!/usr/bin/env python3
"""
Run GLM simulations for different flow rates and generate comparison table.
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import subprocess
import os

# Flow rates to test (m³/s)
FLOW_RATES = [0.5, 1.0, 1.5, 2.0]

# Paths
OUTFLOW_FILE = 'outflow_0.csv'
NC_OFF = 'output_off/output.nc'
NC_ON = 'output_on/output.nc'
CSV_OFF = 'output_off/lake.csv'
CSV_ON = 'output_on/lake.csv'
OUTPUT_DIR = 'analysis_results'

def update_outflow(flow_rate):
    """Update outflow_0.csv with new flow rate"""
    # Read existing file to get dates
    df = pd.read_csv(OUTFLOW_FILE)
    df['flow'] = flow_rate
    df.to_csv(OUTFLOW_FILE, index=False)
    print(f"  Updated {OUTFLOW_FILE} to {flow_rate} m³/s")

def run_glm(nml_file):
    """Run GLM simulation"""
    result = subprocess.run(['./glm', '--nml', nml_file], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error running GLM: {result.stderr}")
    return result.returncode == 0

def get_stats():
    """Extract temperature statistics"""
    df_off = pd.read_csv(CSV_OFF)
    df_on = pd.read_csv(CSV_ON)
    
    with Dataset(NC_OFF) as nc:
        temp_off = nc.variables['temp'][:]
    with Dataset(NC_ON) as nc:
        temp_on = nc.variables['temp'][:]
    
    return {
        'Mean Lake Temp (°C)': np.nanmean(temp_on) - np.nanmean(temp_off),
        'Mean Surface Temp (°C)': df_on['Surface Temp'].mean() - df_off['Surface Temp'].mean(),
        'Mean Max Temp (°C)': df_on['Max Temp'].mean() - df_off['Max Temp'].mean(),
        'Mean Min Temp (°C)': df_on['Min Temp'].mean() - df_off['Min Temp'].mean(),
    }

def main():
    print("="*70)
    print("FLOW RATE COMPARISON - Heat Pump Impact")
    print("="*70)
    
    # First, run the OFF case (baseline) - only once
    print("\n[1/5] Running baseline (HP OFF)...")
    if not run_glm('glm4_off.nml'):
        print("ERROR: Failed to run baseline simulation")
        return
    print("  Baseline complete.")
    
    results = {}
    
    for i, flow in enumerate(FLOW_RATES):
        print(f"\n[{i+2}/{len(FLOW_RATES)+1}] Running HP ON with {flow} m³/s...")
        update_outflow(flow)
        if run_glm('glm4.nml'):
            results[flow] = get_stats()
            print(f"  Complete.")
        else:
            print(f"  FAILED!")
    
    # Create results DataFrame
    print("\n" + "="*70)
    print("RESULTS: Temperature Difference (HP ON - HP OFF)")
    print("="*70)
    
    df = pd.DataFrame(results).T
    df.index.name = 'Flow Rate (m³/s)'
    df = df.round(4)
    
    print("\n" + df.to_string())
    
    # Save to CSV
    csv_file = f'{OUTPUT_DIR}/flow_rate_comparison.csv'
    df.to_csv(csv_file)
    print(f"\nSaved: {csv_file}")
    
    return df

if __name__ == '__main__':
    main()
