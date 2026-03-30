#!/usr/bin/env python3
"""
OPERATIONAL SENSITIVITY ANALYSIS - Heat Pump Flow Rate and Injection Mode

This script investigates:
1. FLOW RATE SENSITIVITY: How the lake responds to different heat pump flow rates
   (low vs high flow) - to find operational limits where the simple heat pump
   module works well without intrusion/entrainment modeling

2. INJECTION MODE SENSITIVITY: Comparing density-based insertion (finds layer of
   equal density) vs fixed elevation injection (e.g., 25m above bottom)

References for thermal dispersion / jet behavior:
- THERMDIS model: https://thermdis.eawag.ch/en/model
- IGKB thermal jet: https://www.h2o-online.com/igkb_therm/igkb_therm_jet.php

Key physics considerations:
- At LOW flow rates: thermal plume behavior dominates, reinjection finds its
  density level naturally, less mixing, better stratification preservation
- At HIGH flow rates: momentum dominates, forced convection, more mixing,
  potential for artificial destratification, possible over/undershooting of
  density level when using fixed elevation injection

Usage:
    python 6_operational_sensitivity.py [--flow-only] [--injection-only] [--analysis-only]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pandas as pd
from netCDF4 import Dataset
import os
import sys
import subprocess
import shutil
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_EXECUTABLE = './glm'
NML_FILE = 'glm4.nml'
NML_BACKUP = 'glm4.nml.operational_backup'

OUTFLOW_FILE = 'outflow_0.csv'
OUTFLOW_BACKUP = 'outflow_0.csv.operational_backup'

INFLOW_HP_FILE = 'inflow_heatpump.csv'
INFLOW_HP_BACKUP = 'inflow_heatpump.csv.operational_backup'

# Output directories
OUTPUT_DIR = 'operational_sensitivity_results'
BASELINE_OUTPUT_DIR = 'output_off'

# ============================================================================
# FLOW RATE SENSITIVITY CONFIGURATION
# ============================================================================
# Flow rates to test (m³/s)
# Reference: typical heat pump flows range from 0.1 to 2+ m³/s
# Converting to m³/day: 0.1 m³/s = 8640 m³/day, 2 m³/s = 172800 m³/day
# The outflow_0.csv uses m³/s values

FLOW_RATES_M3S = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # m³/s
FLOW_RATE_LABELS = ['Very Low (0.1)', 'Low (0.25)', 'Moderate (0.5)', 
                    'Medium (1.0)', 'High (1.5)', 'Very High (2.0)',
                    'Extreme (3.0)', 'Ultra High (5.0)']

# ============================================================================
# INJECTION MODE SENSITIVITY CONFIGURATION
# ============================================================================
# Test different injection strategies:
# 1. subm_flag = .false. -> density-based insertion (GLM finds density layer)
# 2. subm_flag = .true., subm_elev = 25 -> fixed 25m above bottom
# 3. subm_flag = .true., subm_elev = 5 -> near bottom injection
# 4. subm_flag = .true., subm_elev = 40 -> higher up injection

INJECTION_MODES = {
    'density_based': {
        'name': 'Density-Based Insertion',
        'subm_flag': False,
        'subm_elev': None,
        'description': 'Water finds its density level naturally'
    },
    'fixed_25m': {
        'name': 'Fixed 25m Above Bottom',
        'subm_flag': True,
        'subm_elev': 25.0,
        'description': 'Standard injection at extraction depth'
    },
    'fixed_5m': {
        'name': 'Fixed 5m Above Bottom',
        'subm_flag': True,
        'subm_elev': 5.0,
        'description': 'Deep injection (near hypolimnion)'
    },
    'fixed_35m': {
        'name': 'Fixed 35m Above Bottom',
        'subm_flag': True,
        'subm_elev': 35.0,
        'description': 'Upper injection (near metalimnion/epilimnion)'
    }
}

# Analysis depths
ANALYSIS_DEPTHS = [5, 10, 15, 20, 25, 30, 35]  # meters above bottom
EXTRACTION_DEPTH = 25.0  # standard extraction depth

# Plot settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def backup_files():
    """Backup original configuration files"""
    print("  Backing up original files...")
    for src, dst in [(NML_FILE, NML_BACKUP), (OUTFLOW_FILE, OUTFLOW_BACKUP),
                     (INFLOW_HP_FILE, INFLOW_HP_BACKUP)]:
        if os.path.exists(src):
            shutil.copy(src, dst)


def restore_files():
    """Restore original configuration files"""
    print("  Restoring original files...")
    for src, dst in [(NML_BACKUP, NML_FILE), (OUTFLOW_BACKUP, OUTFLOW_FILE),
                     (INFLOW_HP_BACKUP, INFLOW_HP_FILE)]:
        if os.path.exists(src):
            shutil.copy(src, dst)


def set_output_dir(output_dir):
    """Modify output directory in glm4.nml"""
    with open(NML_FILE, 'r') as f:
        content = f.read()
    
    content = re.sub(
        r"out_dir\s*=\s*'[^']+'",
        f"out_dir = './{output_dir}'",
        content
    )
    
    with open(NML_FILE, 'w') as f:
        f.write(content)


def set_flow_rate(flow_rate_m3s):
    """Set heat pump flow rate in outflow_0.csv"""
    if os.path.exists(OUTFLOW_BACKUP):
        df = pd.read_csv(OUTFLOW_BACKUP)
    else:
        df = pd.read_csv(OUTFLOW_FILE)
    df['flow'] = flow_rate_m3s
    df.to_csv(OUTFLOW_FILE, index=False)
    
    # Also update inflow to match
    if os.path.exists(INFLOW_HP_BACKUP):
        df_in = pd.read_csv(INFLOW_HP_BACKUP)
    else:
        df_in = pd.read_csv(INFLOW_HP_FILE)
    # Note: Inflow is handled by heat pump module, but we need to check format
    df_in.to_csv(INFLOW_HP_FILE, index=False)


def set_injection_mode(subm_flag, subm_elev=None):
    """
    Set injection mode in glm4.nml
    
    Parameters:
    -----------
    subm_flag : bool
        If False: density-based insertion (water finds its density level)
        If True: fixed elevation insertion at subm_elev
    subm_elev : float or None
        Elevation above bottom for fixed injection (only used if subm_flag=True)
    """
    with open(NML_FILE, 'r') as f:
        content = f.read()
    
    # Modify subm_flag for the heat pump inflow (first value in array)
    if subm_flag:
        # Fixed elevation mode
        content = re.sub(
            r"subm_flag\s*=\s*[^!\n]+",
            f"subm_flag      = .true., .false., .false., .false.",
            content
        )
        if subm_elev is not None:
            content = re.sub(
                r"subm_elev\s*=\s*[^!\n]+",
                f"subm_elev      = {subm_elev}, 0.0, 0.0, 0.0",
                content
            )
    else:
        # Density-based mode
        content = re.sub(
            r"subm_flag\s*=\s*[^!\n]+",
            f"subm_flag      = .false., .false., .false., .false.",
            content
        )
    
    with open(NML_FILE, 'w') as f:
        f.write(content)


def enable_heat_pump():
    """Ensure heat pump is enabled"""
    with open(NML_FILE, 'r') as f:
        content = f.read()
    
    content = re.sub(
        r"heat_pump_switch\s*=\s*\d+",
        "heat_pump_switch = 1",
        content
    )
    
    with open(NML_FILE, 'w') as f:
        f.write(content)


def run_glm():
    """Run GLM simulation"""
    try:
        result = subprocess.run(
            [GLM_EXECUTABLE, '--nml', NML_FILE],
            capture_output=True,
            text=True,
            timeout=1200  # 20 minute timeout for long runs
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("    GLM timed out!")
        return False
    except Exception as e:
        print(f"    GLM error: {e}")
        return False


def calculate_water_density(temp, salinity=0.0):
    """
    Calculate water density using UNESCO equation (simplified)
    Used for density-based analysis
    
    Parameters:
    -----------
    temp : float or array
        Temperature in °C
    salinity : float
        Salinity in ppt (default 0 for fresh water)
    
    Returns:
    --------
    density : float or array
        Density in kg/m³
    """
    # Simplified UNESCO equation for fresh water
    rho = (999.842594 + 6.793952e-2 * temp 
           - 9.095290e-3 * temp**2 
           + 1.001685e-4 * temp**3 
           - 1.120083e-6 * temp**4 
           + 6.536336e-9 * temp**5)
    return rho


def extract_comprehensive_stats(output_dir):
    """Extract comprehensive statistics from simulation output"""
    stats = {
        'output_dir': output_dir,
        'success': False
    }
    
    # Load lake.csv
    lake_csv = os.path.join(output_dir, 'lake.csv')
    nc_file = os.path.join(output_dir, 'output.nc')
    
    if not os.path.exists(nc_file):
        return stats
    
    try:
        # Lake CSV statistics
        if os.path.exists(lake_csv):
            df = pd.read_csv(lake_csv)
            if 'Tot Avg Temp' in df.columns:
                stats['mean_lake_temp'] = df['Tot Avg Temp'].mean()
                stats['max_lake_temp'] = df['Tot Avg Temp'].max()
                stats['min_lake_temp'] = df['Tot Avg Temp'].min()
            if 'Surface Temp' in df.columns:
                stats['mean_surface_temp'] = df['Surface Temp'].mean()
                stats['max_surface_temp'] = df['Surface Temp'].max()
        
        # NetCDF detailed analysis
        with Dataset(nc_file, 'r') as nc:
            temp = nc.variables['temp'][:]
            z = nc.variables['z'][:]
            time = nc.variables['time'][:]
            
            # Overall temperature statistics
            stats['overall_max_temp'] = float(np.nanmax(temp))
            stats['overall_min_temp'] = float(np.nanmin(temp))
            stats['overall_mean_temp'] = float(np.nanmean(temp))
            
            # Temperature at multiple depths
            for depth in ANALYSIS_DEPTHS:
                temp_at_depth = []
                for t in range(temp.shape[0]):
                    z_t = z[t, :]
                    temp_t = temp[t, :]
                    valid = ~np.isnan(z_t) & ~np.isnan(temp_t)
                    if np.sum(valid) > 1:
                        try:
                            temp_interp = np.interp(depth, z_t[valid], temp_t[valid])
                            temp_at_depth.append(temp_interp)
                        except:
                            pass
                
                if temp_at_depth:
                    stats[f'mean_temp_{depth}m'] = np.mean(temp_at_depth)
                    stats[f'max_temp_{depth}m'] = np.max(temp_at_depth)
                    stats[f'min_temp_{depth}m'] = np.min(temp_at_depth)
                    stats[f'std_temp_{depth}m'] = np.std(temp_at_depth)
            
            # Stratification analysis
            strat_values = []
            schmidt_stability = []
            
            for t in range(temp.shape[0]):
                z_t = z[t, :]
                temp_t = temp[t, :]
                valid = ~np.isnan(z_t) & ~np.isnan(temp_t)
                
                if np.sum(valid) > 2:
                    z_valid = z_t[valid]
                    temp_valid = temp_t[valid]
                    z_range = np.max(z_valid) - np.min(z_valid)
                    
                    if z_range > 1:
                        # Temperature gradient (stratification strength)
                        temp_range = np.max(temp_valid) - np.min(temp_valid)
                        strat_values.append(temp_range / z_range)
                        
                        # Top-bottom temperature difference
                        top_temp = temp_valid[np.argmax(z_valid)]
                        bot_temp = temp_valid[np.argmin(z_valid)]
                        
                        # Simple Schmidt stability proxy (temperature-based)
                        rho_t = calculate_water_density(temp_valid)
                        if len(rho_t) > 2:
                            mean_z = np.mean(z_valid)
                            # Simplified stability calculation
                            stability = np.sum((z_valid - mean_z) * (rho_t - np.mean(rho_t)))
                            schmidt_stability.append(stability)
            
            if strat_values:
                stats['mean_stratification'] = np.mean(strat_values)
                stats['max_stratification'] = np.max(strat_values)
                stats['min_stratification'] = np.min(strat_values)
            
            if schmidt_stability:
                stats['mean_schmidt_stability'] = np.mean(schmidt_stability)
            
            # Mixing events (days where stratification is very weak)
            if strat_values:
                mixing_threshold = 0.05  # °C/m
                mixing_events = np.sum(np.array(strat_values) < mixing_threshold)
                stats['mixing_events'] = mixing_events
                stats['mixing_fraction'] = mixing_events / len(strat_values)
            
            # Store time series for detailed analysis
            stats['time_series_available'] = True
            stats['n_timesteps'] = len(time)
        
        stats['success'] = True
        
    except Exception as e:
        print(f"    Warning: Error extracting stats from {output_dir}: {e}")
        stats['error'] = str(e)
    
    return stats


def load_baseline_stats():
    """Load baseline (heat pump OFF) statistics"""
    print("  Loading baseline (HP OFF) statistics...")
    stats = extract_comprehensive_stats(BASELINE_OUTPUT_DIR)
    if not stats.get('success', False):
        print("    Warning: Could not load baseline statistics")
    return stats


# ============================================================================
# FLOW RATE SENSITIVITY ANALYSIS
# ============================================================================
def run_flow_rate_sensitivity():
    """Run flow rate sensitivity analysis"""
    print("\n" + "="*70)
    print("FLOW RATE SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Testing flow rates: {FLOW_RATES_M3S} m³/s")
    print(f"Total scenarios: {len(FLOW_RATES_M3S)}")
    
    results = []
    
    for i, flow in enumerate(FLOW_RATES_M3S):
        scenario_name = f"flow_{flow:.2f}m3s"
        output_subdir = os.path.join(OUTPUT_DIR, 'flow_sensitivity', scenario_name)
        
        print(f"\n[{i+1}/{len(FLOW_RATES_M3S)}] Running: Flow = {flow} m³/s")
        print(f"  Output: {output_subdir}")
        
        os.makedirs(output_subdir, exist_ok=True)
        
        # Configure simulation
        set_flow_rate(flow)
        set_output_dir(output_subdir)
        set_injection_mode(subm_flag=True, subm_elev=EXTRACTION_DEPTH)  # Fixed elevation
        enable_heat_pump()
        
        # Run GLM
        print("  Running GLM simulation...")
        success = run_glm()
        
        if success:
            print("  Complete. Extracting statistics...")
            stats = extract_comprehensive_stats(output_subdir)
            stats['flow_rate_m3s'] = flow
            stats['flow_rate_label'] = FLOW_RATE_LABELS[i] if i < len(FLOW_RATE_LABELS) else f"{flow} m³/s"
            stats['scenario'] = scenario_name
            results.append(stats)
        else:
            print("  FAILED!")
            results.append({
                'flow_rate_m3s': flow,
                'flow_rate_label': FLOW_RATE_LABELS[i] if i < len(FLOW_RATE_LABELS) else f"{flow} m³/s",
                'scenario': scenario_name,
                'success': False
            })
    
    return results


# ============================================================================
# INJECTION MODE SENSITIVITY ANALYSIS
# ============================================================================
def run_injection_mode_sensitivity(test_flow=1.0):
    """Run injection mode sensitivity analysis at a fixed flow rate"""
    print("\n" + "="*70)
    print("INJECTION MODE SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Testing flow rate: {test_flow} m³/s")
    print(f"Injection modes: {list(INJECTION_MODES.keys())}")
    
    results = []
    
    for mode_key, mode_config in INJECTION_MODES.items():
        scenario_name = f"injection_{mode_key}"
        output_subdir = os.path.join(OUTPUT_DIR, 'injection_sensitivity', scenario_name)
        
        print(f"\n[{mode_key}] {mode_config['name']}")
        print(f"  Description: {mode_config['description']}")
        print(f"  Output: {output_subdir}")
        
        os.makedirs(output_subdir, exist_ok=True)
        
        # Configure simulation
        set_flow_rate(test_flow)
        set_output_dir(output_subdir)
        set_injection_mode(
            subm_flag=mode_config['subm_flag'],
            subm_elev=mode_config.get('subm_elev')
        )
        enable_heat_pump()
        
        # Run GLM
        print("  Running GLM simulation...")
        success = run_glm()
        
        if success:
            print("  Complete. Extracting statistics...")
            stats = extract_comprehensive_stats(output_subdir)
            stats['injection_mode'] = mode_key
            stats['injection_name'] = mode_config['name']
            stats['injection_description'] = mode_config['description']
            stats['subm_flag'] = mode_config['subm_flag']
            stats['subm_elev'] = mode_config.get('subm_elev')
            stats['test_flow'] = test_flow
            results.append(stats)
        else:
            print("  FAILED!")
            results.append({
                'injection_mode': mode_key,
                'injection_name': mode_config['name'],
                'success': False
            })
    
    return results


# ============================================================================
# COMBINED FLOW + INJECTION ANALYSIS
# ============================================================================
def run_combined_sensitivity():
    """Run combined flow rate × injection mode analysis"""
    print("\n" + "="*70)
    print("COMBINED FLOW RATE × INJECTION MODE ANALYSIS")
    print("="*70)
    
    # Reduced flow rates for combined analysis
    flows_to_test = [0.5, 1.0, 2.0]  # Low, Medium, High
    modes_to_test = ['density_based', 'fixed_25m']  # Most relevant comparison
    
    print(f"Flow rates: {flows_to_test} m³/s")
    print(f"Injection modes: {modes_to_test}")
    print(f"Total scenarios: {len(flows_to_test) * len(modes_to_test)}")
    
    results = []
    scenario_num = 0
    total = len(flows_to_test) * len(modes_to_test)
    
    for flow in flows_to_test:
        for mode_key in modes_to_test:
            scenario_num += 1
            mode_config = INJECTION_MODES[mode_key]
            
            scenario_name = f"flow{flow}_mode_{mode_key}"
            output_subdir = os.path.join(OUTPUT_DIR, 'combined_sensitivity', scenario_name)
            
            print(f"\n[{scenario_num}/{total}] Flow={flow} m³/s, Mode={mode_key}")
            print(f"  Output: {output_subdir}")
            
            os.makedirs(output_subdir, exist_ok=True)
            
            # Configure simulation
            set_flow_rate(flow)
            set_output_dir(output_subdir)
            set_injection_mode(
                subm_flag=mode_config['subm_flag'],
                subm_elev=mode_config.get('subm_elev')
            )
            enable_heat_pump()
            
            # Run GLM
            print("  Running GLM simulation...")
            success = run_glm()
            
            if success:
                print("  Complete. Extracting statistics...")
                stats = extract_comprehensive_stats(output_subdir)
                stats['flow_rate_m3s'] = flow
                stats['injection_mode'] = mode_key
                stats['injection_name'] = mode_config['name']
                results.append(stats)
            else:
                print("  FAILED!")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_flow_sensitivity(results, baseline_stats):
    """Generate flow rate sensitivity plots"""
    print("\n" + "="*70)
    print("GENERATING: Flow Rate Sensitivity Plots")
    print("="*70)
    
    df = pd.DataFrame([r for r in results if r.get('success', False)])
    
    if len(df) == 0:
        print("  No successful results to plot!")
        return
    
    output_subdir = os.path.join(OUTPUT_DIR, 'flow_sensitivity')
    
    # Figure 1: Temperature at different depths vs flow rate
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1a: Mean temperature at extraction depth
    ax = axes[0, 0]
    if 'mean_temp_25m' in df.columns:
        ax.plot(df['flow_rate_m3s'], df['mean_temp_25m'], 'bo-', markersize=8, linewidth=2)
        if 'mean_temp_25m' in baseline_stats:
            ax.axhline(y=baseline_stats['mean_temp_25m'], color='r', linestyle='--', 
                      label='Baseline (HP OFF)', linewidth=1.5)
        ax.set_xlabel('Flow Rate (m³/s)')
        ax.set_ylabel('Mean Temperature (°C)')
        ax.set_title(f'Mean Temperature at {EXTRACTION_DEPTH}m vs Flow Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 1b: Temperature difference from baseline
    ax = axes[0, 1]
    if 'mean_temp_25m' in df.columns and 'mean_temp_25m' in baseline_stats:
        temp_diff = df['mean_temp_25m'] - baseline_stats['mean_temp_25m']
        colors = ['blue' if d < 0 else 'red' for d in temp_diff]
        ax.bar(range(len(df)), temp_diff, color=colors, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"{f:.1f}" for f in df['flow_rate_m3s']], rotation=45)
        ax.set_xlabel('Flow Rate (m³/s)')
        ax.set_ylabel('Temperature Change (°C)')
        ax.set_title(f'Temperature Change at {EXTRACTION_DEPTH}m (HP ON - OFF)')
    
    # 1c: Stratification vs flow rate
    ax = axes[1, 0]
    if 'mean_stratification' in df.columns:
        ax.plot(df['flow_rate_m3s'], df['mean_stratification'], 'go-', markersize=8, linewidth=2)
        if 'mean_stratification' in baseline_stats:
            ax.axhline(y=baseline_stats['mean_stratification'], color='r', linestyle='--',
                      label='Baseline (HP OFF)', linewidth=1.5)
        ax.set_xlabel('Flow Rate (m³/s)')
        ax.set_ylabel('Mean Stratification (°C/m)')
        ax.set_title('Stratification Strength vs Flow Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 1d: Mixing fraction vs flow rate
    ax = axes[1, 1]
    if 'mixing_fraction' in df.columns:
        ax.plot(df['flow_rate_m3s'], df['mixing_fraction'] * 100, 'mo-', markersize=8, linewidth=2)
        if 'mixing_fraction' in baseline_stats:
            ax.axhline(y=baseline_stats['mixing_fraction'] * 100, color='r', linestyle='--',
                      label='Baseline (HP OFF)', linewidth=1.5)
        ax.set_xlabel('Flow Rate (m³/s)')
        ax.set_ylabel('Time with Weak Stratification (%)')
        ax.set_title('Mixing Events vs Flow Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'flow_sensitivity_main.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_subdir}/flow_sensitivity_main.png")
    
    # Figure 2: Temperature profiles at different depths
    fig, ax = plt.subplots(figsize=(12, 8))
    
    depths = [d for d in ANALYSIS_DEPTHS if f'mean_temp_{d}m' in df.columns]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        temps = [row.get(f'mean_temp_{d}m', np.nan) for d in depths]
        ax.plot(temps, depths, 'o-', color=colors[i], 
                label=f"{row['flow_rate_m3s']:.1f} m³/s", linewidth=2, markersize=6)
    
    # Add baseline
    if baseline_stats.get('success', False):
        temps_base = [baseline_stats.get(f'mean_temp_{d}m', np.nan) for d in depths]
        ax.plot(temps_base, depths, 'k--o', label='Baseline (HP OFF)', linewidth=3, markersize=8)
    
    ax.set_xlabel('Mean Temperature (°C)')
    ax.set_ylabel('Depth Above Bottom (m)')
    ax.set_title('Temperature Profiles at Different Flow Rates')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Depth increases downward
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'flow_sensitivity_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_subdir}/flow_sensitivity_profiles.png")
    
    # Figure 3: Flow rate operational limits assessment
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'mean_stratification' in df.columns and 'mean_temp_25m' in df.columns:
        # Normalize metrics for comparison
        strat_change = (df['mean_stratification'] - baseline_stats.get('mean_stratification', df['mean_stratification'].mean())) 
        strat_change = strat_change / baseline_stats.get('mean_stratification', strat_change.abs().max())
        
        temp_change = df['mean_temp_25m'] - baseline_stats.get('mean_temp_25m', df['mean_temp_25m'].mean())
        
        ax2 = ax.twinx()
        
        line1, = ax.plot(df['flow_rate_m3s'], strat_change * 100, 'g^-', 
                        markersize=10, linewidth=2, label='Stratification Change (%)')
        line2, = ax2.plot(df['flow_rate_m3s'], temp_change, 'b^-',
                         markersize=10, linewidth=2, label='Temperature Change (°C)')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-20, color='orange', linestyle=':', alpha=0.7, label='-20% threshold')
        ax.axhline(y=-50, color='red', linestyle=':', alpha=0.7, label='-50% threshold')
        
        ax.set_xlabel('Flow Rate (m³/s)')
        ax.set_ylabel('Stratification Change (% of baseline)', color='g')
        ax2.set_ylabel('Temperature Change at 25m (°C)', color='b')
        
        ax.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Combined legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.set_title('Operational Limits Assessment: Flow Rate Impact')
        
        # Add operational zones annotation
        ax.axvspan(0, 0.5, alpha=0.1, color='green', label='Low Impact Zone')
        ax.axvspan(2, 5, alpha=0.1, color='red', label='High Impact Zone')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'flow_operational_limits.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_subdir}/flow_operational_limits.png")


def plot_injection_sensitivity(results, baseline_stats):
    """Generate injection mode sensitivity plots"""
    print("\n" + "="*70)
    print("GENERATING: Injection Mode Sensitivity Plots")
    print("="*70)
    
    df = pd.DataFrame([r for r in results if r.get('success', False)])
    
    if len(df) == 0:
        print("  No successful results to plot!")
        return
    
    output_subdir = os.path.join(OUTPUT_DIR, 'injection_sensitivity')
    
    # Figure 1: Comparison of injection modes
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    modes = df['injection_mode'].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))
    
    # 1a: Mean temperature at extraction depth by mode
    ax = axes[0, 0]
    if 'mean_temp_25m' in df.columns:
        bars = ax.bar(range(len(df)), df['mean_temp_25m'], color=colors, edgecolor='black')
        if 'mean_temp_25m' in baseline_stats:
            ax.axhline(y=baseline_stats['mean_temp_25m'], color='r', linestyle='--',
                      label='Baseline (HP OFF)', linewidth=2)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([INJECTION_MODES[m]['name'][:15] for m in modes], rotation=30, ha='right')
        ax.set_ylabel('Mean Temperature (°C)')
        ax.set_title(f'Mean Temperature at {EXTRACTION_DEPTH}m by Injection Mode')
        ax.legend()
    
    # 1b: Stratification by mode
    ax = axes[0, 1]
    if 'mean_stratification' in df.columns:
        bars = ax.bar(range(len(df)), df['mean_stratification'], color=colors, edgecolor='black')
        if 'mean_stratification' in baseline_stats:
            ax.axhline(y=baseline_stats['mean_stratification'], color='r', linestyle='--',
                      label='Baseline (HP OFF)', linewidth=2)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([INJECTION_MODES[m]['name'][:15] for m in modes], rotation=30, ha='right')
        ax.set_ylabel('Mean Stratification (°C/m)')
        ax.set_title('Stratification Strength by Injection Mode')
        ax.legend()
    
    # 1c: Temperature profiles by injection mode
    ax = axes[1, 0]
    depths = [d for d in ANALYSIS_DEPTHS if f'mean_temp_{d}m' in df.columns]
    
    for i, (_, row) in enumerate(df.iterrows()):
        temps = [row.get(f'mean_temp_{d}m', np.nan) for d in depths]
        ax.plot(temps, depths, 'o-', color=colors[i], 
                label=INJECTION_MODES[row['injection_mode']]['name'][:20], 
                linewidth=2, markersize=6)
    
    if baseline_stats.get('success', False):
        temps_base = [baseline_stats.get(f'mean_temp_{d}m', np.nan) for d in depths]
        ax.plot(temps_base, depths, 'k--o', label='Baseline (HP OFF)', linewidth=3, markersize=8)
    
    ax.set_xlabel('Mean Temperature (°C)')
    ax.set_ylabel('Depth Above Bottom (m)')
    ax.set_title('Temperature Profiles by Injection Mode')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1d: Mixing events by mode
    ax = axes[1, 1]
    if 'mixing_fraction' in df.columns:
        bars = ax.bar(range(len(df)), df['mixing_fraction'] * 100, color=colors, edgecolor='black')
        if 'mixing_fraction' in baseline_stats:
            ax.axhline(y=baseline_stats['mixing_fraction'] * 100, color='r', linestyle='--',
                      label='Baseline (HP OFF)', linewidth=2)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([INJECTION_MODES[m]['name'][:15] for m in modes], rotation=30, ha='right')
        ax.set_ylabel('Time with Weak Stratification (%)')
        ax.set_title('Mixing Events by Injection Mode')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'injection_sensitivity_main.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_subdir}/injection_sensitivity_main.png")
    
    # Figure 2: Density-based vs Fixed elevation detailed comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    density_row = df[df['injection_mode'] == 'density_based'].iloc[0] if 'density_based' in modes else None
    fixed_row = df[df['injection_mode'] == 'fixed_25m'].iloc[0] if 'fixed_25m' in modes else None
    
    if density_row is not None and fixed_row is not None:
        # Left: Temperature profile comparison
        ax = axes[0]
        
        temps_density = [density_row.get(f'mean_temp_{d}m', np.nan) for d in depths]
        temps_fixed = [fixed_row.get(f'mean_temp_{d}m', np.nan) for d in depths]
        
        ax.plot(temps_density, depths, 'b-o', linewidth=3, markersize=8, label='Density-Based Insertion')
        ax.plot(temps_fixed, depths, 'r-s', linewidth=3, markersize=8, label='Fixed 25m Elevation')
        
        if baseline_stats.get('success', False):
            temps_base = [baseline_stats.get(f'mean_temp_{d}m', np.nan) for d in depths]
            ax.plot(temps_base, depths, 'k--^', linewidth=2, markersize=6, label='Baseline (HP OFF)')
        
        ax.set_xlabel('Mean Temperature (°C)')
        ax.set_ylabel('Depth Above Bottom (m)')
        ax.set_title('Temperature Profile: Density-Based vs Fixed Elevation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right: Difference analysis
        ax = axes[1]
        diff = np.array(temps_density) - np.array(temps_fixed)
        
        colors_bar = ['blue' if d < 0 else 'red' for d in diff]
        ax.barh(depths, diff, color=colors_bar, edgecolor='black', height=2)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel('Temperature Difference (°C)\n(Density-Based - Fixed)')
        ax.set_ylabel('Depth Above Bottom (m)')
        ax.set_title('Temperature Difference: Density-Based vs Fixed')
        
        # Add annotation
        ax.text(0.95, 0.05, 'Blue: Density-based is cooler\nRed: Fixed is cooler',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'density_vs_fixed_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_subdir}/density_vs_fixed_comparison.png")


def create_summary_report(flow_results, injection_results, baseline_stats):
    """Create summary text report and CSV"""
    print("\n" + "="*70)
    print("GENERATING: Summary Report")
    print("="*70)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("OPERATIONAL SENSITIVITY ANALYSIS - SUMMARY REPORT")
    report_lines.append("="*70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Baseline section
    report_lines.append("-"*70)
    report_lines.append("BASELINE (Heat Pump OFF)")
    report_lines.append("-"*70)
    if baseline_stats.get('success', False):
        mean_lake = baseline_stats.get('mean_lake_temp')
        mean_25m = baseline_stats.get('mean_temp_25m')
        mean_strat = baseline_stats.get('mean_stratification')
        mix_frac = baseline_stats.get('mixing_fraction', 0)
        report_lines.append(f"  Mean Lake Temperature: {mean_lake:.2f} °C" if mean_lake else "  Mean Lake Temperature: N/A")
        report_lines.append(f"  Mean Temp at 25m: {mean_25m:.2f} °C" if mean_25m else "  Mean Temp at 25m: N/A")
        report_lines.append(f"  Mean Stratification: {mean_strat:.4f} °C/m" if mean_strat else "  Mean Stratification: N/A")
        report_lines.append(f"  Mixing Fraction: {mix_frac*100:.1f}%")
    else:
        report_lines.append("  Baseline statistics not available")
    
    # Flow rate sensitivity section
    if flow_results:
        report_lines.append("")
        report_lines.append("-"*70)
        report_lines.append("FLOW RATE SENSITIVITY ANALYSIS")
        report_lines.append("-"*70)
        
        df_flow = pd.DataFrame([r for r in flow_results if r.get('success', False)])
        if len(df_flow) > 0:
            report_lines.append(f"\nTested flow rates: {list(df_flow['flow_rate_m3s'])} m³/s")
            report_lines.append("")
            
            for _, row in df_flow.iterrows():
                flow = row['flow_rate_m3s']
                temp_25 = row.get('mean_temp_25m', np.nan)
                strat = row.get('mean_stratification', np.nan)
                
                temp_diff = temp_25 - baseline_stats.get('mean_temp_25m', temp_25)
                strat_diff = strat - baseline_stats.get('mean_stratification', strat)
                
                report_lines.append(f"  Flow = {flow:.2f} m³/s:")
                report_lines.append(f"    Temp at 25m: {temp_25:.2f}°C (Δ = {temp_diff:+.2f}°C)")
                report_lines.append(f"    Stratification: {strat:.4f}°C/m (Δ = {strat_diff:+.4f})")
            
            # Recommendations
            report_lines.append("")
            report_lines.append("  RECOMMENDATIONS:")
            
            # Find flow with minimal stratification impact
            if 'mean_stratification' in df_flow.columns:
                strat_changes = abs(df_flow['mean_stratification'] - baseline_stats.get('mean_stratification', 0))
                best_flow_idx = strat_changes.idxmin()
                best_flow = df_flow.loc[best_flow_idx, 'flow_rate_m3s']
                report_lines.append(f"    - Best stratification preservation: {best_flow:.2f} m³/s")
            
            # Flow range assessment
            report_lines.append(f"    - Low impact range: < 0.5 m³/s")
            report_lines.append(f"    - Moderate impact range: 0.5 - 2.0 m³/s")
            report_lines.append(f"    - High impact range: > 2.0 m³/s")
        
        # Save as CSV
        df_flow.to_csv(os.path.join(OUTPUT_DIR, 'flow_sensitivity_results.csv'), index=False)
        print(f"  Saved: {OUTPUT_DIR}/flow_sensitivity_results.csv")
    
    # Injection mode section
    if injection_results:
        report_lines.append("")
        report_lines.append("-"*70)
        report_lines.append("INJECTION MODE SENSITIVITY ANALYSIS")
        report_lines.append("-"*70)
        
        df_inj = pd.DataFrame([r for r in injection_results if r.get('success', False)])
        if len(df_inj) > 0:
            report_lines.append(f"\nTested modes: {list(df_inj['injection_mode'])}")
            report_lines.append("")
            
            for _, row in df_inj.iterrows():
                mode = row['injection_mode']
                name = row.get('injection_name', mode)
                temp_25 = row.get('mean_temp_25m', np.nan)
                strat = row.get('mean_stratification', np.nan)
                
                report_lines.append(f"  {name}:")
                report_lines.append(f"    Temp at 25m: {temp_25:.2f}°C")
                report_lines.append(f"    Stratification: {strat:.4f}°C/m")
            
            # Compare density-based vs fixed
            report_lines.append("")
            report_lines.append("  KEY FINDING: Density-Based vs Fixed Elevation Injection")
            
            density_row = df_inj[df_inj['injection_mode'] == 'density_based']
            fixed_row = df_inj[df_inj['injection_mode'] == 'fixed_25m']
            
            if len(density_row) > 0 and len(fixed_row) > 0:
                d_strat = density_row['mean_stratification'].values[0]
                f_strat = fixed_row['mean_stratification'].values[0]
                
                if d_strat > f_strat:
                    report_lines.append("    → Density-based insertion better preserves stratification")
                else:
                    report_lines.append("    → Fixed elevation injection may be acceptable")
                
                report_lines.append(f"    → Stratification difference: {abs(d_strat - f_strat):.4f} °C/m")
        
        # Save as CSV
        df_inj.to_csv(os.path.join(OUTPUT_DIR, 'injection_sensitivity_results.csv'), index=False)
        print(f"  Saved: {OUTPUT_DIR}/injection_sensitivity_results.csv")
    
    # Physics discussion
    report_lines.append("")
    report_lines.append("-"*70)
    report_lines.append("PHYSICAL INTERPRETATION")
    report_lines.append("-"*70)
    report_lines.append("""
    Flow Rate Effects:
    - LOW flow rates: Thermal plume behavior dominates. The warmed/cooled water
      has time to find its density level. Less momentum means less forced mixing.
      The simple heat pump module (without entrainment) is most accurate here.
    
    - HIGH flow rates: Momentum dominates over buoyancy. The jet penetrates
      through density layers before equilibrating. Without proper entrainment
      modeling, the module may under-predict mixing effects.
    
    Injection Mode Effects:
    - DENSITY-BASED: Water is inserted at the layer matching its density.
      Most physically accurate for low-momentum flows. Preserves stratification.
    
    - FIXED ELEVATION: Water is forced into a specific depth regardless of
      density. Can cause:
      * Ascending plumes (if injected too deep for its density)
      * Descending plumes (if injected too shallow)
      * Artificial mixing at injection point
    
    Module Limitations:
    The current heat pump module does not include:
    - Jet entrainment (mixing of ambient water into the plume)
    - Plume dynamics (spreading, rise/sink behavior)
    
    These effects become more significant at higher flow rates and when
    injection/extraction are at different depths than the natural density level.
    
    References:
    - THERMDIS model: https://thermdis.eawag.ch/en/model
    - IGKB thermal jet: https://www.h2o-online.com/igkb_therm/igkb_therm_jet.php
    """)
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(os.path.join(OUTPUT_DIR, 'sensitivity_report.txt'), 'w') as f:
        f.write(report_text)
    
    print(f"  Saved: {OUTPUT_DIR}/sensitivity_report.txt")
    print("\n" + report_text)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*70)
    print("OPERATIONAL SENSITIVITY ANALYSIS")
    print("Heat Pump Flow Rate and Injection Mode Investigation")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {WORK_DIR}")
    
    # Parse command line arguments
    run_flow = '--flow-only' in sys.argv or '--all' in sys.argv or len([a for a in sys.argv if a.startswith('--')]) == 0
    run_injection = '--injection-only' in sys.argv or '--all' in sys.argv or len([a for a in sys.argv if a.startswith('--')]) == 0
    analysis_only = '--analysis-only' in sys.argv
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    flow_results = []
    injection_results = []
    
    if not analysis_only:
        # Backup files
        backup_files()
        
        try:
            # Load baseline
            baseline_stats = load_baseline_stats()
            
            # Run flow rate sensitivity
            if run_flow:
                flow_results = run_flow_rate_sensitivity()
            
            # Run injection mode sensitivity
            if run_injection:
                injection_results = run_injection_mode_sensitivity(test_flow=1.0)
        
        finally:
            # Always restore files
            restore_files()
    else:
        # Load existing results
        print("\n  Loading existing results...")
        baseline_stats = load_baseline_stats()
        
        flow_csv = os.path.join(OUTPUT_DIR, 'flow_sensitivity_results.csv')
        if os.path.exists(flow_csv):
            flow_results = pd.read_csv(flow_csv).to_dict('records')
            
        inj_csv = os.path.join(OUTPUT_DIR, 'injection_sensitivity_results.csv')
        if os.path.exists(inj_csv):
            injection_results = pd.read_csv(inj_csv).to_dict('records')
    
    # Load baseline for plotting if not already loaded
    if 'baseline_stats' not in dir():
        baseline_stats = load_baseline_stats()
    
    # Generate visualizations and reports
    if flow_results:
        plot_flow_sensitivity(flow_results, baseline_stats)
    
    if injection_results:
        plot_injection_sensitivity(injection_results, baseline_stats)
    
    create_summary_report(flow_results, injection_results, baseline_stats)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files saved in: {OUTPUT_DIR}/")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
