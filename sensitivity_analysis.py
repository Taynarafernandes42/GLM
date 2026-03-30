#!/usr/bin/env python3
"""
SENSITIVITY ANALYSIS SCRIPT - Heat Pump Flow and dT Parameter Exploration
Runs GLM simulations across a grid of flow rates and temperature changes,
then generates summary tables, heatmaps, and line plots.

Flow values: 0.5, 1, 1.5, 2, 2.5 (m³/day scaling factor applied to outflow)
dT values: 0.5, 1, 1.5, 2, 2.5 (°C temperature change)
Total: 25 simulation scenarios
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from netCDF4 import Dataset
import os
import sys
import subprocess
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_EXECUTABLE = './glm'
NML_FILE = 'glm4.nml'
NML_BACKUP = 'glm4.nml.sensitivity_backup'
OUTFLOW_FILE = 'outflow_0.csv'
OUTFLOW_BACKUP = 'outflow_0.csv.sensitivity_backup'

# Parameter grid
FLOW_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]  # Flow rate multipliers
DT_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]    # Temperature change (°C)

# Output directories
SENSITIVITY_OUTPUT_DIR = 'sensitivity_results'
BASELINE_OUTPUT_DIR = 'output_off'  # Reference (no heat pump)

# Depths of interest for analysis
EXTRACTION_DEPTH = 25.0  # meters above bottom

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
    if os.path.exists(NML_FILE):
        shutil.copy(NML_FILE, NML_BACKUP)
    if os.path.exists(OUTFLOW_FILE):
        shutil.copy(OUTFLOW_FILE, OUTFLOW_BACKUP)


def restore_files():
    """Restore original configuration files"""
    print("  Restoring original files...")
    if os.path.exists(NML_BACKUP):
        shutil.copy(NML_BACKUP, NML_FILE)
    if os.path.exists(OUTFLOW_BACKUP):
        shutil.copy(OUTFLOW_BACKUP, OUTFLOW_FILE)


def modify_nml_dt(dt_value):
    """Modify heat_pump_temp_change in glm4.nml"""
    with open(NML_FILE, 'r') as f:
        content = f.read()
    
    # Replace heat_pump_temp_change value
    import re
    content = re.sub(
        r'heat_pump_temp_change\s*=\s*[\d.]+',
        f'heat_pump_temp_change = {dt_value}',
        content
    )
    
    with open(NML_FILE, 'w') as f:
        f.write(content)


def modify_outflow_flow(flow_value):
    """Modify flow values in outflow_0.csv"""
    df = pd.read_csv(OUTFLOW_BACKUP)  # Always read from backup
    df['flow'] = flow_value
    df.to_csv(OUTFLOW_FILE, index=False)


def set_output_dir(output_dir):
    """Modify output directory in glm4.nml"""
    with open(NML_FILE, 'r') as f:
        content = f.read()
    
    import re
    content = re.sub(
        r"out_dir\s*=\s*'[^']+'",
        f"out_dir = './{output_dir}'",
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
            timeout=600  # 10 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("    GLM timed out!")
        return False
    except Exception as e:
        print(f"    GLM error: {e}")
        return False


def extract_temperature_stats(output_dir):
    """Extract temperature statistics from simulation output"""
    stats = {}
    
    # Load lake.csv
    lake_csv = os.path.join(output_dir, 'lake.csv')
    if os.path.exists(lake_csv):
        df = pd.read_csv(lake_csv)
        if 'Tot Avg Temp' in df.columns:
            stats['mean_temp'] = df['Tot Avg Temp'].mean()
        if 'Surface Temp' in df.columns:
            stats['mean_surface_temp'] = df['Surface Temp'].mean()
            stats['max_surface_temp'] = df['Surface Temp'].max()
            stats['min_surface_temp'] = df['Surface Temp'].min()
    
    # Load output.nc for depth-specific data
    nc_file = os.path.join(output_dir, 'output.nc')
    if os.path.exists(nc_file):
        try:
            with Dataset(nc_file, 'r') as nc:
                temp = nc.variables['temp'][:]
                z = nc.variables['z'][:]
                
                # Overall statistics
                stats['max_temp'] = float(np.nanmax(temp))
                stats['min_temp'] = float(np.nanmin(temp))
                
                # Temperature at extraction depth
                temp_at_depth = []
                for t in range(temp.shape[0]):
                    z_t = z[t, :]
                    temp_t = temp[t, :]
                    valid = ~np.isnan(z_t) & ~np.isnan(temp_t)
                    if np.sum(valid) > 1:
                        try:
                            temp_interp = np.interp(EXTRACTION_DEPTH, z_t[valid], temp_t[valid])
                            temp_at_depth.append(temp_interp)
                        except:
                            pass
                
                if temp_at_depth:
                    stats['mean_temp_at_depth'] = np.mean(temp_at_depth)
                    stats['max_temp_at_depth'] = np.max(temp_at_depth)
                    stats['min_temp_at_depth'] = np.min(temp_at_depth)
                
                # Stratification (dT/dz)
                strat_values = []
                for t in range(temp.shape[0]):
                    z_t = z[t, :]
                    temp_t = temp[t, :]
                    valid = ~np.isnan(z_t) & ~np.isnan(temp_t)
                    if np.sum(valid) > 2:
                        z_range = np.max(z_t[valid]) - np.min(z_t[valid])
                        if z_range > 1:
                            temp_range = np.max(temp_t[valid]) - np.min(temp_t[valid])
                            strat_values.append(temp_range / z_range)
                
                if strat_values:
                    stats['mean_stratification'] = np.mean(strat_values)
                    
        except Exception as e:
            print(f"    Warning: Could not read {nc_file}: {e}")
    
    return stats


def load_baseline_stats():
    """Load or compute baseline (HP OFF) statistics"""
    print("  Loading baseline (HP OFF) statistics...")
    return extract_temperature_stats(BASELINE_OUTPUT_DIR)


# ============================================================================
# MAIN SENSITIVITY ANALYSIS
# ============================================================================
def run_sensitivity_analysis():
    """Run all sensitivity scenarios"""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS - Heat Pump Flow and dT Exploration")
    print("="*70)
    print(f"Flow values: {FLOW_VALUES}")
    print(f"dT values: {DT_VALUES}")
    print(f"Total scenarios: {len(FLOW_VALUES) * len(DT_VALUES)}")
    print()
    
    os.makedirs(SENSITIVITY_OUTPUT_DIR, exist_ok=True)
    
    # Backup original files
    backup_files()
    
    # Load baseline stats
    baseline_stats = load_baseline_stats()
    
    # Results storage
    results = []
    
    try:
        scenario_num = 0
        total_scenarios = len(FLOW_VALUES) * len(DT_VALUES)
        
        for flow in FLOW_VALUES:
            for dt in DT_VALUES:
                scenario_num += 1
                scenario_name = f"flow{flow}_dt{dt}"
                output_dir = os.path.join(SENSITIVITY_OUTPUT_DIR, scenario_name)
                
                print(f"\n[{scenario_num}/{total_scenarios}] Running: Flow={flow}, dT={dt}°C")
                print(f"  Output: {output_dir}")
                
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Modify configuration
                modify_nml_dt(dt)
                modify_outflow_flow(flow)
                set_output_dir(output_dir)
                
                # Run GLM
                print("  Running GLM simulation...")
                success = run_glm()
                
                if success:
                    print("  Simulation complete. Extracting statistics...")
                    stats = extract_temperature_stats(output_dir)
                    stats['flow'] = flow
                    stats['dt'] = dt
                    stats['success'] = True
                    
                    # Compute differences from baseline
                    for key in ['mean_temp', 'mean_surface_temp', 'mean_temp_at_depth', 'mean_stratification']:
                        if key in stats and key in baseline_stats:
                            stats[f'{key}_diff'] = stats[key] - baseline_stats[key]
                    
                    results.append(stats)
                    print(f"    Mean temp at {EXTRACTION_DEPTH}m: {stats.get('mean_temp_at_depth', 'N/A'):.2f}°C")
                else:
                    print("  Simulation FAILED!")
                    results.append({
                        'flow': flow,
                        'dt': dt,
                        'success': False
                    })
    
    finally:
        # Restore original files
        restore_files()
    
    return results, baseline_stats


def create_results_dataframe(results):
    """Convert results to pandas DataFrame"""
    df = pd.DataFrame(results)
    return df


def save_results_table(df, baseline_stats):
    """Save summary table as CSV"""
    output_file = os.path.join(SENSITIVITY_OUTPUT_DIR, 'sensitivity_summary.csv')
    df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    
    # Also save a formatted pivot table
    if 'mean_temp_at_depth_diff' in df.columns:
        pivot = df.pivot(index='flow', columns='dt', values='mean_temp_at_depth_diff')
        pivot_file = os.path.join(SENSITIVITY_OUTPUT_DIR, 'pivot_temp_at_depth.csv')
        pivot.to_csv(pivot_file)
        print(f"  Saved: {pivot_file}")
    
    # Print baseline stats
    baseline_file = os.path.join(SENSITIVITY_OUTPUT_DIR, 'baseline_stats.csv')
    pd.DataFrame([baseline_stats]).to_csv(baseline_file, index=False)
    print(f"  Saved: {baseline_file}")


def create_heatmaps(df):
    """Create heatmap visualizations"""
    print("\n" + "="*70)
    print("GENERATING: Heatmaps")
    print("="*70)
    
    metrics = [
        ('mean_temp_at_depth_diff', f'Mean Temperature Change at {EXTRACTION_DEPTH}m (°C)', 'coolwarm'),
        ('mean_temp_diff', 'Mean Lake Temperature Change (°C)', 'coolwarm'),
        ('mean_stratification_diff', 'Mean Stratification Change (°C/m)', 'RdBu'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (metric, title, cmap) in enumerate(metrics):
        if metric not in df.columns:
            axes[idx].text(0.5, 0.5, 'Data not available', ha='center', va='center')
            axes[idx].set_title(title)
            continue
        
        pivot = df.pivot(index='flow', columns='dt', values=metric)
        
        # Find the maximum absolute value for symmetric colormap
        vmax = np.nanmax(np.abs(pivot.values))
        vmin = -vmax
        
        im = axes[idx].imshow(pivot.values, cmap=cmap, aspect='auto', 
                               vmin=vmin, vmax=vmax, origin='lower')
        
        axes[idx].set_xticks(range(len(DT_VALUES)))
        axes[idx].set_xticklabels([f'{d}' for d in DT_VALUES])
        axes[idx].set_yticks(range(len(FLOW_VALUES)))
        axes[idx].set_yticklabels([f'{f}' for f in FLOW_VALUES])
        axes[idx].set_xlabel('dT (°C)')
        axes[idx].set_ylabel('Flow Rate')
        axes[idx].set_title(title)
        
        # Add value annotations
        for i in range(len(FLOW_VALUES)):
            for j in range(len(DT_VALUES)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    axes[idx].text(j, i, f'{val:.3f}', ha='center', va='center', 
                                   color=color, fontsize=9)
        
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
    
    plt.tight_layout()
    output_file = os.path.join(SENSITIVITY_OUTPUT_DIR, 'heatmaps_sensitivity.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def create_line_plots(df):
    """Create line plots for trend analysis"""
    print("\n" + "="*70)
    print("GENERATING: Line Plots")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Temperature at depth vs Flow (lines for each dT)
    ax = axes[0, 0]
    for dt in DT_VALUES:
        subset = df[df['dt'] == dt].sort_values('flow')
        if 'mean_temp_at_depth_diff' in subset.columns:
            ax.plot(subset['flow'], subset['mean_temp_at_depth_diff'], 
                    marker='o', label=f'dT = {dt}°C')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Flow Rate')
    ax.set_ylabel('Temperature Change (°C)')
    ax.set_title(f'Temperature Change at {EXTRACTION_DEPTH}m vs Flow Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Temperature at depth vs dT (lines for each Flow)
    ax = axes[0, 1]
    for flow in FLOW_VALUES:
        subset = df[df['flow'] == flow].sort_values('dt')
        if 'mean_temp_at_depth_diff' in subset.columns:
            ax.plot(subset['dt'], subset['mean_temp_at_depth_diff'], 
                    marker='s', label=f'Flow = {flow}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('dT (°C)')
    ax.set_ylabel('Temperature Change (°C)')
    ax.set_title(f'Temperature Change at {EXTRACTION_DEPTH}m vs dT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean lake temperature vs Flow
    ax = axes[1, 0]
    for dt in DT_VALUES:
        subset = df[df['dt'] == dt].sort_values('flow')
        if 'mean_temp_diff' in subset.columns:
            ax.plot(subset['flow'], subset['mean_temp_diff'], 
                    marker='o', label=f'dT = {dt}°C')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Flow Rate')
    ax.set_ylabel('Mean Lake Temperature Change (°C)')
    ax.set_title('Mean Lake Temperature Change vs Flow Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cooling efficiency (dT achieved / dT applied) vs Flow
    ax = axes[1, 1]
    if 'mean_temp_at_depth_diff' in df.columns:
        df_copy = df.copy()
        df_copy['efficiency'] = -df_copy['mean_temp_at_depth_diff'] / df_copy['dt']
        for flow in FLOW_VALUES:
            subset = df_copy[df_copy['flow'] == flow].sort_values('dt')
            ax.plot(subset['dt'], subset['efficiency'], 
                    marker='s', label=f'Flow = {flow}')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='100% efficiency')
    ax.set_xlabel('dT (°C)')
    ax.set_ylabel('Cooling Efficiency (realized/applied)')
    ax.set_title('Cooling Efficiency vs dT')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(SENSITIVITY_OUTPUT_DIR, 'line_plots_sensitivity.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def create_response_surface(df):
    """Create 3D response surface plot"""
    print("\n" + "="*70)
    print("GENERATING: 3D Response Surface")
    print("="*70)
    
    if 'mean_temp_at_depth_diff' not in df.columns:
        print("  Skipping: Data not available")
        return
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    flow_grid, dt_grid = np.meshgrid(FLOW_VALUES, DT_VALUES)
    
    # Reshape data for surface plot
    pivot = df.pivot(index='dt', columns='flow', values='mean_temp_at_depth_diff')
    Z = pivot.values
    
    # Plot surface
    surf = ax.plot_surface(flow_grid, dt_grid, Z, cmap='coolwarm', 
                           edgecolor='gray', linewidth=0.5, alpha=0.8)
    
    # Add scatter points for actual data
    ax.scatter(df['flow'], df['dt'], df['mean_temp_at_depth_diff'], 
               c='black', s=50, zorder=5)
    
    ax.set_xlabel('Flow Rate')
    ax.set_ylabel('dT (°C)')
    ax.set_zlabel(f'Temp Change at {EXTRACTION_DEPTH}m (°C)')
    ax.set_title('Response Surface: Temperature Change vs Flow and dT')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Temperature Change (°C)')
    
    output_file = os.path.join(SENSITIVITY_OUTPUT_DIR, 'response_surface.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def create_contour_plot(df):
    """Create contour plot for identifying optimal operating ranges"""
    print("\n" + "="*70)
    print("GENERATING: Contour Plot")
    print("="*70)
    
    if 'mean_temp_at_depth_diff' not in df.columns:
        print("  Skipping: Data not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create fine grid for interpolation
    flow_fine = np.linspace(min(FLOW_VALUES), max(FLOW_VALUES), 50)
    dt_fine = np.linspace(min(DT_VALUES), max(DT_VALUES), 50)
    flow_grid, dt_grid = np.meshgrid(flow_fine, dt_fine)
    
    # Interpolate data
    from scipy.interpolate import griddata
    points = df[['flow', 'dt']].values
    values = df['mean_temp_at_depth_diff'].values
    
    Z = griddata(points, values, (flow_grid, dt_grid), method='cubic')
    
    # Plot contour
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 15)
    contour = ax.contourf(flow_grid, dt_grid, Z, levels=levels, cmap='coolwarm')
    ax.contour(flow_grid, dt_grid, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
    
    # Mark actual data points
    ax.scatter(df['flow'], df['dt'], c='black', s=80, marker='o', edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Flow Rate')
    ax.set_ylabel('dT (°C)')
    ax.set_title(f'Temperature Change at {EXTRACTION_DEPTH}m - Contour Plot')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Temperature Change (°C)')
    
    output_file = os.path.join(SENSITIVITY_OUTPUT_DIR, 'contour_plot.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def print_summary(df, baseline_stats):
    """Print summary to console"""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nBaseline (HP OFF) Statistics:")
    print(f"  Mean Lake Temperature: {baseline_stats.get('mean_temp', 'N/A'):.2f}°C")
    print(f"  Mean Temp at {EXTRACTION_DEPTH}m: {baseline_stats.get('mean_temp_at_depth', 'N/A'):.2f}°C")
    
    if 'mean_temp_at_depth_diff' in df.columns:
        print(f"\nTemperature Change at {EXTRACTION_DEPTH}m (°C) - Matrix:")
        pivot = df.pivot(index='flow', columns='dt', values='mean_temp_at_depth_diff')
        print(pivot.to_string())
        
        # Find optimal scenario
        min_idx = df['mean_temp_at_depth_diff'].idxmin()
        min_row = df.loc[min_idx]
        print(f"\nMaximum cooling: Flow={min_row['flow']}, dT={min_row['dt']}°C "
              f"→ ΔT={min_row['mean_temp_at_depth_diff']:.3f}°C")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS - Heat Pump Parameter Exploration")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all scenarios
    results, baseline_stats = run_sensitivity_analysis()
    
    # Convert to DataFrame
    df = create_results_dataframe(results)
    
    # Filter successful runs
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("\nERROR: No successful simulations!")
        return
    
    print(f"\n{len(df_success)}/{len(df)} simulations completed successfully")
    
    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)
    
    save_results_table(df_success, baseline_stats)
    create_heatmaps(df_success)
    create_line_plots(df_success)
    create_response_surface(df_success)
    create_contour_plot(df_success)
    print_summary(df_success, baseline_stats)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files saved in: {SENSITIVITY_OUTPUT_DIR}/")
    print(f"  - sensitivity_summary.csv      : Full results table")
    print(f"  - pivot_temp_at_depth.csv      : Pivot table (Flow × dT)")
    print(f"  - baseline_stats.csv           : Baseline statistics")
    print(f"  - heatmaps_sensitivity.png     : Heatmap visualizations")
    print(f"  - line_plots_sensitivity.png   : Line plots for trends")
    print(f"  - response_surface.png         : 3D response surface")
    print(f"  - contour_plot.png             : Contour plot")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
