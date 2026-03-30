#!/usr/bin/env python3
"""
WATER BALANCE ANALYSIS - Heat Pump Impact on Lake Hydrology
Generates all water balance related plots:
  - Lake level comparison
  - Lake volume comparison
  - Evaporation comparison
  - Mass conservation verification
  - Inflow/Outflow analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_OFF = 'output_off/lake.csv'
CSV_ON = 'output_on/lake.csv'
OUTPUT_DIR = 'analysis_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Heat pump parameters
HP_FLOW_RATE = 25920  # m³/day (0.3 m³/s)

def read_hp_depths_from_nml(nml_file='glm4.nml'):
    """Read heat pump extraction and injection elevations from glm4.nml"""
    extraction_elev = 25.0
    injection_elev = 25.0
    
    try:
        with open(nml_file, 'r') as f:
            for line in f:
                line_clean = line.split('!')[0].strip()
                if 'subm_elev_outflow' in line_clean:
                    val_str = line_clean.split('=')[1].strip()
                    extraction_elev = float(val_str.split(',')[0].strip())
                elif 'subm_elev' in line_clean and 'outflow' not in line_clean:
                    val_str = line_clean.split('=')[1].strip()
                    injection_elev = float(val_str.split(',')[0].strip())
        print(f"  Read from {nml_file}: extraction={extraction_elev}m, injection={injection_elev}m")
    except Exception as e:
        print(f"  Warning: Could not read {nml_file}: {e}")
    
    return extraction_elev, injection_elev

HP_EXTRACTION_ELEV, HP_INJECTION_ELEV = read_hp_depths_from_nml()

# Plot settings
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*70)
print("WATER BALANCE ANALYSIS - Heat Pump Impact")
print("="*70)

print("\n[1] Loading data...")

def load_lake_csv(csv_file):
    """Load lake.csv output file"""
    print(f"  Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    df["time"] = df["time"].str.replace(" 24:00:00", " 00:00:00")
    df["datetime"] = pd.to_datetime(df["time"]) + pd.Timedelta(days=1)
    return df

df_off = load_lake_csv(CSV_OFF)
df_on = load_lake_csv(CSV_ON)

# Align data by datetime (in case simulations have different periods)
df_off = df_off.set_index('datetime')
df_on = df_on.set_index('datetime')

# Find common date range
common_start = max(df_off.index.min(), df_on.index.min())
common_end = min(df_off.index.max(), df_on.index.max())

df_off = df_off.loc[common_start:common_end].reset_index()
df_on = df_on.loc[common_start:common_end].reset_index()

print(f"  OFF period: {df_off['datetime'].min()} to {df_off['datetime'].max()} ({len(df_off)} days)")
print(f"  ON period:  {df_on['datetime'].min()} to {df_on['datetime'].max()} ({len(df_on)} days)")
print(f"  Common period: {common_start} to {common_end} ({len(df_off)} days)")

# ============================================================================
# FIGURE 1: Lake Level and Volume
# ============================================================================
print("\n[2] Creating Figure: Lake Level and Volume...")

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

# Lake Level
ax = axes1[0, 0]
ax.plot(df_off['datetime'], df_off['Lake Level'], 'r-', lw=2.5, label='HP OFF', alpha=0.85)
ax.plot(df_on['datetime'], df_on['Lake Level'], 'b-', lw=2.5, label='HP ON', alpha=0.85)
ax.set_ylabel('Lake Level (m)')
ax.set_title('(a) Lake Water Level')
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Level Difference
ax = axes1[0, 1]
level_diff = df_on['Lake Level'].values - df_off['Lake Level'].values
ax.plot(df_off['datetime'], level_diff * 1000, 'k-', lw=2.0)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.set_ylabel('Level Difference (mm)')
ax.set_title('(b) Water Level Difference (ON - OFF)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
stats_text = f'Mean: {np.mean(level_diff)*1000:.3f} mm\nStd: {np.std(level_diff)*1000:.3f} mm\nMax: {np.max(np.abs(level_diff))*1000:.3f} mm'
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Volume
ax = axes1[1, 0]
ax.plot(df_off['datetime'], df_off['Volume']/1e6, 'r-', lw=2.5, label='HP OFF', alpha=0.85)
ax.plot(df_on['datetime'], df_on['Volume']/1e6, 'b-', lw=2.5, label='HP ON', alpha=0.85)
ax.set_ylabel('Lake Volume (× 10⁶ m³)')
ax.set_xlabel('Year')
ax.set_title('(c) Lake Volume')
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Volume Difference
ax = axes1[1, 1]
vol_diff = df_on['Volume'].values - df_off['Volume'].values
ax.plot(df_off['datetime'], vol_diff, 'k-', lw=2.0)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.set_ylabel('Volume Difference (m³)')
ax.set_xlabel('Year')
ax.set_title('(d) Volume Difference (ON - OFF)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Add statistics (matching format of level difference panel)
stats_text = f'Mean: {np.mean(vol_diff):.1f} m³\nStd: {np.std(vol_diff):.1f} m³\nMax: {np.max(np.abs(vol_diff)):.1f} m³'
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig1.savefig(f'{OUTPUT_DIR}/2_fig_waterbalance_level_volume.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/2_fig_waterbalance_level_volume.png")

# ============================================================================
# FIGURE 2: Evaporation Analysis
# ============================================================================
print("\n[3] Creating Figure: Evaporation Analysis...")

fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Daily Evaporation
ax = axes2[0]
ax.plot(df_off['datetime'], -df_off['Evaporation'], 'r-', lw=2.5, alpha=0.85, label='HP OFF')
ax.plot(df_on['datetime'], -df_on['Evaporation'], 'b-', lw=2.5, alpha=0.85, label='HP ON')
ax.set_ylabel('Evaporation (m³/day)')
ax.set_title('(a) Daily Evaporation')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Evaporation Difference
ax = axes2[1]
evap_diff = (-df_on['Evaporation'].values) - (-df_off['Evaporation'].values)
ax.plot(df_off['datetime'], evap_diff, 'k-', lw=2.0, alpha=0.8)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.axhline(np.mean(evap_diff), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(evap_diff):+.2f} m³/day')
ax.fill_between(df_off['datetime'], 0, evap_diff, where=(evap_diff > 0), alpha=0.3, color='red')
ax.fill_between(df_off['datetime'], 0, evap_diff, where=(evap_diff < 0), alpha=0.3, color='blue')
ax.set_ylabel('Δ Evaporation (m³/day)')
ax.set_title('(b) Evaporation Difference (ON - OFF)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Cumulative Evaporation Difference
ax = axes2[2]
cumsum_evap_diff = np.cumsum(evap_diff)
ax.plot(df_off['datetime'], cumsum_evap_diff/1e3, 'k-', lw=2.5)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.set_ylabel('Cumulative Δ Evap (× 10³ m³)')
ax.set_xlabel('Year')
ax.set_title('(c) Cumulative Evaporation Difference')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.text(0.88, 0.97, f'Total: {cumsum_evap_diff[-1]/1e3:+.1f} × 10³ m³', transform=ax.transAxes,
        fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/2_fig_waterbalance_evaporation.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/2_fig_waterbalance_evaporation.png")

# ============================================================================
# FIGURE 3: Mass Conservation Verification
# ============================================================================
print("\n[4] Creating Figure: Mass Conservation Verification...")

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

# Lake Level Difference Time Series
ax = axes3[0, 0]
ax.plot(df_off['datetime'], level_diff * 1000, 'b-', lw=2.5)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.axhline(np.mean(level_diff)*1000, color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(level_diff)*1000:.2f} mm')
ax.set_ylabel('Level Difference (mm)')
ax.set_title('(a) Lake Level Difference Over Time')
ax.legend(fontsize=12, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Volume Difference Time Series
ax = axes3[0, 1]
ax.plot(df_off['datetime'], vol_diff, 'b-', lw=2.5)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.axhline(np.mean(vol_diff), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(vol_diff):.0f} m³')
ax.set_ylabel('Volume Difference (m³)')
ax.set_title('(b) Lake Volume Difference Over Time')
ax.legend(fontsize=12, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Level Difference Histogram
ax = axes3[1, 0]
ax.hist(level_diff * 1000, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(0, color='gray', ls='--', lw=2.5)
ax.axvline(np.mean(level_diff)*1000, color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(level_diff)*1000:.2f} mm')
ax.set_xlabel('Level Difference (mm)')
ax.set_ylabel('Frequency')
ax.set_title('(c) Distribution of Level Differences')
ax.legend(fontsize=12, framealpha=0.9)

# Volume Difference Histogram
ax = axes3[1, 1]
ax.hist(vol_diff, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(0, color='gray', ls='--', lw=2.5)
ax.axvline(np.mean(vol_diff), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(vol_diff):.0f} m³')
ax.set_xlabel('Volume Difference (m³)')
ax.set_ylabel('Frequency')
ax.set_title('(d) Distribution of Volume Differences')
ax.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/3_fig_waterbalance_mass_conservation.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/3_fig_waterbalance_mass_conservation.png")

# ============================================================================
# FIGURE 4: Water Balance Components (if available)
# ============================================================================
print("\n[5] Creating Figure: Inflows and Outflows...")

fig4, axes4 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Check available columns for water balance
wb_cols = ['Tot Inflow Vol', 'Tot Outflow Vol', 'Overflow Vol']
available_cols = [col for col in wb_cols if col in df_off.columns]

if len(available_cols) >= 2:
    # Inflows - heat pump injection is handled as submerged inflow, 
    # but GLM tracks it separately, so no correction needed here
    if 'Tot Inflow Vol' in df_off.columns:
        ax = axes4[0]
        ax.plot(df_off['datetime'], df_off['Tot Inflow Vol'], 'r-', lw=2.5, alpha=0.85, label='HP OFF')
        ax.plot(df_on['datetime'], df_on['Tot Inflow Vol'], 'b-', lw=2.5, alpha=0.85, label='HP ON')
        ax.set_ylabel('Inflow (m³/day)')
        ax.set_title('(a) Total Inflow Volume (external inflows only)')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Outflows - subtract HP extraction from ON simulation for fair comparison
    # because HP extraction is counted in Tot Outflow Vol but water returns to lake
    if 'Tot Outflow Vol' in df_off.columns:
        ax = axes4[1]
        # HP extraction is part of Tot Outflow Vol in ON simulation - subtract it
        outflow_on_corrected = df_on['Tot Outflow Vol'] - HP_FLOW_RATE
        ax.plot(df_off['datetime'], df_off['Tot Outflow Vol'], 'r-', lw=2.5, alpha=0.85, label='HP OFF')
        ax.plot(df_on['datetime'], outflow_on_corrected, 'b-', lw=2.5, alpha=0.85, label='HP ON (excl. HP extraction)')
        ax.set_ylabel('Outflow (m³/day)')
        ax.set_title('(b) Total Outflow Volume (excluding heat pump extraction)')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Add annotation about HP flow
        ax.text(0.98, 0.95, f'HP extraction excluded: {HP_FLOW_RATE:,} m³/day', 
                transform=ax.transAxes, fontsize=12, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Net balance
    ax = axes4[2]
    if 'Tot Inflow Vol' in df_off.columns and 'Tot Outflow Vol' in df_off.columns:
        # Note: Evaporation is already NEGATIVE in GLM output (water loss)
        # So: Net = Inflow - Outflow + Evaporation (adding negative = subtracting)
        net_off = df_off['Tot Inflow Vol'] - df_off['Tot Outflow Vol'] + df_off['Evaporation']
        # For HP ON: subtract HP flow from outflow only (it returns via separate pathway)
        net_on = df_on['Tot Inflow Vol'] - (df_on['Tot Outflow Vol'] - HP_FLOW_RATE) + df_on['Evaporation']
        ax.plot(df_off['datetime'], net_off, 'r-', lw=2.5, alpha=0.85, label='HP OFF')
        ax.plot(df_on['datetime'], net_on, 'b-', lw=2.5, alpha=0.85, label='HP ON')
        ax.axhline(0, color='gray', ls='--', lw=2.0)
        ax.set_ylabel('Net Balance (m³/day)')
        ax.set_title('(c) Net Water Balance (Inflow - Outflow + Evap, where Evap<0)')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Panel (d): Difference in net water balance (ON - OFF) - shows evaporation impact
        ax = axes4[3]
        net_diff = net_on - net_off
        ax.plot(df_off['datetime'], net_diff, 'purple', lw=2.5, alpha=0.85)
        ax.axhline(0, color='gray', ls='--', lw=2.0)
        mean_diff = net_diff.mean()
        ax.axhline(mean_diff, color='red', ls='-', lw=3.0, label=f'Mean: {mean_diff:.1f} m³/day')
        ax.set_ylabel('Δ Net Balance (m³/day)')
        ax.set_title('(d) Net Water Balance Difference (ON - OFF) — primarily from evaporation changes')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Add annotation explaining this is mostly evaporation difference
        evap_diff_mean = (df_on['Evaporation'] - df_off['Evaporation']).mean()
        ax.text(0.02, 0.95, f'Mean evaporation difference: {evap_diff_mean:.1f} m³/day\n(explains most of this signal)', 
                transform=ax.transAxes, fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
else:
    # If water balance columns not available, show storage change
    ax = axes4[0]
    storage_change_off = np.diff(df_off['Volume'].values, prepend=df_off['Volume'].values[0])
    storage_change_on = np.diff(df_on['Volume'].values, prepend=df_on['Volume'].values[0])
    ax.plot(df_off['datetime'], storage_change_off, 'b-', lw=2.5, alpha=0.85, label='HP OFF')
    ax.plot(df_on['datetime'], storage_change_on, 'r-', lw=2.5, alpha=0.85, label='HP ON')
    ax.set_ylabel('Storage Change (m³/day)')
    ax.set_title('(a) Daily Storage Change')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    ax = axes4[1]
    ax.plot(df_off['datetime'], storage_change_on - storage_change_off, 'k-', lw=2.5)
    ax.axhline(0, color='gray', ls='--', lw=2.0)
    ax.set_ylabel('Δ Storage Change (m³/day)')
    ax.set_title('(b) Storage Change Difference (ON - OFF)')
    
    ax = axes4[2]
    ax.text(0.5, 0.5, 'Additional water balance columns not available', 
            transform=ax.transAxes, ha='center', va='center', fontsize=12)

axes4[-1].set_xlabel('Year')
axes4[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes4[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig4.savefig(f'{OUTPUT_DIR}/3_fig_waterbalance_inflow_outflow.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/3_fig_waterbalance_inflow_outflow.png")

# ============================================================================
# FIGURE 5: Volume Difference vs Evaporation Difference
# ============================================================================
print("\n[6] Creating Figure: Volume vs Evaporation Difference...")

fig5, axes5 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel (a): Volume Difference (ON - OFF)
ax = axes5[0]
vol_diff = df_on['Volume'].values - df_off['Volume'].values
ax.plot(df_off['datetime'], vol_diff, 'purple', lw=0.8, label='Volume Difference (ON - OFF)')
ax.axhline(0, color='gray', ls='--', lw=1)
ax.axhline(np.mean(vol_diff), color='red', ls='-', lw=1.5, alpha=0.7, label=f'Mean: {np.mean(vol_diff):.0f} m³')
ax.set_ylabel('Volume Difference (m³)')
ax.set_title('(a) Lake Volume Difference (ON - OFF)')
ax.legend(loc='upper right', fontsize=9)

# Panel (b): Cumulative Evaporation Difference
ax = axes5[1]
# Evaporation is negative in GLM (water loss), so evap_diff = ON - OFF
# If HP ON has less evaporation (less negative), evap_diff > 0
evap_diff = df_on['Evaporation'].values - df_off['Evaporation'].values  # This is in "negative space"
# Cumulative: positive means HP ON lost LESS water to evaporation
cumsum_evap_diff = np.cumsum(evap_diff)
ax.plot(df_off['datetime'], cumsum_evap_diff, 'g-', lw=0.8, label='Cumulative Evap. Difference')
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_ylabel('Cumulative Δ Evaporation (m³)')
ax.set_title('(b) Cumulative Evaporation Difference (ON - OFF)')
ax.legend(loc='upper right', fontsize=9)
ax.text(0.02, 0.95, f'Total: {cumsum_evap_diff[-1]:.0f} m³\n(positive = HP ON evaporated less)', 
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Panel (c): Cumulative Overflow Difference
ax = axes5[2]
if 'Overflow Vol' in df_off.columns:
    overflow_diff = df_on['Overflow Vol'].values - df_off['Overflow Vol'].values
    cumsum_overflow_diff = np.cumsum(overflow_diff)
    ax.plot(df_off['datetime'], cumsum_overflow_diff, 'orange', lw=0.8, label='Cumulative Overflow Difference')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_ylabel('Cumulative Δ Overflow (m³)')
    ax.set_title('(c) Cumulative Overflow Difference (ON - OFF)')
    ax.legend(loc='upper right', fontsize=9)
    ax.text(0.02, 0.95, f'Total: {cumsum_overflow_diff[-1]:.0f} m³\n(positive = HP ON had more overflow)', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.5))
else:
    ax.text(0.5, 0.5, 'Overflow Vol not available in output', 
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    cumsum_overflow_diff = np.zeros_like(vol_diff)

# Panel (d): Overlay comparison
ax = axes5[3]
ax.plot(df_off['datetime'], vol_diff, 'purple', lw=0.8, alpha=0.8, label='Volume Difference')
ax.plot(df_off['datetime'], cumsum_evap_diff, 'g-', lw=0.8, alpha=0.8, label='Cumulative Evap. Difference')

# Add cumulative overflow difference (negative because overflow removes water)
if 'Overflow Vol' in df_off.columns:
    overflow_diff = df_on['Overflow Vol'].values - df_off['Overflow Vol'].values
    cumsum_overflow_diff = -np.cumsum(overflow_diff)  # Negative: more overflow = less volume retained
    ax.plot(df_off['datetime'], cumsum_overflow_diff, 'orange', lw=0.8, alpha=0.8, label='Cumulative Overflow Loss (ON-OFF)')
    
    # Also show the combined effect: evap saved - overflow lost
    combined_effect = cumsum_evap_diff + cumsum_overflow_diff
    ax.plot(df_off['datetime'], combined_effect, 'r--', lw=1.2, alpha=0.8, label='Net Effect (Evap - Overflow)')

ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_ylabel('Difference (m³)')
ax.set_xlabel('Year')
ax.set_title('(d) Comparison: Volume Difference vs Cumulative Evaporation & Overflow Differences')
ax.legend(loc='upper right', fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Calculate correlations
correlation_evap = np.corrcoef(vol_diff, cumsum_evap_diff)[0, 1]
if 'Overflow Vol' in df_off.columns:
    correlation_combined = np.corrcoef(vol_diff, combined_effect)[0, 1]
    ax.text(0.02, 0.95, f'Correlation with Evap only: r = {correlation_evap:.3f}\nCorrelation with Net Effect: r = {correlation_combined:.3f}\n\nNet Effect = Evap saved - Overflow lost', 
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax.text(0.02, 0.95, f'Correlation: r = {correlation_evap:.3f}\n\nIf r ≈ 1: Volume change is mostly\nexplained by evaporation difference', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig5.savefig(f'{OUTPUT_DIR}/3_fig_waterbalance_volume_vs_evaporation.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/3_fig_waterbalance_volume_vs_evaporation.png")

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================
print("\n[7] Generating Water Balance Statistics...")

print("\n" + "="*70)
print("WATER BALANCE STATISTICS SUMMARY")
print("="*70)

print(f"\n{'Parameter':<35} {'HP OFF':>15} {'HP ON':>15} {'Difference':>15} {'Unit':>10}")
print("-"*90)

# Lake Level
print(f"{'Mean Lake Level':<35} {df_off['Lake Level'].mean():>15.4f} {df_on['Lake Level'].mean():>15.4f} {df_on['Lake Level'].mean() - df_off['Lake Level'].mean():>+15.4f} {'m':>10}")
print(f"{'Final Lake Level':<35} {df_off['Lake Level'].iloc[-1]:>15.4f} {df_on['Lake Level'].iloc[-1]:>15.4f} {df_on['Lake Level'].iloc[-1] - df_off['Lake Level'].iloc[-1]:>+15.4f} {'m':>10}")

# Lake Volume
print(f"{'Mean Lake Volume':<35} {df_off['Volume'].mean()/1e6:>15.4f} {df_on['Volume'].mean()/1e6:>15.4f} {(df_on['Volume'].mean() - df_off['Volume'].mean())/1e6:>+15.6f} {'×10⁶ m³':>10}")
print(f"{'Final Lake Volume':<35} {df_off['Volume'].iloc[-1]/1e6:>15.4f} {df_on['Volume'].iloc[-1]/1e6:>15.4f} {(df_on['Volume'].iloc[-1] - df_off['Volume'].iloc[-1])/1e6:>+15.6f} {'×10⁶ m³':>10}")

# Evaporation
print(f"{'Mean Daily Evaporation':<35} {-df_off['Evaporation'].mean():>15.2f} {-df_on['Evaporation'].mean():>15.2f} {-df_on['Evaporation'].mean() + df_off['Evaporation'].mean():>+15.2f} {'m³/day':>10}")
print(f"{'Total Evaporation':<35} {-df_off['Evaporation'].sum()/1e6:>15.4f} {-df_on['Evaporation'].sum()/1e6:>15.4f} {(-df_on['Evaporation'].sum() + df_off['Evaporation'].sum())/1e6:>+15.4f} {'×10⁶ m³':>10}")

# ============================================================================
# MASS CONSERVATION VERIFICATION
# ============================================================================
print("\n" + "="*70)
print("MASS CONSERVATION VERIFICATION")
print("="*70)

print(f"\nHeat Pump Flow Rate: {HP_FLOW_RATE:,.0f} m³/day")
print(f"Simulation Duration: {len(df_off)} days")
print(f"Total HP Flow (if NOT conserved): {HP_FLOW_RATE * len(df_off) / 1e6:.2f} × 10⁶ m³")

print(f"\nActual Differences:")
print(f"  Level Difference - Mean:  {np.mean(level_diff)*1000:.4f} mm")
print(f"  Level Difference - Std:   {np.std(level_diff)*1000:.4f} mm")
print(f"  Level Difference - Max:   {np.max(np.abs(level_diff))*1000:.4f} mm")
print(f"  Level Difference - Final: {level_diff[-1]*1000:.4f} mm")
print(f"\n  Volume Difference - Mean:  {np.mean(vol_diff):.2f} m³")
print(f"  Volume Difference - Std:   {np.std(vol_diff):.2f} m³")
print(f"  Volume Difference - Final: {vol_diff[-1]:.2f} m³")

# Calculate drift rate
x = np.arange(len(vol_diff))
coeffs = np.polyfit(x, vol_diff, 1)
slope = coeffs[0]
print(f"\nVolume Drift Rate: {slope:.4f} m³/day")
print(f"  (Expected if NOT conserved: ~{HP_FLOW_RATE:,.0f} m³/day)")

if abs(slope) < 100:
    print("\n✓ MASS CONSERVATION VERIFIED: Volume drift is negligible")
    mass_conserved = True
else:
    print("\n✗ WARNING: Significant volume drift detected")
    mass_conserved = False

print("\n" + "="*70)
print("WATER BALANCE ANALYSIS COMPLETE")
print("="*70)
print(f"\nGenerated figures in {OUTPUT_DIR}/:")
print("  - fig_waterbalance_level_volume.png")
print("  - fig_waterbalance_evaporation.png")
print("  - fig_waterbalance_mass_conservation.png")
print("  - fig_waterbalance_inflow_outflow.png")

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("SUMMARY AND INTERPRETATION OF RESULTS")
print("="*70)

# Calculate key metrics
mean_level_diff = np.mean(level_diff) * 1000  # mm
final_level_diff = level_diff[-1] * 1000  # mm
mean_vol_diff = np.mean(vol_diff)  # m³
final_vol_diff = vol_diff[-1]  # m³
evap_total_diff = (-df_on['Evaporation'].sum() + df_off['Evaporation'].sum()) / 1e6  # ×10⁶ m³
total_hp_flow = HP_FLOW_RATE * len(df_off) / 1e6  # ×10⁶ m³ (if not conserved)

summary = f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    PLOT INTERPRETATION SUMMARY                       │
├─────────────────────────────────────────────────────────────────────┤

FIGURE 1: Lake Level and Volume Comparison
─────────────────────────────────────────────
Panels (a,c) show lake water level and volume time series for HP OFF 
(red) and HP ON (blue) scenarios. Panels (b,d) show the differences.

Key Finding: 
  • Mean level difference: {mean_level_diff:+.4f} mm
  • Final level difference: {final_level_diff:+.4f} mm
  • Mean volume difference: {mean_vol_diff:+.2f} m³

{"→ The heat pump maintains nearly identical water levels, indicating proper mass conservation in the closed-loop system." if abs(mean_level_diff) < 1 else f"→ Small level differences exist ({mean_level_diff:+.4f} mm mean) which may be due to indirect effects on evaporation."}

FIGURE 2: Evaporation Analysis
─────────────────────────────────────────────
Panel (a) shows daily evaporation rates for both scenarios.
Panel (b) shows the evaporation difference (ON - OFF).
Panel (c) shows cumulative evaporation difference over time.

Key Finding:
  • Total evaporation difference: {evap_total_diff:+.4f} ×10⁶ m³
  {"• HP ON has LESS evaporation (cooler surface → reduced evaporative loss)" if evap_total_diff > 0 else "• HP ON has MORE evaporation (warmer surface → increased evaporative loss)" if evap_total_diff < 0 else "• No significant difference in evaporation"}

The evaporation difference is a key indirect effect of the heat pump:
by modifying surface temperature, it affects the lake's water budget.

FIGURE 3: Mass Conservation Verification
─────────────────────────────────────────────
Histograms and time series of level/volume differences between scenarios.

Key Finding:
  • Volume drift rate: {slope:.4f} m³/day
  • Expected drift if NOT conserved: {HP_FLOW_RATE:,.0f} m³/day
  • Conservation status: {"✓ VERIFIED - Heat pump is mass-conserving" if mass_conserved else "✗ WARNING - Possible mass leak"}

{"The heat pump extracts and re-injects the same water volume, so any volume difference is due to secondary effects (evaporation, overflow) rather than direct water loss." if mass_conserved else "Significant volume drift suggests a possible issue with the heat pump mass balance implementation."}

FIGURE 4: Inflows and Outflows
─────────────────────────────────────────────
Compares external water fluxes between scenarios:
  • Panel (a): Total inflow volume (should be identical)
  • Panel (b): Total outflow volume (HP extraction excluded for fair comparison)
  • Panel (c): Net water balance (Inflow - Outflow + Evaporation)
  • Panel (d): Difference in net balance (shows evaporation impact)

Key Finding: By excluding the HP recirculation flow from outflows, the 
remaining difference between scenarios is due to:
  1. Changes in evaporation (from modified surface temperature)
  2. Changes in overflow (from modified lake level)

FIGURE 5: Volume vs Evaporation Difference
─────────────────────────────────────────────
Compares volume difference (ON-OFF) with cumulative evaporation and 
overflow differences to explain what drives the volume change.

Key Finding: The volume difference between scenarios can be explained by:
  • Cumulative evaporation difference
  • Cumulative overflow difference
  • Combined effect (should track volume difference closely)

├─────────────────────────────────────────────────────────────────────┤
│                    KEY STATISTICS TABLE                              │
├─────────────────────────────────────────────────────────────────────┤

Parameter                      HP OFF          HP ON      Difference    Unit
──────────────────────────────────────────────────────────────────────────
Mean Lake Level              {df_off['Lake Level'].mean():>8.4f}      {df_on['Lake Level'].mean():>8.2f}      {df_on['Lake Level'].mean() - df_off['Lake Level'].mean():>+8.4f}      m
Mean Lake Volume             {df_off['Volume'].mean()/1e6:>8.4f}      {df_on['Volume'].mean()/1e6:>8.4f}      {(df_on['Volume'].mean() - df_off['Volume'].mean())/1e6:>+8.6f}      ×10⁶ m³
Total Evaporation            {-df_off['Evaporation'].sum()/1e6:>8.4f}      {-df_on['Evaporation'].sum()/1e6:>8.4f}      {(-df_on['Evaporation'].sum() + df_off['Evaporation'].sum())/1e6:>+8.4f}      ×10⁶ m³

├─────────────────────────────────────────────────────────────────────┤
│                         MAIN CONCLUSIONS                             │
├─────────────────────────────────────────────────────────────────────┤

1. MASS CONSERVATION: {"✓ The heat pump implementation correctly conserves water mass." if mass_conserved else "✗ Mass conservation issue detected - review implementation."}
   Volume drift ({slope:.4f} m³/day) is {"negligible" if abs(slope) < 100 else "significant"} compared to 
   HP flow rate ({HP_FLOW_RATE:,} m³/day).

2. WATER LEVEL IMPACT: The heat pump causes a mean water level change 
   of only {mean_level_diff:+.4f} mm, {"which is within measurement uncertainty and confirms the closed-loop design works correctly." if abs(mean_level_diff) < 10 else "which may indicate indirect effects through evaporation or overflow changes."}

3. EVAPORATION FEEDBACK: {"The cooler lake surface with HP operation reduces evaporation, slightly increasing water retention." if evap_total_diff > 0 else "The warmer lake surface with HP operation increases evaporation, slightly reducing water retention." if evap_total_diff < 0 else "No significant change in evaporation detected."}
   Total evaporation difference: {evap_total_diff:+.4f} ×10⁶ m³ over simulation period.

4. HYDROLOGICAL NEUTRALITY: {"The heat pump is effectively hydrologically neutral - it extracts and reinjects equal volumes without affecting the lake's water budget significantly." if mass_conserved and abs(mean_level_diff) < 10 else "Some hydrological impact detected, likely due to thermal effects on evaporation."}

└─────────────────────────────────────────────────────────────────────┘
"""

print(summary)

# Save summary to text file
summary_file = f'{OUTPUT_DIR}/3_Waterbalance_analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write("WATER BALANCE ANALYSIS - Heat Pump Impact on Lake Hydrology\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Simulation period: {df_off['datetime'].min().strftime('%Y-%m-%d')} to {df_off['datetime'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"Heat pump configuration: Flow={HP_FLOW_RATE:,} m³/day\n")
    f.write(f"Extraction/Injection depth: {HP_EXTRACTION_ELEV}m / {HP_INJECTION_ELEV}m\n")
    f.write("\n")
    f.write(summary)
print(f"\n  Summary saved to: {summary_file}")
