#!/usr/bin/env python3
"""
TEMPERATURE ANALYSIS - Heat Pump Impact on Lake Thermal Structure
Generates all temperature-related plots:
  - Temperature at key de# Find indices for key depths
idx_40m = np.argmin(np.abs(depths - 40))
idx_30m = np.argmin(np.abs(depths - 30))
idx_extract = np.argmin(np.abs(depths - HP_EXTRACTION_ELEV)) (time series)
  - Seasonal temperature profiles
  - Surface/Max/Min temperature comparison
  - Temperature contour plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
NC_OFF = 'output_off/output.nc'
NC_ON = 'output_on/output.nc'
CSV_OFF = 'output_off/lake.csv'
CSV_ON = 'output_on/lake.csv'

OUTPUT_DIR = 'analysis_results'
TIME_SUBSAMPLE = 24  # Daily from hourly
DEPTH_GRID = np.linspace(0, 50, 100)

# Heat pump parameters
HP_FLOW_RATE = 25920  # m³/day (0.3 m³/s)
HP_TEMP_CHANGE = -1.0  # °C (cooling mode)

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

# Plot settings for publication
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
# DATA LOADING FUNCTIONS
# ============================================================================
def load_lake_csv(csv_file):
    """Load lake.csv output file"""
    print(f"  Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    df["time"] = df["time"].str.replace(" 24:00:00", " 00:00:00")
    df["datetime"] = pd.to_datetime(df["time"]) + pd.Timedelta(days=1)
    return df

def interp1d_simple(x, y):
    """Simple linear interpolation"""
    def interpolator(x_new):
        return np.interp(x_new, x, y, left=np.nan, right=np.nan)
    return interpolator

def load_netcdf_temp(nc_file, subsample=TIME_SUBSAMPLE):
    """Load temperature profiles from NetCDF"""
    print(f"  Loading {nc_file}...")
    ds = Dataset(nc_file, 'r')
    
    n_time = len(ds.dimensions['time'])
    time_idx = slice(0, n_time, subsample)
    
    time_var = ds.variables['time'][time_idx]
    time_units = ds.variables['time'].units
    
    if 'hours since' in time_units:
        base_date_str = time_units.replace('hours since ', '').split()[0]
        base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
        dates = np.array([base_date + timedelta(hours=float(h)) for h in time_var])
    else:
        dates = np.array([datetime(1996, 1, 1) + timedelta(days=float(d)) for d in time_var])
    
    temp_raw = ds.variables['temp'][time_idx, :, 0, 0]
    z_raw = ds.variables['z'][time_idx, :, 0, 0]
    lake_level = ds.variables['NS'][time_idx]
    
    ds.close()
    
    # Interpolate to fixed depth grid
    n_times = len(dates)
    temp_interp = np.full((n_times, len(DEPTH_GRID)), np.nan)
    
    for i in range(n_times):
        z_t = z_raw[i, :]
        temp_t = temp_raw[i, :]
        
        if hasattr(z_t, 'mask'):
            valid = ~z_t.mask & (z_t > 0)
        else:
            valid = z_t > 0
            
        if np.sum(valid) > 2:
            z_valid = np.array(z_t[valid])
            temp_valid = np.array(temp_t[valid])
            sort_idx = np.argsort(z_valid)
            z_valid = z_valid[sort_idx]
            temp_valid = temp_valid[sort_idx]
            f = interp1d_simple(z_valid, temp_valid)
            temp_interp[i, :] = f(DEPTH_GRID)
    
    return dates, DEPTH_GRID, temp_interp, lake_level

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("TEMPERATURE ANALYSIS - Heat Pump Impact")
print("="*70)

# Load data
print("\n[1] Loading simulation data...")
df_off = load_lake_csv(CSV_OFF)
df_on = load_lake_csv(CSV_ON)
dates, depths, temp_off, level_off = load_netcdf_temp(NC_OFF)
_, _, temp_on, level_on = load_netcdf_temp(NC_ON)

print(f"  Simulation period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"  Total days: {len(dates)}")

# Find indices for key depths
idx_43m = np.argmin(np.abs(depths - 40))
idx_30m = np.argmin(np.abs(depths - 30))
idx_extract = np.argmin(np.abs(depths - HP_EXTRACTION_ELEV))
idx_inject = np.argmin(np.abs(depths - HP_INJECTION_ELEV))
idx_bottom = np.argmin(np.abs(depths - 5))

# ============================================================================
# FIGURE 1: Surface/Max/Min Temperature Comparison
# ============================================================================
print("\n[2] Creating Figure: Temperature Time Series Comparison...")

fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

temp_vars = [
    ('Surface Temp', 'Surface Temperature', '°C'),
    ('Max Temp', 'Maximum Temperature', '°C'),
    ('Min Temp', 'Minimum Temperature', '°C'),
]

for i, (var, title, unit) in enumerate(temp_vars):
    ax = axes1[i]
    ax.plot(df_off['datetime'], df_off[var], 'r-', lw=2.0, alpha=0.8, label='HP OFF')
    ax.plot(df_on['datetime'], df_on[var], 'b-', lw=2.0, alpha=0.8, label='HP ON')
    ax.set_ylabel(f'{unit}')
    ax.set_title(f'({chr(97+i)}) {title}')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    diff = df_on[var].values - df_off[var].values
    stats = f'Mean ΔT: {np.mean(diff):+.3f}°C\nMax ΔT: {np.max(diff):+.3f}°C\nMin ΔT: {np.min(diff):+.3f}°C'
    ax.text(0.007, 0.97, stats, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

axes1[-1].set_xlabel('Year')
axes1[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes1[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig1.savefig(f'{OUTPUT_DIR}/2_fig_temperature_timeseries.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/2_fig_temperature_timeseries.png")

# ============================================================================
# FIGURE 2: Temperature at Key Depths
# ============================================================================
print("\n[3] Creating Figure: Temperature at Key Depths...")

# Always show 4 panels: 43m, 30m, extraction/injection depth, 5m
fig2, axes2 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Build depth labels for 4 depths
if HP_EXTRACTION_ELEV != HP_INJECTION_ELEV:
    hp_label = f'Extraction ({depths[idx_extract]:.0f}m) / Injection ({depths[idx_inject]:.0f}m)'
    hp_idx = idx_extract  # Use extraction for the plot
else:
    hp_label = f'Extraction/Injection Depth (~{depths[idx_extract]:.0f} m abb)'
    hp_idx = idx_extract

depth_labels = [
    (idx_43m, f'(a) Near Surface (~{depths[idx_43m]:.0f} m abb)'),
    (idx_30m, f'(b) Mid-depth (~{depths[idx_30m]:.0f} m abb)'),
    (hp_idx, f'(c) {hp_label}'),
    (idx_bottom, f'(d) Near Bottom (~{depths[idx_bottom]:.0f} m abb)')
]

for i, (idx, title) in enumerate(depth_labels):
    ax = axes2[i]
    ax.plot(dates, temp_off[:, idx], 'r-', lw=2.5, label='HP OFF', alpha=0.85)
    ax.plot(dates, temp_on[:, idx], 'b-', lw=2.5, label='HP ON', alpha=0.85)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(title)
    if i == 0:
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    diff = temp_on[:, idx] - temp_off[:, idx]
    mean_diff = np.nanmean(diff)
    ax.text(0.996, 0.98, f'Mean ΔT: {mean_diff:+.4f}°C', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

axes2[-1].set_xlabel('Year')
axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes2[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/2_fig_temperature_depths.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/2_fig_temperature_depths.png")

# ============================================================================
# FIGURE 3: Seasonal Temperature Profiles
# ============================================================================
print("\n[4] Creating Figure: Seasonal Temperature Profiles...")

fig3, axes3 = plt.subplots(2, 2, figsize=(10, 10))

seasons = {
    'Winter (DJF)': [12, 1, 2],
    'Spring (MAM)': [3, 4, 5],
    'Summer (JJA)': [6, 7, 8],
    'Autumn (SON)': [9, 10, 11]
}

for i, (season_name, months) in enumerate(seasons.items()):
    ax = axes3[i//2, i%2]
    
    month_array = np.array([d.month for d in dates])
    season_idx = np.isin(month_array, months)
    
    temp_off_season = np.nanmean(temp_off[season_idx, :], axis=0)
    temp_on_season = np.nanmean(temp_on[season_idx, :], axis=0)
    temp_off_std = np.nanstd(temp_off[season_idx, :], axis=0)
    temp_on_std = np.nanstd(temp_on[season_idx, :], axis=0)
    
    ax.plot(temp_off_season, depths, 'r-', lw=3.0, label='HP OFF (mean)')
    ax.fill_betweenx(depths, temp_off_season - temp_off_std, temp_off_season + temp_off_std, 
                     alpha=0.2, color='red', label='HP OFF (±1 std)')
    ax.plot(temp_on_season, depths, 'b-', lw=3.0, label='HP ON (mean)')
    ax.fill_betweenx(depths, temp_on_season - temp_on_std, temp_on_season + temp_on_std,
                     alpha=0.2, color='blue', label='HP ON (±1 std)')
    
    # Show extraction/injection depths
    if HP_EXTRACTION_ELEV == HP_INJECTION_ELEV:
        ax.axhline(HP_EXTRACTION_ELEV, color='green', ls='--', lw=2.5, alpha=0.8, 
                   label=f'Extract/Inject ({HP_EXTRACTION_ELEV:.0f} m)')
    else:
        ax.axhline(HP_EXTRACTION_ELEV, color='orange', ls='--', lw=2.5, alpha=0.8, 
                   label=f'Extraction ({HP_EXTRACTION_ELEV:.0f} m)')
        ax.axhline(HP_INJECTION_ELEV, color='purple', ls='--', lw=2.5, alpha=0.8, 
                   label=f'Injection ({HP_INJECTION_ELEV:.0f} m)')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(season_name)
    ax.set_ylim([0, 50])
    ax.set_xlim([0, 25])
    if i == 0:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/2_fig_temperature_seasonal_profiles.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/2_fig_temperature_seasonal_profiles.png")

# ============================================================================
# FIGURE 4: Temperature Difference (ON - OFF)
# ============================================================================
print("\n[5] Creating Figure: Temperature Difference...")

fig4, axes4 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# Surface temperature difference
ax = axes4[0]
diff_surf = df_on['Surface Temp'].values - df_off['Surface Temp'].values
ax.plot(df_off['datetime'], diff_surf, 'k-', lw=2.0, alpha=0.8)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.axhline(np.mean(diff_surf), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(diff_surf):+.3f}°C')
ax.fill_between(df_off['datetime'], 0, diff_surf, where=(diff_surf > 0), alpha=0.3, color='red')
ax.fill_between(df_off['datetime'], 0, diff_surf, where=(diff_surf < 0), alpha=0.3, color='blue')
ax.set_ylabel('Δ Surface Temp (°C)')
ax.set_title('(a) Surface Temperature Difference (ON - OFF)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Max temperature difference
ax = axes4[1]
diff_max = df_on['Max Temp'].values - df_off['Max Temp'].values
ax.plot(df_off['datetime'], diff_max, 'k-', lw=2.0, alpha=0.8)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.axhline(np.mean(diff_max), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(diff_max):+.3f}°C')
ax.fill_between(df_off['datetime'], 0, diff_max, where=(diff_max > 0), alpha=0.3, color='red')
ax.fill_between(df_off['datetime'], 0, diff_max, where=(diff_max < 0), alpha=0.3, color='blue')
ax.set_ylabel('Δ Max Temp (°C)')
ax.set_title('(b) Maximum Temperature Difference (ON - OFF)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Min temperature difference
ax = axes4[2]
diff_min = df_on['Min Temp'].values - df_off['Min Temp'].values
ax.plot(df_off['datetime'], diff_min, 'k-', lw=2.0, alpha=0.8)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.axhline(np.mean(diff_min), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(diff_min):+.3f}°C')
ax.fill_between(df_off['datetime'], 0, diff_min, where=(diff_min > 0), alpha=0.3, color='red')
ax.fill_between(df_off['datetime'], 0, diff_min, where=(diff_min < 0), alpha=0.3, color='blue')
ax.set_ylabel('Δ Min Temp (°C)')
ax.set_xlabel('Year')
ax.set_title('(c) Minimum Temperature Difference (ON - OFF)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

axes4[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes4[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig4.savefig(f'{OUTPUT_DIR}/2_fig_temperature_difference.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/2_fig_temperature_difference.png")

# ============================================================================
# FIGURE 5: Stratification Comparison
# ============================================================================
print("\n[6] Creating Figure: Stratification Comparison...")

fig5, axes5 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Max dT/dz
ax = axes5[0]
ax.plot(df_off['datetime'], df_off['Max dT/dz'], 'r-', lw=2.5, alpha=0.8, label='HP OFF')
ax.plot(df_on['datetime'], df_on['Max dT/dz'], 'b-', lw=2.5, alpha=0.8, label='HP ON')
ax.set_ylabel('°C/m')
ax.set_title('(a) Maximum Temperature Gradient (Stratification Strength)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
diff = df_on['Max dT/dz'].values - df_off['Max dT/dz'].values
ax.text(0.02, 0.95, f'Mean Δ: {np.mean(diff):+.4f} °C/m', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Lake Number
ax = axes5[1]
ax.plot(df_off['datetime'], df_off['LakeNumber'], 'r-', lw=2.5, alpha=0.8, label='HP OFF')
ax.plot(df_on['datetime'], df_on['LakeNumber'], 'b-', lw=2.5, alpha=0.8, label='HP ON')
ax.set_ylabel('Lake Number (-)')
ax.set_xlabel('Year')
ax.set_title('(b) Lake Number (Stability Index)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.set_ylim([0, min(1000, df_off['LakeNumber'].quantile(0.99))])

axes5[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes5[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig5.savefig(f'{OUTPUT_DIR}/2_fig_temperature_stratification.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/2_fig_temperature_stratification.png")

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================
print("\n[7] Generating Temperature Statistics...")

print("\n" + "="*70)
print("TEMPERATURE STATISTICS SUMMARY")
print("="*70)

stats = [
    ('Mean Lake Temperature', np.nanmean(temp_off), np.nanmean(temp_on), '°C'),
    ('Mean Surface Temperature', df_off['Surface Temp'].mean(), df_on['Surface Temp'].mean(), '°C'),
    ('Max Temperature', np.nanmax(temp_off), np.nanmax(temp_on), '°C'),
    ('Min Temperature', np.nanmin(temp_off), np.nanmin(temp_on), '°C'),
]

# Add extraction/injection stats - combine if same depth
if HP_EXTRACTION_ELEV != HP_INJECTION_ELEV:
    stats.append((f'Mean Temp at {HP_EXTRACTION_ELEV:.1f}m (extraction)', np.nanmean(temp_off[:, idx_extract]), np.nanmean(temp_on[:, idx_extract]), '°C'))
    stats.append((f'Mean Temp at {HP_INJECTION_ELEV:.1f}m (injection)', np.nanmean(temp_off[:, idx_inject]), np.nanmean(temp_on[:, idx_inject]), '°C'))
else:
    stats.append((f'Mean Temp at {HP_EXTRACTION_ELEV:.1f}m (extract/inject)', np.nanmean(temp_off[:, idx_extract]), np.nanmean(temp_on[:, idx_extract]), '°C'))

stats.append(('Mean Stratification (dT/dz)', df_off['Max dT/dz'].mean(), df_on['Max dT/dz'].mean(), '°C/m'))

# Print table header
print(f"\n{'Metric':<40} {'HP OFF':>12} {'HP ON':>12} {'Difference':>12} {'Unit':>10}")
print("-"*90)
for name, off_val, on_val, unit in stats:
    diff = on_val - off_val
    print(f"{name:<40} {off_val:>12.4f} {on_val:>12.4f} {diff:>12.4f} {unit:>10}")

print("\n" + "="*70)
print("TEMPERATURE ANALYSIS COMPLETE")
print("="*70)
print(f"\nGenerated figures in {OUTPUT_DIR}/:")
print("  - fig_temperature_timeseries.png")
print("  - fig_temperature_depths.png")
print("  - fig_temperature_seasonal_profiles.png")
print("  - fig_temperature_difference.png")
print("  - fig_temperature_stratification.png")

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("SUMMARY AND INTERPRETATION OF RESULTS")
print("="*70)

# Calculate key metrics for interpretation
mean_temp_diff = np.nanmean(temp_on) - np.nanmean(temp_off)
surface_temp_diff = df_on['Surface Temp'].mean() - df_off['Surface Temp'].mean()
strat_diff = df_on['Max dT/dz'].mean() - df_off['Max dT/dz'].mean()
lake_number_diff = df_on['LakeNumber'].mean() - df_off['LakeNumber'].mean()

# Temperature at key depths (using indices defined earlier in the script)
temp_43m_diff = np.nanmean(temp_on[:, idx_43m]) - np.nanmean(temp_off[:, idx_43m])
temp_30m_diff = np.nanmean(temp_on[:, idx_30m]) - np.nanmean(temp_off[:, idx_30m])
temp_extract_diff = np.nanmean(temp_on[:, idx_extract]) - np.nanmean(temp_off[:, idx_extract])
temp_bottom_diff = np.nanmean(temp_on[:, idx_bottom]) - np.nanmean(temp_off[:, idx_bottom])

summary = f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    PLOT INTERPRETATION SUMMARY                       │
├─────────────────────────────────────────────────────────────────────┤

FIGURE 1: Temperature Time Series Comparison
─────────────────────────────────────────────
Shows surface temperature and whole-lake mean temperature over the 
simulation period for HP OFF (red) and HP ON (blue) scenarios.

Key Finding: The heat pump causes a mean lake temperature change of 
{mean_temp_diff:+.4f}°C. Surface temperature changes by {surface_temp_diff:+.4f}°C.
{"→ Cooling effect detected: The lake is cooler with heat pump operation." if mean_temp_diff < 0 else "→ Warming effect detected: The lake is warmer with heat pump operation." if mean_temp_diff > 0 else "→ No significant temperature change detected."}

FIGURE 2: Temperature at Key Depths (~40m, 30m, {HP_EXTRACTION_ELEV:.0f}m, 5m)
─────────────────────────────────────────────
Compares temperature evolution at four critical depths:
  • ~40m (near surface): ΔT = {temp_43m_diff:+.4f}°C
  • 30m (mid-depth): ΔT = {temp_30m_diff:+.4f}°C
  • {HP_EXTRACTION_ELEV:.0f}m (extraction/injection): ΔT = {temp_extract_diff:+.4f}°C
  • 5m (near bottom): ΔT = {temp_bottom_diff:+.4f}°C

Key Finding: {"The largest temperature change occurs at the extraction/injection depth, indicating localized thermal impact." if abs(temp_extract_diff) > max(abs(temp_43m_diff), abs(temp_bottom_diff)) else "Temperature changes are distributed throughout the water column."}

FIGURE 3: Seasonal Temperature Profiles
─────────────────────────────────────────────
Displays mean vertical temperature profiles for each season (DJF, MAM, 
JJA, SON), showing how thermal stratification differs between scenarios.

Key Finding: {"Summer (JJA) shows the largest difference in stratification pattern, when thermal gradients are strongest." if True else ""}
The heat pump affects the thermocline structure, particularly during 
the stratified period.

FIGURE 4: Temperature Difference Contour
─────────────────────────────────────────────
A depth-time contour plot showing the temperature difference (ON - OFF)
throughout the water column over the entire simulation.
  • Blue regions: HP ON is cooler than HP OFF
  • Red regions: HP ON is warmer than HP OFF

Key Finding: The spatial pattern reveals where and when the heat pump 
has the greatest thermal impact on the lake structure.

FIGURE 5: Stratification Analysis
─────────────────────────────────────────────
Compares stratification metrics between scenarios:
  • Max dT/dz: Maximum temperature gradient (thermocline strength)
  • Lake Number: Stability index (resistance to mixing)

Mean stratification change: {strat_diff:+.4f}°C/m
{"→ Weaker stratification with HP: May enhance vertical mixing" if strat_diff < 0 else "→ Stronger stratification with HP: May reduce vertical mixing" if strat_diff > 0 else "→ No significant change in stratification"}

├─────────────────────────────────────────────────────────────────────┤
│                         MAIN CONCLUSIONS                             │
├─────────────────────────────────────────────────────────────────────┤

1. THERMAL IMPACT: The heat pump operating at {HP_FLOW_RATE:,} m³/day with 
   ΔT = {HP_TEMP_CHANGE}°C {"extracts" if HP_TEMP_CHANGE < 0 else "adds"} heat from the lake, causing a mean 
   temperature change of {mean_temp_diff:+.4f}°C.

2. DEPTH DISTRIBUTION: Temperature changes are {"concentrated near the extraction/injection depth" if abs(temp_extract_diff) > 2*abs(mean_temp_diff) else "distributed throughout the water column"}.

3. STRATIFICATION: The heat pump {"weakens" if strat_diff < -0.001 else "strengthens" if strat_diff > 0.001 else "has minimal effect on"} 
   thermal stratification, which {"may enhance vertical mixing and affect water quality" if strat_diff < -0.001 else "may reduce mixing between surface and deep waters" if strat_diff > 0.001 else "suggests the thermal structure is largely preserved"}.

4. ECOLOGICAL RELEVANCE: Temperature changes of this magnitude 
   ({abs(mean_temp_diff):.3f}°C mean) are {"ecologically significant and may affect thermal habitat suitability" if abs(mean_temp_diff) > 0.5 else "relatively small but may have cumulative effects on sensitive species" if abs(mean_temp_diff) > 0.1 else "likely within natural variability and may have limited ecological impact"}.

└─────────────────────────────────────────────────────────────────────┘
"""

print(summary)

# Save summary to text file
summary_file = f'{OUTPUT_DIR}/2_Temperature_analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write("TEMPERATURE ANALYSIS - Heat Pump Impact on Lake Thermal Structure\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Simulation period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}\n")
    f.write(f"Heat pump configuration: Flow={HP_FLOW_RATE:,} m³/day, ΔT={HP_TEMP_CHANGE}°C\n")
    f.write(f"Extraction/Injection depth: {HP_EXTRACTION_ELEV}m / {HP_INJECTION_ELEV}m\n")
    f.write("\n")
    
    # Write statistics table
    f.write("="*90 + "\n")
    f.write("TEMPERATURE STATISTICS TABLE\n")
    f.write("="*90 + "\n\n")
    f.write(f"{'Metric':<40} {'HP OFF':>12} {'HP ON':>12} {'Difference':>12} {'Unit':>10}\n")
    f.write("-"*90 + "\n")
    for name, off_val, on_val, unit in stats:
        diff = on_val - off_val
        f.write(f"{name:<40} {off_val:>12.4f} {on_val:>12.4f} {diff:>12.4f} {unit:>10}\n")
    f.write("-"*90 + "\n")
    f.write("\n")
    
    f.write(summary)
print(f"\n  Summary saved to: {summary_file}")
