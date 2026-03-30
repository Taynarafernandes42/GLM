#!/usr/bin/env python3
"""
ENERGY BUDGET ANALYSIS - Heat Pump Impact on Lake Energy Balance
Generates all energy/heat flux related plots:
  - Surface heat fluxes (shortwave, longwave, latent, sensible)
  - Heat pump thermal power
  - Lake heat content
  - Energy budget comparison
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Physical constants
RHO_WATER = 1000  # kg/m³
CP_WATER = 4186   # J/(kg·K)

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
print("ENERGY BUDGET ANALYSIS - Heat Pump Impact")
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

print(f"  Period: {df_off['datetime'].min()} to {df_off['datetime'].max()}")
print(f"  Records: {len(df_off)} days")

# Calculate heat pump thermal power
HP_FLOW_RATE_M3S = HP_FLOW_RATE / 86400  # m³/s
THERMAL_POWER_W = RHO_WATER * CP_WATER * HP_FLOW_RATE_M3S * abs(HP_TEMP_CHANGE)
THERMAL_POWER_MW = THERMAL_POWER_W / 1e6

print(f"\n  Heat pump thermal power: {THERMAL_POWER_MW:.3f} MW")

# ============================================================================
# FIGURE 1: Surface Heat Fluxes Comparison
# ============================================================================
print("\n[2] Creating Figure: Surface Heat Fluxes...")

fig1, axes1 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

flux_vars = [
    ('Daily Qsw', 'Shortwave Radiation (absorbed)', 'W/m²'),
    ('Daily Qlw', 'Longwave Radiation (net)', 'W/m²'),
    ('Daily Qe', 'Latent Heat Flux (evaporation)', 'W/m²'),
    ('Daily Qh', 'Sensible Heat Flux', 'W/m²'),
]

for i, (var, title, unit) in enumerate(flux_vars):
    ax = axes1[i]
    ax.plot(df_off['datetime'], df_off[var], 'r-', lw=2.5, alpha=0.85, label='HP OFF')
    ax.plot(df_on['datetime'], df_on[var], 'b-', lw=2.5, alpha=0.85, label='HP ON')
    ax.set_ylabel(f'{unit}')
    ax.set_title(f'({chr(97+i)}) {title}')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add mean values
    mean_off = df_off[var].mean()
    mean_on = df_on[var].mean()
    diff = mean_on - mean_off
    ax.text(0.02, 0.95, f'Mean OFF: {mean_off:.2f}\nMean ON: {mean_on:.2f}\nΔ: {diff:+.3f}', 
            transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

axes1[-1].set_xlabel('Year')
axes1[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes1[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig1.savefig(f'{OUTPUT_DIR}/4_fig_energy_heatfluxes.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/4_fig_energy_heatfluxes.png")

# ============================================================================
# FIGURE 2: Heat Flux Differences
# ============================================================================
print("\n[3] Creating Figure: Heat Flux Differences...")

fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

for i, (var, title, unit) in enumerate(flux_vars):
    ax = axes2[i]
    diff = df_on[var].values - df_off[var].values
    
    ax.plot(df_off['datetime'], diff, 'k-', lw=2.0, alpha=0.8)
    ax.axhline(0, color='gray', ls='--', lw=2.0)
    ax.axhline(np.mean(diff), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(diff):+.3f} W/m²')
    ax.fill_between(df_off['datetime'], 0, diff, where=(diff > 0), alpha=0.3, color='red')
    ax.fill_between(df_off['datetime'], 0, diff, where=(diff < 0), alpha=0.3, color='blue')
    ax.set_ylabel(f'Δ {unit}')
    ax.set_title(f'({chr(97+i)}) {title} Difference (ON - OFF)')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

axes2[-1].set_xlabel('Year')
axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes2[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/4_fig_energy_heatflux_differences.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/4_fig_energy_heatflux_differences.png")

# ============================================================================
# FIGURE 3: Net Surface Heat Budget
# ============================================================================
print("\n[4] Creating Figure: Net Surface Heat Budget...")

fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Calculate net surface heat flux (positive = warming)
# Q_net = Qsw + Qlw + Qe + Qh (all fluxes, Qsw positive, others typically negative)
net_flux_off = df_off['Daily Qsw'] + df_off['Daily Qlw'] + df_off['Daily Qe'] + df_off['Daily Qh']
net_flux_on = df_on['Daily Qsw'] + df_on['Daily Qlw'] + df_on['Daily Qe'] + df_on['Daily Qh']

# Net flux comparison
ax = axes3[0]
ax.plot(df_off['datetime'], net_flux_off, 'r-', lw=2.5, alpha=0.85, label='HP OFF')
ax.plot(df_on['datetime'], net_flux_on, 'b-', lw=2.5, alpha=0.85, label='HP ON')
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.set_ylabel('Net Flux (W/m²)')
ax.set_title('(a) Net Surface Heat Flux (Qsw + Qlw + Qe + Qh)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Net flux difference
ax = axes3[1]
net_diff = net_flux_on.values - net_flux_off.values
ax.plot(df_off['datetime'], net_diff, 'k-', lw=2.0, alpha=0.8)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.axhline(np.mean(net_diff), color='red', ls='-', lw=3.0, label=f'Mean: {np.mean(net_diff):+.3f} W/m²')
ax.fill_between(df_off['datetime'], 0, net_diff, where=(net_diff > 0), alpha=0.3, color='red', label='More warming')
ax.fill_between(df_off['datetime'], 0, net_diff, where=(net_diff < 0), alpha=0.3, color='blue', label='More cooling')
ax.set_ylabel('Δ Net Flux (W/m²)')
ax.set_title('(b) Net Surface Heat Flux Difference (ON - OFF)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Cumulative heat input difference
ax = axes3[2]
# Convert W/m² to total energy: W/m² * area * seconds_per_day / 1e15 = PJ
# Assume lake area ~2 km² = 2e6 m²
LAKE_AREA = 2e6  # m²
SECONDS_PER_DAY = 86400
energy_diff = net_diff * LAKE_AREA * SECONDS_PER_DAY / 1e15  # PJ per day
cumsum_energy = np.cumsum(energy_diff)
ax.plot(df_off['datetime'], cumsum_energy, 'k-', lw=2.5)
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.set_ylabel('Cumulative Δ Energy (PJ)')
ax.set_xlabel('Year')
ax.set_title('(c) Cumulative Surface Heat Input Difference')
ax.text(0.02, 0.95, f'Total: {cumsum_energy[-1]:+.4f} PJ', transform=ax.transAxes,
        fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/4_fig_energy_net_budget.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/4_fig_energy_net_budget.png")

# ============================================================================
# FIGURE 4: Heat Pump Energy Input
# ============================================================================
print("\n[5] Creating Figure: Heat Pump Thermal Power...")

fig4, axes4 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Heat pump removes heat (negative energy input to lake)
# This is constant when heat pump is ON
hp_energy_per_day_J = THERMAL_POWER_W * SECONDS_PER_DAY  # Joules per day
hp_energy_per_day_PJ = hp_energy_per_day_J / 1e15  # PJ per day

# Create heat pump operation indicator (always ON in this simulation)
hp_power_time = np.ones(len(df_off)) * (-THERMAL_POWER_MW)  # Negative = heat removal

ax = axes4[0]
ax.axhline(-THERMAL_POWER_MW, color='purple', ls='-', lw=3.0, label=f'HP Power: {-THERMAL_POWER_MW:.3f} MW')
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.set_ylabel('Power (MW)')
ax.set_title(f'(a) Heat Pump Thermal Power (Flow: {HP_FLOW_RATE:,} m³/day, ΔT: {HP_TEMP_CHANGE}°C)')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.set_ylim([-2, 0.5])

# Cumulative heat removed by heat pump
ax = axes4[1]
hp_cumulative_energy = np.arange(1, len(df_off)+1) * (-hp_energy_per_day_PJ)
ax.plot(df_off['datetime'], hp_cumulative_energy, 'purple', lw=3.0, label='Heat removed by HP')
ax.set_ylabel('Cumulative Heat Removed (PJ)')
ax.set_title('(b) Cumulative Heat Removed by Heat Pump')
ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
ax.text(0.98, 0.05, f'Total: {hp_cumulative_energy[-1]:.3f} PJ over {len(df_off)} days', 
        transform=ax.transAxes, fontsize=12, ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Compare HP heat removal with surface flux change
ax = axes4[2]
ax.plot(df_off['datetime'], cumsum_energy, 'b-', lw=3.0, label='Δ Surface Heat Input')
ax.plot(df_off['datetime'], hp_cumulative_energy, 'purple', lw=3.0, label='Heat Removed by HP')
ax.plot(df_off['datetime'], cumsum_energy + hp_cumulative_energy, 'g-', lw=3.0, label='Net Energy Change')
ax.axhline(0, color='gray', ls='--', lw=2.0)
ax.set_ylabel('Energy (PJ)')
ax.set_xlabel('Year')
ax.set_title('(c) Energy Budget: Surface Flux Change vs Heat Pump Removal')
ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
fig4.savefig(f'{OUTPUT_DIR}/4_fig_energy_heatpump_power.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/4_fig_energy_heatpump_power.png")

# ============================================================================
# FIGURE 5: Component Contributions
# ============================================================================
print("\n[6] Creating Figure: Heat Flux Component Contributions...")

fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))

# Mean heat fluxes - bar chart
ax = axes5[0, 0]
components = ['Qsw', 'Qlw', 'Qe', 'Qh']
labels = ['Shortwave', 'Longwave', 'Latent', 'Sensible']
off_means = [df_off[f'Daily {c}'].mean() for c in components]
on_means = [df_on[f'Daily {c}'].mean() for c in components]

x = np.arange(len(components))
width = 0.35
bars1 = ax.bar(x - width/2, off_means, width, label='HP OFF', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, on_means, width, label='HP ON', color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='gray', ls='-', lw=2.0)
ax.set_ylabel('Mean Heat Flux (W/m²)')
ax.set_title('(a) Mean Surface Heat Flux Components')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize=12, framealpha=0.9)

# Heat flux differences - bar chart
ax = axes5[0, 1]
diffs = [on - off for on, off in zip(on_means, off_means)]
colors = ['red' if d > 0 else 'blue' for d in diffs]
bars = ax.bar(x, diffs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='gray', ls='-', lw=2.0)
ax.set_ylabel('Δ Heat Flux (W/m²)')
ax.set_title('(b) Heat Flux Differences (ON - OFF)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
for bar, d in zip(bars, diffs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{d:+.3f}', 
            ha='center', va='bottom' if d > 0 else 'top', fontsize=12)

# Seasonal heat flux breakdown (OFF simulation)
ax = axes5[1, 0]
seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
seasonal_data_off = {}
df_off['month'] = df_off['datetime'].dt.month
for season, months in seasons.items():
    mask = df_off['month'].isin(months)
    seasonal_data_off[season] = {c: df_off.loc[mask, f'Daily {c}'].mean() for c in components}

x = np.arange(len(seasons))
width = 0.2
for i, (comp, label) in enumerate(zip(components, labels)):
    values = [seasonal_data_off[s][comp] for s in seasons.keys()]
    ax.bar(x + i*width, values, width, label=label, alpha=0.7, edgecolor='black', linewidth=1.0)
ax.axhline(0, color='gray', ls='-', lw=2.0)
ax.set_ylabel('Heat Flux (W/m²)')
ax.set_title('(c) Seasonal Heat Fluxes (HP OFF)')
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(seasons.keys())
ax.legend(fontsize=11, framealpha=0.9)

# Net flux seasonal comparison
ax = axes5[1, 1]
df_on['month'] = df_on['datetime'].dt.month
net_seasonal_off = []
net_seasonal_on = []
for season, months in seasons.items():
    mask_off = df_off['month'].isin(months)
    mask_on = df_on['month'].isin(months)
    net_off = sum(df_off.loc[mask_off, f'Daily {c}'].mean() for c in components)
    net_on = sum(df_on.loc[mask_on, f'Daily {c}'].mean() for c in components)
    net_seasonal_off.append(net_off)
    net_seasonal_on.append(net_on)

x = np.arange(len(seasons))
width = 0.35
ax.bar(x - width/2, net_seasonal_off, width, label='HP OFF', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.bar(x + width/2, net_seasonal_on, width, label='HP ON', color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='gray', ls='-', lw=2.0)
ax.set_ylabel('Net Heat Flux (W/m²)')
ax.set_title('(d) Seasonal Net Surface Heat Flux')
ax.set_xticks(x)
ax.set_xticklabels(seasons.keys())
ax.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
fig5.savefig(f'{OUTPUT_DIR}/4_fig_energy_components.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/4_fig_energy_components.png")

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================
print("\n[7] Generating Energy Budget Statistics...")

print("\n" + "="*70)
print("ENERGY BUDGET STATISTICS SUMMARY")
print("="*70)

print(f"\n{'Parameter':<40} {'HP OFF':>12} {'HP ON':>12} {'Difference':>12} {'Unit':>10}")
print("-"*90)

for var, title, unit in flux_vars:
    off_mean = df_off[var].mean()
    on_mean = df_on[var].mean()
    diff = on_mean - off_mean
    print(f"{title:<40} {off_mean:>12.3f} {on_mean:>12.3f} {diff:>+12.4f} {unit:>10}")

# Net flux
print(f"{'Net Surface Heat Flux':<40} {net_flux_off.mean():>12.3f} {net_flux_on.mean():>12.3f} {net_flux_on.mean()-net_flux_off.mean():>+12.4f} {'W/m²':>10}")

print("\n" + "="*70)
print("HEAT PUMP THERMAL IMPACT")
print("="*70)

print(f"\nHeat Pump Configuration:")
print(f"  Flow rate: {HP_FLOW_RATE:,} m³/day ({HP_FLOW_RATE_M3S:.3f} m³/s)")
print(f"  Temperature change: {HP_TEMP_CHANGE}°C")
print(f"  Extraction depth: {HP_EXTRACTION_ELEV} m")
print(f"  Injection depth: {HP_INJECTION_ELEV} m")

print(f"\nThermal Power:")
print(f"  Power = ρ × Cp × Q × ΔT")
print(f"  Power = {RHO_WATER} × {CP_WATER} × {HP_FLOW_RATE_M3S:.3f} × {abs(HP_TEMP_CHANGE)}")
print(f"  Power = {THERMAL_POWER_W:.0f} W = {THERMAL_POWER_MW:.3f} MW")

print(f"\nEnergy Removed Over Simulation:")
print(f"  Duration: {len(df_off)} days")
print(f"  Daily energy: {hp_energy_per_day_J/1e9:.3f} GJ/day")
print(f"  Total energy: {hp_cumulative_energy[-1]:.4f} PJ")

print(f"\nSurface Heat Flux Response:")
print(f"  Change in net surface flux: {np.mean(net_diff):+.4f} W/m²")
print(f"  Cumulative surface energy change: {cumsum_energy[-1]:+.4f} PJ")
print(f"  Net energy change (surface + HP): {cumsum_energy[-1] + hp_cumulative_energy[-1]:.4f} PJ")

print("\n" + "="*70)
print("ENERGY BUDGET ANALYSIS COMPLETE")
print("="*70)
print(f"\nGenerated figures in {OUTPUT_DIR}/:")
print("  - fig_energy_heatfluxes.png")
print("  - fig_energy_heatflux_differences.png")
print("  - fig_energy_net_budget.png")
print("  - fig_energy_heatpump_power.png")
print("  - fig_energy_components.png")

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("SUMMARY AND INTERPRETATION OF RESULTS")
print("="*70)

# Calculate key metrics for interpretation
mean_net_flux_diff = net_flux_on.mean() - net_flux_off.mean()
qsw_diff = df_on['Daily Qsw'].mean() - df_off['Daily Qsw'].mean()
qlw_diff = df_on['Daily Qlw'].mean() - df_off['Daily Qlw'].mean()
qe_diff = df_on['Daily Qe'].mean() - df_off['Daily Qe'].mean()
qh_diff = df_on['Daily Qh'].mean() - df_off['Daily Qh'].mean()

total_hp_heat_removed = hp_cumulative_energy[-1]  # PJ
total_surface_energy_change = cumsum_energy[-1]  # PJ
net_energy_change = total_surface_energy_change + total_hp_heat_removed  # PJ

summary = f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    PLOT INTERPRETATION SUMMARY                       │
├─────────────────────────────────────────────────────────────────────┤

FIGURE 1: Surface Heat Fluxes
─────────────────────────────────────────────
Time series of the four main surface heat flux components for HP OFF 
(red) and HP ON (blue):
  • Qsw: Shortwave radiation (solar heating - always positive)
  • Qlw: Longwave radiation (thermal exchange with atmosphere)
  • Qe: Latent heat flux (evaporative cooling - usually negative)
  • Qh: Sensible heat flux (conductive exchange)

Key Finding: The heat pump affects surface heat fluxes indirectly by 
modifying lake surface temperature:
  • ΔQsw: {qsw_diff:+.4f} W/m² (minimal change - incoming solar is unchanged)
  • ΔQlw: {qlw_diff:+.4f} W/m² {"(cooler surface emits less longwave)" if qlw_diff > 0 else "(warmer surface emits more longwave)"}
  • ΔQe: {qe_diff:+.4f} W/m² {"(reduced evaporation from cooler surface)" if qe_diff > 0 else "(increased evaporation from warmer surface)"}
  • ΔQh: {qh_diff:+.4f} W/m² (sensible heat exchange change)

FIGURE 2: Heat Flux Differences
─────────────────────────────────────────────
Shows the difference (ON - OFF) for each flux component over time,
highlighting seasonal variations in the heat pump's impact.

Key Finding: {"Latent heat (Qe) shows the largest response, indicating evaporation changes dominate the surface energy feedback." if abs(qe_diff) > max(abs(qlw_diff), abs(qh_diff)) else "Multiple flux components respond to the changed surface temperature."}

FIGURE 3: Net Surface Energy Budget
─────────────────────────────────────────────
Panel (a): Net surface heat flux (Qsw + Qlw + Qe + Qh) for both scenarios
Panel (b): Difference in net flux (ON - OFF) with positive = more warming
Panel (c): Cumulative energy change from modified surface fluxes

Key Finding:
  • Mean net flux difference: {mean_net_flux_diff:+.4f} W/m²
  • Cumulative surface energy change: {total_surface_energy_change:+.4f} PJ

{"→ The lake surface receives MORE energy with HP operation (positive feedback)" if mean_net_flux_diff > 0 else "→ The lake surface receives LESS energy with HP operation (negative feedback)" if mean_net_flux_diff < 0 else "→ No significant change in surface energy budget"}

FIGURE 4: Heat Pump Thermal Power
─────────────────────────────────────────────
Panel (a): Heat pump thermal power ({THERMAL_POWER_MW:.3f} MW = constant)
Panel (b): Cumulative heat removed by heat pump
Panel (c): Energy budget comparison (surface change vs HP removal)

Key Finding:
  • Heat pump power: {THERMAL_POWER_MW:.3f} MW (= ρ × Cp × Q × ΔT)
  • Total heat removed: {abs(total_hp_heat_removed):.4f} PJ over {len(df_off)} days
  • Surface energy feedback: {total_surface_energy_change:+.4f} PJ
  • Net energy change: {net_energy_change:+.4f} PJ

{"→ Surface feedback partially compensates for HP heat removal" if abs(total_surface_energy_change) > 0.1 * abs(total_hp_heat_removed) else "→ Surface feedback is small compared to HP heat removal"}

FIGURE 5: Energy Component Analysis
─────────────────────────────────────────────
Bar charts comparing mean heat flux values and seasonal patterns:
  • Mean flux components for OFF vs ON
  • Flux differences (ON - OFF)
  • Seasonal breakdown (DJF, MAM, JJA, SON)

Key Finding: Seasonal analysis reveals when the heat pump has the 
greatest surface energy impact (typically during stratified periods
when surface temperature changes are most pronounced).

├─────────────────────────────────────────────────────────────────────┤
│                         MAIN CONCLUSIONS                             │
├─────────────────────────────────────────────────────────────────────┤

1. DIRECT HEAT EXTRACTION: The heat pump removes {abs(total_hp_heat_removed):.4f} PJ 
   of thermal energy over the simulation period, operating at a constant 
   power of {THERMAL_POWER_MW:.3f} MW.

2. SURFACE FEEDBACK MECHANISM: By cooling the lake, the heat pump 
   triggers a surface energy feedback:
   {"• Reduced evaporation (cooler surface → less evaporative heat loss)" if qe_diff > 0 else "• Increased evaporation"}
   {"• Reduced longwave emission (cooler surface → less thermal radiation)" if qlw_diff > 0 else "• Increased longwave emission"}
   Net surface feedback: {total_surface_energy_change:+.4f} PJ

3. ENERGY BALANCE: 
   Heat removed by HP:        {total_hp_heat_removed:+.4f} PJ
   Surface energy change:     {total_surface_energy_change:+.4f} PJ
   ─────────────────────────────────────────
   Net energy change:         {net_energy_change:+.4f} PJ

   {"The surface feedback partially offsets the heat extraction, reducing the net cooling effect." if total_surface_energy_change > 0 and total_hp_heat_removed < 0 else "The net energy change represents the lake's thermal response to heat pump operation."}

4. EFFICIENCY IMPLICATIONS: The feedback fraction is 
   {abs(total_surface_energy_change/total_hp_heat_removed)*100:.1f}% of HP heat removal.
   {"This means the lake partially 'resists' cooling through modified surface fluxes." if abs(total_surface_energy_change) > 0.1 * abs(total_hp_heat_removed) else "Surface feedback is relatively small, so most HP cooling is retained."}

5. THERMAL EQUILIBRIUM: {"The system appears to approach a new thermal equilibrium where reduced surface heat losses partially compensate for HP heat extraction." if abs(total_surface_energy_change) > 0.2 * abs(total_hp_heat_removed) else "The lake has not fully adjusted its surface fluxes to compensate for HP operation."}

└─────────────────────────────────────────────────────────────────────┘
"""

print(summary)

# Save summary to text file
summary_file = f'{OUTPUT_DIR}/4_Energy_analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write("ENERGY BUDGET ANALYSIS - Heat Pump Impact on Lake Energy Balance\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Simulation period: {df_off['datetime'].min().strftime('%Y-%m-%d')} to {df_off['datetime'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"Heat pump configuration: Flow={HP_FLOW_RATE:,} m³/day, ΔT={HP_TEMP_CHANGE}°C\n")
    f.write(f"Thermal power: {THERMAL_POWER_MW:.3f} MW\n")
    f.write(f"Extraction/Injection depth: {HP_EXTRACTION_ELEV}m / {HP_INJECTION_ELEV}m\n")
    f.write("\n")
    f.write(summary)
print(f"\n  Summary saved to: {summary_file}")
