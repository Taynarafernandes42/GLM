#!/usr/bin/env python3
"""
Heat Pump Diagnostic Plot - Demonstrates heat pump is working correctly
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset
from datetime import datetime, timedelta
import os

# Configuration
NC_OFF = 'output_off/output.nc'
NC_ON = 'output_on/output.nc'
OUTPUT_DIR = os.path.expanduser('~/Desktop/GLM_Analysis_Plots_3K')
HP_DEPTH = 25.0  # Heat pump extraction/injection depth (m above bottom)

def nc_time_to_datetime(nc_time):
    """Convert NetCDF time (Julian day) to datetime"""
    return [datetime(1,1,1) + timedelta(days=float(t)-1) for t in nc_time]

def get_temp_at_height(nc_file, target_height):
    """Extract temperature at a specific height above bottom"""
    with Dataset(nc_file, 'r') as nc:
        time = nc.variables['time'][:]
        temp = nc.variables['temp'][:]  # (time, layers)
        z = nc.variables['z'][:]  # layer heights
        
        # Get temperature at target height for each timestep
        temps = []
        for t in range(len(time)):
            z_t = z[t, :]
            temp_t = temp[t, :]
            # Find valid layers
            valid = ~np.isnan(z_t) & ~np.isnan(temp_t)
            if np.sum(valid) > 1:
                z_valid = z_t[valid]
                temp_valid = temp_t[valid]
                # Interpolate to target height
                if target_height <= np.max(z_valid) and target_height >= np.min(z_valid):
                    temps.append(np.interp(target_height, z_valid, temp_valid))
                else:
                    temps.append(np.nan)
            else:
                temps.append(np.nan)
        
        dates = nc_time_to_datetime(time)
    return np.array(dates), np.array(temps)

print("Loading data...")
dates_off, temp_off_25m = get_temp_at_height(NC_OFF, HP_DEPTH)
dates_on, temp_on_25m = get_temp_at_height(NC_ON, HP_DEPTH)

# Also get surface and bottom temps
with Dataset(NC_OFF, 'r') as nc:
    time = nc.variables['time'][:]
    temp = nc.variables['temp'][:]
    surf_temp_off = np.array([temp[t, ~np.isnan(temp[t, :])].max() if np.sum(~np.isnan(temp[t, :])) > 0 else np.nan for t in range(len(time))])
    bot_temp_off = np.array([temp[t, ~np.isnan(temp[t, :])].min() if np.sum(~np.isnan(temp[t, :])) > 0 else np.nan for t in range(len(time))])

with Dataset(NC_ON, 'r') as nc:
    temp = nc.variables['temp'][:]
    surf_temp_on = np.array([temp[t, ~np.isnan(temp[t, :])].max() if np.sum(~np.isnan(temp[t, :])) > 0 else np.nan for t in range(len(time))])
    bot_temp_on = np.array([temp[t, ~np.isnan(temp[t, :])].min() if np.sum(~np.isnan(temp[t, :])) > 0 else np.nan for t in range(len(time))])

dates = nc_time_to_datetime(time)

# Subsample for cleaner plot
step = 24  # Daily
dates_sub = dates[::step]
temp_off_25m_sub = temp_off_25m[::step]
temp_on_25m_sub = temp_on_25m[::step]
surf_off_sub = surf_temp_off[::step]
surf_on_sub = surf_temp_on[::step]
bot_off_sub = bot_temp_off[::step]
bot_on_sub = bot_temp_on[::step]

# Create figure
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel (a): Temperature at heat pump depth (25m)
ax1 = axes[0]
ax1.plot(dates_sub, temp_off_25m_sub, 'b-', alpha=0.7, linewidth=0.8, label='HP OFF')
ax1.plot(dates_sub, temp_on_25m_sub, 'r-', alpha=0.7, linewidth=0.8, label='HP ON')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('Temperature (°C)')
ax1.set_title(f'(a) Temperature at Heat Pump Depth ({HP_DEPTH}m above bottom)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel (b): Temperature difference at 25m
ax2 = axes[1]
temp_diff_25m = temp_on_25m_sub - temp_off_25m_sub
ax2.fill_between(dates_sub, 0, temp_diff_25m, where=temp_diff_25m<0, 
                  color='blue', alpha=0.5, label='Cooling (expected)')
ax2.fill_between(dates_sub, 0, temp_diff_25m, where=temp_diff_25m>=0, 
                  color='red', alpha=0.5, label='Warming')
ax2.axhline(y=-3, color='green', linestyle='--', linewidth=2, label='Target ΔT = -3°C')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
mean_diff = np.nanmean(temp_diff_25m)
ax2.axhline(y=mean_diff, color='orange', linestyle=':', linewidth=2, 
            label=f'Mean ΔT = {mean_diff:.2f}°C')
ax2.set_ylabel('ΔT (ON - OFF) (°C)')
ax2.set_title(f'(b) Temperature Difference at {HP_DEPTH}m - Heat Pump Effect')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-6, 2)

# Panel (c): Surface temperature comparison
ax3 = axes[2]
ax3.plot(dates_sub, surf_off_sub, 'b-', alpha=0.7, linewidth=0.8, label='HP OFF')
ax3.plot(dates_sub, surf_on_sub, 'r-', alpha=0.7, linewidth=0.8, label='HP ON')
ax3.set_ylabel('Temperature (°C)')
ax3.set_title('(c) Surface Temperature Comparison')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Panel (d): Surface temperature difference
ax4 = axes[3]
surf_diff = surf_on_sub - surf_off_sub
ax4.fill_between(dates_sub, 0, surf_diff, where=surf_diff<0, 
                  color='blue', alpha=0.5, label='Surface cooling')
ax4.fill_between(dates_sub, 0, surf_diff, where=surf_diff>=0, 
                  color='red', alpha=0.5, label='Surface warming')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
mean_surf_diff = np.nanmean(surf_diff)
ax4.axhline(y=mean_surf_diff, color='orange', linestyle=':', linewidth=2, 
            label=f'Mean ΔT = {mean_surf_diff:.2f}°C')
ax4.set_ylabel('ΔT (ON - OFF) (°C)')
ax4.set_title('(d) Surface Temperature Difference - Propagation to Surface')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-2, 1)

ax4.xaxis.set_major_locator(mdates.YearLocator(2))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax4.set_xlabel('Year')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig9_heat_pump_diagnostic.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/Fig9_heat_pump_diagnostic.png")

# Print statistics
print("\n" + "="*70)
print("HEAT PUMP DIAGNOSTIC STATISTICS")
print("="*70)
print(f"Temperature at {HP_DEPTH}m:")
print(f"  HP OFF - Mean: {np.nanmean(temp_off_25m):.2f}°C, Min: {np.nanmin(temp_off_25m):.2f}°C, Max: {np.nanmax(temp_off_25m):.2f}°C")
print(f"  HP ON  - Mean: {np.nanmean(temp_on_25m):.2f}°C, Min: {np.nanmin(temp_on_25m):.2f}°C, Max: {np.nanmax(temp_on_25m):.2f}°C")
print(f"  Difference (ON-OFF): Mean = {np.nanmean(temp_on_25m - temp_off_25m):.2f}°C")
print(f"\nSurface Temperature:")
print(f"  HP OFF - Mean: {np.nanmean(surf_temp_off):.2f}°C")
print(f"  HP ON  - Mean: {np.nanmean(surf_temp_on):.2f}°C")
print(f"  Difference (ON-OFF): Mean = {np.nanmean(surf_temp_on - surf_temp_off):.2f}°C")
print("="*70)

# Check for sub-zero temperatures
min_temp_on = np.nanmin(temp_on_25m)
if min_temp_on < 0:
    print(f"\n⚠️  WARNING: Minimum temperature at {HP_DEPTH}m is {min_temp_on:.2f}°C (below freezing!)")
    print("    Consider adding a minimum temperature constraint to the heat pump code.")
