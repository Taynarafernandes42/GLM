#!/usr/bin/env python3
"""
Temperature Difference Contour - Shows exactly where heat pump affects the lake
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import os

NC_OFF = 'output_off/output.nc'
NC_ON = 'output_on/output.nc'
OUTPUT_DIR = os.path.expanduser('~/Desktop/GLM_Analysis_Plots_3K')
HP_DEPTH = 25.0

def nc_time_to_datetime(nc_time):
    return [datetime(1,1,1) + timedelta(days=float(t)-1) for t in nc_time]

def interpolate_to_depth_grid(nc_file, depth_grid, time_subsample=24):
    """Interpolate temperature to regular depth grid"""
    with Dataset(nc_file, 'r') as nc:
        time = nc.variables['time'][::time_subsample]
        temp_raw = nc.variables['temp'][::time_subsample, :]
        z_raw = nc.variables['z'][::time_subsample, :]
        
        nt = len(time)
        nd = len(depth_grid)
        temp_grid = np.full((nt, nd), np.nan)
        
        for t in range(nt):
            z_t = z_raw[t, :]
            temp_t = temp_raw[t, :]
            valid = ~np.isnan(z_t) & ~np.isnan(temp_t)
            if np.sum(valid) > 2:
                z_v = z_t[valid]
                temp_v = temp_t[valid]
                # Sort by depth
                idx = np.argsort(z_v)
                z_v = z_v[idx]
                temp_v = temp_v[idx]
                # Interpolate
                for d, depth in enumerate(depth_grid):
                    if depth >= z_v.min() and depth <= z_v.max():
                        temp_grid[t, d] = np.interp(depth, z_v, temp_v)
        
        dates = nc_time_to_datetime(time)
    return dates, temp_grid

print("Loading and interpolating data...")
depth_grid = np.linspace(0, 50, 100)
dates, temp_off = interpolate_to_depth_grid(NC_OFF, depth_grid)
_, temp_on = interpolate_to_depth_grid(NC_ON, depth_grid)

# Calculate difference
temp_diff = temp_on - temp_off

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Create meshgrid for contour plots
dates_num = mdates.date2num(dates)
D, T = np.meshgrid(depth_grid, dates_num)

# Panel (a): HP OFF temperature
ax1 = axes[0]
c1 = ax1.contourf(T, D, temp_off, levels=np.linspace(0, 25, 26), cmap='RdYlBu_r', extend='both')
ax1.axhline(y=HP_DEPTH, color='black', linestyle='--', linewidth=1.5, label=f'HP depth ({HP_DEPTH}m)')
ax1.set_ylabel('Depth (m above bottom)')
ax1.set_title('(a) Temperature - Heat Pump OFF (Baseline)')
ax1.legend(loc='upper right')
plt.colorbar(c1, ax=ax1, label='Temperature (°C)')

# Panel (b): HP ON temperature
ax2 = axes[1]
c2 = ax2.contourf(T, D, temp_on, levels=np.linspace(0, 25, 26), cmap='RdYlBu_r', extend='both')
ax2.axhline(y=HP_DEPTH, color='black', linestyle='--', linewidth=1.5, label=f'HP depth ({HP_DEPTH}m)')
ax2.set_ylabel('Depth (m above bottom)')
ax2.set_title('(b) Temperature - Heat Pump ON')
ax2.legend(loc='upper right')
plt.colorbar(c2, ax=ax2, label='Temperature (°C)')

# Panel (c): Temperature difference - THE KEY PLOT
ax3 = axes[2]
# Use diverging colormap centered at 0
levels_diff = np.linspace(-5, 2, 29)
c3 = ax3.contourf(T, D, temp_diff, levels=levels_diff, cmap='RdBu', extend='both')
ax3.axhline(y=HP_DEPTH, color='black', linestyle='--', linewidth=2, label=f'HP depth ({HP_DEPTH}m)')
# Add contour lines at key values
cs = ax3.contour(T, D, temp_diff, levels=[-3, -2, -1, 0], colors='black', linewidths=0.5)
ax3.clabel(cs, inline=True, fontsize=8, fmt='%.0f°C')
ax3.set_ylabel('Depth (m above bottom)')
ax3.set_title('(c) Temperature Difference (ON - OFF) - Heat Pump Cooling Effect')
ax3.legend(loc='upper right')
cbar3 = plt.colorbar(c3, ax=ax3, label='ΔT (°C)')

# Format x-axis
ax3.xaxis.set_major_locator(mdates.YearLocator(2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.set_xlabel('Year')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig10_temperature_diff_contour.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/Fig10_temperature_diff_contour.png")

# Statistics by depth zone
print("\n" + "="*70)
print("TEMPERATURE CHANGE BY DEPTH ZONE")
print("="*70)
for zone in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]:
    mask = (depth_grid >= zone[0]) & (depth_grid < zone[1])
    mean_diff = np.nanmean(temp_diff[:, mask])
    print(f"  {zone[0]:2d}-{zone[1]:2d}m: Mean ΔT = {mean_diff:+.2f}°C")

# At HP depth specifically
hp_idx = np.argmin(np.abs(depth_grid - HP_DEPTH))
hp_diff = temp_diff[:, hp_idx]
print(f"\n  At HP depth ({HP_DEPTH}m): Mean ΔT = {np.nanmean(hp_diff):+.2f}°C")
print(f"                         Min ΔT = {np.nanmin(hp_diff):+.2f}°C")
print(f"                         Max ΔT = {np.nanmax(hp_diff):+.2f}°C")
print("="*70)
