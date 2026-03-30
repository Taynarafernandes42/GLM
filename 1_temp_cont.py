#!/usr/bin/env python3
"""
Temperature contour comparison - No interpolation above surface
Plots actual data with white areas above the varying lake water level
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Publication style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# File paths
off_file = "/home/taynara/Desktop/AED_Tools_GD_BACKUP_UWA_2709/GLM/output_off/output.nc"
on_file = "/home/taynara/Desktop/AED_Tools_GD_BACKUP_UWA_2709/GLM/output_on/output.nc"

print("Loading data...")

# Load OFF data
nc_off = Dataset(off_file, 'r')
time_off = nc_off.variables['time'][:]
temp_off = nc_off.variables['temp'][:, :, 0, 0]
z_off = nc_off.variables['z'][:, :, 0, 0]
time_units = nc_off.variables['time'].units
nc_off.close()

# Load ON data
nc_on = Dataset(on_file, 'r')
temp_on = nc_on.variables['temp'][:, :, 0, 0]
z_on = nc_on.variables['z'][:, :, 0, 0]
nc_on.close()

# Load lake level from CSV (more reliable than NetCDF lake_level variable)
import pandas as pd
csv_off = pd.read_csv('output_off/lake.csv')
csv_on = pd.read_csv('output_on/lake.csv')
surf_off = csv_off['Lake Level'].values
surf_on = csv_on['Lake Level'].values

print(f"Data shape: temp={temp_off.shape}, z={z_off.shape}")

# Parse time
base_date_str = time_units.replace('hours since ', '').split()[0]
base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
dates = np.array([base_date + timedelta(hours=float(h)) for h in time_off])

# Subsample for plotting
subsample = 24  # Daily
idx = np.arange(0, len(time_off), subsample)
dates_sub = dates[idx]
dates_num = mdates.date2num(dates_sub)

print(f"Time points after subsampling: {len(idx)}")

# Get the maximum number of layers from the data
n_layers = temp_off.shape[1]
print(f"Number of model layers: {n_layers}")

# Subsample data
temp_off_sub = temp_off[idx, :]
temp_on_sub = temp_on[idx, :]
z_off_sub = z_off[idx, :]
z_on_sub = z_on[idx, :]
# CSV data is already daily, so use directly (no subsampling needed)
surf_sub_off = surf_off  # Already at daily resolution
surf_sub_on = surf_on

# Handle masked arrays - convert to NaN where masked
def unmask_array(arr):
    """Convert masked array to regular array with NaN for masked values"""
    if hasattr(arr, 'mask'):
        return np.where(arr.mask, np.nan, arr)
    return np.array(arr)

temp_off_sub = unmask_array(temp_off_sub)
temp_on_sub = unmask_array(temp_on_sub)
z_off_sub = unmask_array(z_off_sub)
z_on_sub = unmask_array(z_on_sub)

# Create masked arrays - mask values where z is 0 or NaN (above water surface)
temp_off_ma = np.ma.masked_where((z_off_sub <= 0) | np.isnan(z_off_sub) | np.isnan(temp_off_sub), temp_off_sub)
temp_on_ma = np.ma.masked_where((z_on_sub <= 0) | np.isnan(z_on_sub) | np.isnan(temp_on_sub), temp_on_sub)

# For pcolormesh, we need to create a regular grid since z varies with time
# We'll interpolate to a common depth grid for visualization
max_plot_depth = 50
n_depths = 200
depth_grid = np.linspace(0, max_plot_depth, n_depths)

def interp_to_grid_no_extrap(temp_data, z_data, surf_data):
    """
    Interpolate to regular grid WITHOUT extrapolation above the surface.
    Uses the actual layer depths from the model.
    """
    nt = temp_data.shape[0]
    temp_grid = np.full((nt, len(depth_grid)), np.nan)
    
    for i in range(nt):
        z_t = z_data[i, :]
        temp_t = temp_data[i, :]
        lake_level = float(surf_data[i])
        
        # Find valid data (where z > 0 and not NaN)
        valid = (z_t > 0) & ~np.isnan(z_t) & ~np.isnan(temp_t)
        
        if np.sum(valid) > 2:
            z_valid = z_t[valid]
            temp_valid = temp_t[valid]
            
            # Sort by depth
            sort_idx = np.argsort(z_valid)
            z_valid = z_valid[sort_idx]
            temp_valid = temp_valid[sort_idx]
            
            # Extend to bottom (z=0) using lowest layer temperature
            if z_valid[0] > 0:
                z_valid = np.concatenate([[0.0], z_valid])
                temp_valid = np.concatenate([[temp_valid[0]], temp_valid])
            
            # Maximum depth with data
            max_z_data = np.max(z_valid)
            
            for j, d in enumerate(depth_grid):
                # Only interpolate where we have actual data (below lake surface)
                if d <= lake_level and d <= max_z_data:
                    temp_grid[i, j] = np.interp(d, z_valid, temp_valid)
    
    return temp_grid

print("Interpolating OFF data to common grid...")
temp_grid_off = interp_to_grid_no_extrap(temp_off_sub, z_off_sub, surf_sub_off)

print("Interpolating ON data to common grid...")
temp_grid_on = interp_to_grid_no_extrap(temp_on_sub, z_on_sub, surf_sub_on)

# Create masked arrays for plotting
temp_off_plot = np.ma.masked_invalid(temp_grid_off)
temp_on_plot = np.ma.masked_invalid(temp_grid_on)

# Create figure with 4 panels
fig, axes = plt.subplots(4, 1, figsize=(7.5, 12), facecolor='white')
plt.subplots_adjust(hspace=0.28, left=0.10, right=0.88, top=0.96, bottom=0.06)

# Colormaps with white for masked values
cmap_temp = plt.cm.RdYlBu_r.copy()
cmap_temp.set_bad(color='white', alpha=1.0)

cmap_diff = plt.cm.RdBu_r.copy()
cmap_diff.set_bad(color='white', alpha=1.0)

# Colormap for cooling only (blues)
cmap_cooling = plt.cm.Blues_r.copy()
cmap_cooling.set_bad(color='white', alpha=1.0)

# Temperature range from actual data
vmin = min(np.nanmin(temp_grid_off), np.nanmin(temp_grid_on))
vmax = max(np.nanmax(temp_grid_off), np.nanmax(temp_grid_on))
print(f"Temperature range: {vmin:.2f}°C to {vmax:.2f}°C")

# Heat pump depths (from glm4.nml configuration)
# Both extraction and injection at 25m elevation above lake bottom (154 m a.s.l.)
# base_elev = 129 m, so 25m elevation = ~25m above bottom when lake is full
HP_EXTRACTION_ELEV = 25.0  # m above bottom (subm_elev_outflow in nml)
HP_INJECTION_ELEV = 25.0   # m above bottom (subm_elev for heatpump inflow in nml)

# Mesh grid for pcolormesh
X, Y = np.meshgrid(dates_num, depth_grid)

print("Creating plots...")

# ============= Plot 1: Heat Pump OFF =============
ax1 = axes[0]
ax1.set_facecolor('white')

pcm1 = ax1.pcolormesh(X, Y, temp_off_plot.T, cmap=cmap_temp, vmin=vmin, vmax=vmax,
                       shading='nearest', rasterized=True)


ax1.set_ylabel('Elevation (m a.s.l.)')
ax1.set_ylim(0, max_plot_depth)
ax1.set_xlim(dates_num[0], dates_num[-1])
ax1.text(-0.06, 1.15, '(a)', transform=ax1.transAxes, fontsize=12,
         fontweight='bold', va='top', ha='left')
ax1.set_title('Reference simulation (Heat pump OFF)', fontsize=11, pad=8)

cbar1 = plt.colorbar(pcm1, ax=ax1, pad=0.02, aspect=20)
cbar1.set_label('Temperature (°C)')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.tick_params(labelbottom=False)

# ============= Plot 2: Heat Pump ON =============
ax2 = axes[1]
ax2.set_facecolor('white')

pcm2 = ax2.pcolormesh(X, Y, temp_on_plot.T, cmap=cmap_temp, vmin=vmin, vmax=vmax,
                       shading='nearest', rasterized=True)


ax2.set_ylabel('Elevation (m a.s.l.)')
ax2.set_ylim(0, max_plot_depth)
ax2.set_xlim(dates_num[0], dates_num[-1])
ax2.text(-0.06, 1.15, '(b)', transform=ax2.transAxes, fontsize=12,
         fontweight='bold', va='top', ha='left')
ax2.set_title('Heat pump ON simulation (ΔT = −1 °C)', fontsize=11, pad=8)

cbar2 = plt.colorbar(pcm2, ax=ax2, pad=0.02, aspect=20)
cbar2.set_label('Temperature (°C)')

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.tick_params(labelbottom=False)

# ============= Plot 3: Temperature Difference =============
ax3 = axes[2]
ax3.set_facecolor('white')

# Compute difference only where both have data
temp_diff = temp_on_plot - temp_off_plot

# Use only the region where BOTH simulations have valid data
combined_mask = temp_off_plot.mask | temp_on_plot.mask
temp_diff = np.ma.array(temp_diff, mask=combined_mask)

diff_max = np.ma.max(np.ma.abs(temp_diff))
print(f"Max temperature difference: {diff_max:.2f}°C")

# Use actual min/max from the data for symmetric colorbar
diff_min = np.ma.min(temp_diff)
diff_max_val = np.ma.max(temp_diff)
print(f"Temperature difference range: {diff_min:.2f}°C to {diff_max_val:.2f}°C")

# FIXED: Use clipped colorbar limits to show typical differences
# Based on distribution: 95% of data is within ±1.5°C, 99% within ±2°C
# Use ±2°C with extend='both' to show extreme values exist beyond range
diff_lim = 2.0  # Clipped symmetric limit
print(f"Using clipped colorbar: ±{diff_lim}°C (values beyond shown as saturated)")

pcm3 = ax3.pcolormesh(X, Y, temp_diff.T, cmap=cmap_diff, vmin=-diff_lim, vmax=diff_lim,
                       shading='nearest', rasterized=True)

ax3.set_ylabel('Elevation (m a.s.l.)')
ax3.set_xlabel('Year')
ax3.set_ylim(0, max_plot_depth)
ax3.set_xlim(dates_num[0], dates_num[-1])
ax3.text(-0.06, 1.15, '(c)', transform=ax3.transAxes, fontsize=12,
         fontweight='bold', va='top', ha='left')
ax3.set_title('Temperature difference (ΔT)', fontsize=11, pad=8)

# Add colorbar with extend='both' to show that values exist beyond the limits
cbar3 = plt.colorbar(pcm3, ax=ax3, pad=0.02, aspect=20, extend='both')
cbar3.set_label('ΔT (°C)')

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_major_locator(mdates.YearLocator(2))
ax3.tick_params(labelbottom=False)

# ============= Plot 4: Cooling Only (ΔT < 0) =============
ax4 = axes[3]
ax4.set_facecolor('white')

# Create cooling-only data: mask where ΔT >= 0
temp_cooling = temp_diff.copy()
temp_cooling = np.ma.masked_where(temp_diff >= 0, temp_cooling)

# Get cooling statistics
cooling_min = np.ma.min(temp_cooling)
cooling_count = np.ma.count(temp_cooling)
total_count = np.ma.count(temp_diff)
print(f"Cooling cells: {cooling_count} ({100*cooling_count/total_count:.1f}%)")
print(f"Max cooling: {cooling_min:.2f}°C")

# Use clipped colorbar for cooling (0 to -2°C, with extend for extremes)
pcm4 = ax4.pcolormesh(X, Y, temp_cooling.T, cmap=cmap_cooling, vmin=-2.0, vmax=0,
                       shading='nearest', rasterized=True)

ax4.set_ylabel('Elevation (m a.s.l.)')
ax4.set_xlabel('Year')
ax4.set_ylim(0, max_plot_depth)
ax4.set_xlim(dates_num[0], dates_num[-1])
ax4.text(-0.06, 1.15, '(d)', transform=ax4.transAxes, fontsize=12,
         fontweight='bold', va='top', ha='left')
ax4.set_title('Cooling effect only (ΔT < 0)', fontsize=11, pad=8)

# Colorbar with extend='min' to show extreme cooling beyond -2°C
cbar4 = plt.colorbar(pcm4, ax=ax4, pad=0.02, aspect=20, extend='min')
cbar4.set_label('ΔT (°C)')

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax4.xaxis.set_major_locator(mdates.YearLocator(2))

# Save figure
outfile = '/home/taynara/Desktop/AED_Tools_GD_BACKUP_UWA_2709/GLM/Fig_temperature_contour.png'
plt.savefig(outfile, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
print(f"Saved PNG to {outfile}")

plt.close()

# ============= Print Statistics =============
print("\n" + "="*60)
print("TEMPERATURE STATISTICS SUMMARY")
print("="*60)

# Reference simulation (Heat Pump OFF)
print("\n--- Reference Simulation (Heat Pump OFF) ---")
print(f"  Mean temperature:    {np.nanmean(temp_grid_off):.2f} °C")
print(f"  Min temperature:     {np.nanmin(temp_grid_off):.2f} °C")
print(f"  Max temperature:     {np.nanmax(temp_grid_off):.2f} °C")
print(f"  Std deviation:       {np.nanstd(temp_grid_off):.2f} °C")

# Heat pump simulation (Heat Pump ON)
print("\n--- Heat Pump Simulation (Heat Pump ON) ---")
print(f"  Mean temperature:    {np.nanmean(temp_grid_on):.2f} °C")
print(f"  Min temperature:     {np.nanmin(temp_grid_on):.2f} °C")
print(f"  Max temperature:     {np.nanmax(temp_grid_on):.2f} °C")
print(f"  Std deviation:       {np.nanstd(temp_grid_on):.2f} °C")

# Temperature difference statistics
temp_diff_data = temp_grid_on - temp_grid_off
valid_diff = ~np.isnan(temp_diff_data)

print("\n--- Temperature Difference (ON - OFF) ---")
print(f"  Mean ΔT:             {np.nanmean(temp_diff_data):.3f} °C")
print(f"  Min ΔT:              {np.nanmin(temp_diff_data):.3f} °C")
print(f"  Max ΔT:              {np.nanmax(temp_diff_data):.3f} °C")
print(f"  Std deviation:       {np.nanstd(temp_diff_data):.3f} °C")

# Cooling/warming breakdown
cooling_mask = temp_diff_data < 0
warming_mask = temp_diff_data > 0
print(f"\n  Cells with cooling:  {np.sum(cooling_mask & valid_diff)} ({100*np.sum(cooling_mask & valid_diff)/np.sum(valid_diff):.1f}%)")
print(f"  Cells with warming:  {np.sum(warming_mask & valid_diff)} ({100*np.sum(warming_mask & valid_diff)/np.sum(valid_diff):.1f}%)")
print(f"  Mean cooling:        {np.nanmean(temp_diff_data[cooling_mask]):.3f} °C")
print(f"  Mean warming:        {np.nanmean(temp_diff_data[warming_mask]):.3f} °C")

# Simulation period info
print("\n--- Simulation Period ---")
print(f"  Start date:          {dates_sub[0].strftime('%Y-%m-%d')}")
print(f"  End date:            {dates_sub[-1].strftime('%Y-%m-%d')}")
print(f"  Duration:            {(dates_sub[-1] - dates_sub[0]).days / 365.25:.1f} years")
print(f"  Time steps plotted:  {len(idx)}")

# Lake level statistics
print("\n--- Lake Level Statistics ---")
print(f"  OFF - Mean level:    {np.mean(surf_sub_off):.2f} m")
print(f"  OFF - Min level:     {np.min(surf_sub_off):.2f} m")
print(f"  OFF - Max level:     {np.max(surf_sub_off):.2f} m")
print(f"  ON  - Mean level:    {np.mean(surf_sub_on):.2f} m")
print(f"  ON  - Min level:     {np.min(surf_sub_on):.2f} m")
print(f"  ON  - Max level:     {np.max(surf_sub_on):.2f} m")
level_diff = surf_sub_on - surf_sub_off
print(f"  Mean level diff:     {np.mean(level_diff):.4f} m")

print("\n" + "="*60)
print("Done!")
