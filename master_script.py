#!/usr/bin/env python3
"""
MASTER ANALYSIS SCRIPT - Heat Pump Impact on Lake
Generates all publication-quality figures for:
  - Temperature contour analysis
  - Temperature time series and profiles
  - Water balance analysis (level, volume, evaporation)
  
Usage:
    python master_script.py [--all] [--temp-contour] [--temp-analysis] [--water-balance]
    
If no arguments provided, generates all figures.
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
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
NC_OFF = 'output_off/output.nc'
NC_ON = 'output_on/output.nc'
CSV_OFF = 'output_off/lake.csv'
CSV_ON = 'output_on/lake.csv'

OUTPUT_DIR = os.path.expanduser('~/Desktop/GLM_Analysis_Plots_3K')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data processing parameters
TIME_SUBSAMPLE = 24  # Daily from hourly
DEPTH_GRID = np.linspace(0, 50, 100)

# Heat pump parameters
HP_FLOW_RATE = 25920  # m³/day (0.3 m³/s)
HP_TEMP_CHANGE = -3.0  # °C (cooling mode)

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

# ============================================================================
# GMD PUBLICATION-QUALITY PLOT SETTINGS
# ============================================================================
# Color palette - colorblind-friendly, GMD style
COLORS = {
    'hp_off': '#B2182B',      # Red (warmer)
    'hp_on': '#2166AC',       # Blue (cold)  
    'hp_off_fill': '#D6604D', # Light red for HP OFF fill
    'hp_on_fill': '#4393C3',  # Light blue for HP ON fill
    'diff': '#333333',        # Dark gray for difference lines
    'diff_pos': '#D6604D',    # Light red for positive diff
    'diff_neg': '#4393C3',    # Light blue for negative diff
    'mean_line': '#E69F00',   # Orange for mean lines
    'zero_line': '#666666',   # Gray for reference lines
    'fill_pos': '#FDDBC7',    # Very light red fill
    'fill_neg': '#D1E5F0',    # Very light blue fill
    'extract': '#2CA02C',     # Green for extraction depth marker
}

plt.rcParams.update({
    # Font settings - GMD prefers sans-serif
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    
    # Axes
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    
    # Grid
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',
    
    # Ticks
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # Legend
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.8',
    'legend.fancybox': False,
    
    # Lines
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    
    # Figure
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    
    # Saving
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

def add_stats_annotation(ax, data, position='upper right', prefix='', show_n=False):
    """
    Add statistics annotation box to plot without overlapping data.
    
    Parameters:
    -----------
    ax : matplotlib axes
    data : array-like, the data to calculate statistics from
    position : str, one of 'upper right', 'upper left', 'lower right', 'lower left'
    prefix : str, prefix for labels (e.g., 'ΔT')
    show_n : bool, whether to show sample size
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    mean_val = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Build stats text
    if prefix:
        lines = [
            f'{prefix} mean: {mean_val:+.2f}',
            f'{prefix} min: {min_val:+.2f}',
            f'{prefix} max: {max_val:+.2f}',
        ]
    else:
        lines = [
            f'Mean: {mean_val:.2f}',
            f'Min: {min_val:.2f}',
            f'Max: {max_val:.2f}',
        ]
    
    if show_n:
        lines.append(f'n = {len(data)}')
    
    stats_text = '\n'.join(lines)
    
    # Position mapping
    pos_map = {
        'upper right': (0.98, 0.98, 'right', 'top'),
        'upper left': (0.02, 0.98, 'left', 'top'),
        'lower right': (0.98, 0.02, 'right', 'bottom'),
        'lower left': (0.02, 0.02, 'left', 'bottom'),
    }
    
    x, y, ha, va = pos_map.get(position, pos_map['upper right'])
    
    bbox_props = dict(
        boxstyle='round,pad=0.4',
        facecolor='white',
        edgecolor='0.7',
        alpha=0.95,
        linewidth=0.5
    )
    
    ax.text(x, y, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment=va, horizontalalignment=ha,
            bbox=bbox_props, family='monospace')

def style_axis(ax, xlabel=None, ylabel=None, title=None):
    """Apply consistent styling to an axis."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=8)
    ax.tick_params(axis='both', which='major', labelsize=10)

def find_best_stats_position(ax, dates, data1, data2=None, legend_loc='upper right'):
    """
    Find optimal position for stats box based on data distribution.
    Avoids overlap with data and legend.
    
    Returns: (x, y, ha, va) for ax.text positioning
    """
    # Calculate data range in normalized coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    # Check where data is concentrated
    if data2 is not None:
        all_data = np.concatenate([data1, data2])
    else:
        all_data = np.array(data1)
    all_data = all_data[~np.isnan(all_data)]
    
    data_mean = np.nanmean(all_data)
    data_max = np.nanmax(all_data)
    data_min = np.nanmin(all_data)
    
    # Normalized position of data center
    norm_center = (data_mean - ylim[0]) / y_range
    
    # Avoid legend position
    legend_corners = {
        'upper right': ('lower left', 0.02, 0.02, 'left', 'bottom'),
        'upper left': ('lower right', 0.98, 0.02, 'right', 'bottom'),
        'lower right': ('upper left', 0.02, 0.98, 'left', 'top'),
        'lower left': ('upper right', 0.98, 0.98, 'right', 'top'),
    }
    
    # If data is mostly in upper half, place stats in lower corners
    if norm_center > 0.5:
        if legend_loc in ['lower right', 'lower left']:
            # Legend is low, data is high - use opposite low corner
            if legend_loc == 'lower right':
                return 0.02, 0.02, 'left', 'bottom'
            else:
                return 0.98, 0.02, 'right', 'bottom'
        else:
            # Legend is high, data is high - place stats low right
            return 0.98, 0.02, 'right', 'bottom'
    else:
        if legend_loc in ['upper right', 'upper left']:
            # Legend is high, data is low - use opposite high corner
            if legend_loc == 'upper right':
                return 0.02, 0.98, 'left', 'top'
            else:
                return 0.98, 0.98, 'right', 'top'
        else:
            # Legend is low, data is low - place stats high right
            return 0.98, 0.98, 'right', 'top'

def create_stats_text(data, prefix='', unit='', compact=False):
    """
    Create formatted statistics text.
    
    Parameters:
    -----------
    data : array-like
    prefix : str, e.g., 'ΔT'
    unit : str, e.g., '°C'
    compact : bool, if True use single-line format
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    mean_val = np.nanmean(data)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    if compact:
        if prefix:
            return f'{prefix}: mean={mean_val:+.2f}, min={min_val:+.2f}, max={max_val:+.2f}{unit}'
        return f'mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}{unit}'
    else:
        if prefix:
            return f'{prefix}  mean: {mean_val:+.2f}{unit}\n     min: {min_val:+.2f}{unit}\n     max: {max_val:+.2f}{unit}'
        return f'mean: {mean_val:.2f}{unit}\n min: {min_val:.2f}{unit}\n max: {max_val:.2f}{unit}'

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

def unmask_array(arr):
    """Convert masked array to regular array with NaN for masked values"""
    if hasattr(arr, 'mask'):
        return np.where(arr.mask, np.nan, arr)
    return np.array(arr)

def interp_to_grid_no_extrap(temp_data, z_data, surf_data, depth_grid):
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

# ============================================================================
# FIGURE 1: TEMPERATURE CONTOUR PLOTS (2 panels - OFF and ON)
# ============================================================================
def plot_temperature_contour():
    """Generate temperature contour comparison plot (Figure 1)"""
    print("\n" + "="*70)
    print("GENERATING: Temperature Contour Plots")
    print("="*70)
    
    # Load NetCDF data
    nc_off = Dataset(NC_OFF, 'r')
    time_off = nc_off.variables['time'][:]
    temp_off = nc_off.variables['temp'][:, :, 0, 0]
    z_off = nc_off.variables['z'][:, :, 0, 0]
    time_units = nc_off.variables['time'].units
    nc_off.close()

    nc_on = Dataset(NC_ON, 'r')
    temp_on = nc_on.variables['temp'][:, :, 0, 0]
    z_on = nc_on.variables['z'][:, :, 0, 0]
    nc_on.close()

    # Load lake level from CSV
    csv_off = pd.read_csv(CSV_OFF)
    csv_on = pd.read_csv(CSV_ON)
    surf_off = csv_off['Lake Level'].values
    surf_on = csv_on['Lake Level'].values

    # Parse time
    base_date_str = time_units.replace('hours since ', '').split()[0]
    base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
    dates = np.array([base_date + timedelta(hours=float(h)) for h in time_off])

    # Subsample for plotting - use minimum length to avoid index errors
    subsample = 24  # Daily
    min_len = min(len(time_off), temp_on.shape[0])
    idx = np.arange(0, min_len, subsample)
    dates_sub = dates[idx]
    dates_num = mdates.date2num(dates_sub)

    # Subsample data
    temp_off_sub = unmask_array(temp_off[idx, :])
    temp_on_sub = unmask_array(temp_on[idx, :])
    z_off_sub = unmask_array(z_off[idx, :])
    z_on_sub = unmask_array(z_on[idx, :])

    # Create depth grid for interpolation
    max_plot_depth = 50
    n_depths = 200
    depth_grid = np.linspace(0, max_plot_depth, n_depths)
    
    # Interpolate to common grid
    print("  Interpolating OFF data...")
    temp_grid_off = interp_to_grid_no_extrap(temp_off_sub, z_off_sub, surf_off, depth_grid)
    print("  Interpolating ON data...")
    temp_grid_on = interp_to_grid_no_extrap(temp_on_sub, z_on_sub, surf_on, depth_grid)

    # Create masked arrays for plotting
    temp_off_plot = np.ma.masked_invalid(temp_grid_off)
    temp_on_plot = np.ma.masked_invalid(temp_grid_on)

    # Create figure with 2 panels
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), facecolor='white')
    plt.subplots_adjust(hspace=0.25, left=0.08, right=0.92, top=0.94, bottom=0.08)

    # Colormaps
    cmap_temp = plt.cm.RdYlBu_r.copy()
    cmap_temp.set_bad(color='white', alpha=1.0)

    # Temperature range
    vmin = min(np.nanmin(temp_grid_off), np.nanmin(temp_grid_on))
    vmax = max(np.nanmax(temp_grid_off), np.nanmax(temp_grid_on))

    # Mesh grid
    X, Y = np.meshgrid(dates_num, depth_grid)

    # Panel (a): Heat Pump OFF
    ax1 = axes[0]
    ax1.set_facecolor('white')
    pcm1 = ax1.pcolormesh(X, Y, temp_off_plot.T, cmap=cmap_temp, vmin=vmin, vmax=vmax,
                           shading='nearest', rasterized=True)
    ax1.set_ylabel('Elevation (m a.s.l.)')
    ax1.set_ylim(0, max_plot_depth)
    ax1.set_xlim(dates_num[0], dates_num[-1])
    ax1.text(-0.06, 1.2, '(a)', transform=ax1.transAxes, fontsize=14,
             fontweight='bold', va='top', ha='left')
    ax1.set_title('Reference simulation (Heat pump OFF)', fontsize=12, pad=8)
    cbar1 = plt.colorbar(pcm1, ax=ax1, pad=0.02, aspect=20)
    cbar1.set_label('Temperature (°C)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.tick_params(labelbottom=False)

    # Panel (b): Heat Pump ON
    ax2 = axes[1]
    ax2.set_facecolor('white')
    pcm2 = ax2.pcolormesh(X, Y, temp_on_plot.T, cmap=cmap_temp, vmin=vmin, vmax=vmax,
                           shading='nearest', rasterized=True)
    ax2.set_ylabel('Elevation (m a.s.l.)')
    ax2.set_xlabel('Year')
    ax2.set_ylim(0, max_plot_depth)
    ax2.set_xlim(dates_num[0], dates_num[-1])
    ax2.text(-0.06, 1.2, '(b)', transform=ax2.transAxes, fontsize=14,
             fontweight='bold', va='top', ha='left')
    ax2.set_title(f'Heat pump ON simulation (ΔT = {HP_TEMP_CHANGE} °C)', fontsize=12, pad=8)
    cbar2 = plt.colorbar(pcm2, ax=ax2, pad=0.02, aspect=20)
    cbar2.set_label('Temperature (°C)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    outfile = f'{OUTPUT_DIR}/Fig1_temperature_contour.png'
    plt.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# FIGURE 2: LAKE LEVEL AND VOLUME (4-panel)
# ============================================================================
def plot_level_volume():
    """Generate lake level and volume comparison (Figure 2)"""
    print("\n" + "="*70)
    print("GENERATING: Lake Level and Volume")
    print("="*70)
    
    df_off = load_lake_csv(CSV_OFF)
    df_on = load_lake_csv(CSV_ON)
    
    # Align data
    df_off = df_off.set_index('datetime')
    df_on = df_on.set_index('datetime')
    common_start = max(df_off.index.min(), df_on.index.min())
    common_end = min(df_off.index.max(), df_on.index.max())
    df_off = df_off.loc[common_start:common_end].reset_index()
    df_on = df_on.loc[common_start:common_end].reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.28, left=0.08, right=0.95, top=0.94, bottom=0.08)
    
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='0.7', alpha=0.95, linewidth=0.5)

    # (a) Lake Level
    ax = axes[0, 0]
    ax.plot(df_off['datetime'], df_off['Lake Level'], color=COLORS['hp_off'], 
            lw=1.5, label='HP OFF', alpha=0.9, zorder=2)
    ax.plot(df_on['datetime'], df_on['Lake Level'], color=COLORS['hp_on'], 
            lw=1.5, label='HP ON', alpha=0.9, zorder=2)
    style_axis(ax, ylabel='Lake Level (m)', title='(a) Lake Water Level')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, edgecolor='0.8')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Level Difference  
    ax = axes[0, 1]
    level_diff = df_on['Lake Level'].values - df_off['Lake Level'].values
    ax.plot(df_off['datetime'], level_diff * 1000, color=COLORS['diff'], lw=1.0, alpha=0.9, zorder=2)
    ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
    ax.axhline(np.mean(level_diff)*1000, color=COLORS['mean_line'], ls='-', lw=2.0, alpha=0.8, zorder=3)
    style_axis(ax, ylabel='Level Difference (mm)', title='(b) Water Level Difference (ON − OFF)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    stats_text = f'mean: {np.mean(level_diff)*1000:+.2f} mm\n std: {np.std(level_diff)*1000:.2f} mm'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            va='bottom', ha='right', bbox=bbox_props, family='monospace')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Volume
    ax = axes[1, 0]
    ax.plot(df_off['datetime'], df_off['Volume']/1e6, color=COLORS['hp_off'], 
            lw=1.5, label='HP OFF', alpha=0.9, zorder=2)
    ax.plot(df_on['datetime'], df_on['Volume']/1e6, color=COLORS['hp_on'], 
            lw=1.5, label='HP ON', alpha=0.9, zorder=2)
    style_axis(ax, ylabel='Lake Volume (×10⁶ m³)', xlabel='Year', title='(c) Lake Volume')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, edgecolor='0.8')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) Volume Difference
    ax = axes[1, 1]
    vol_diff = df_on['Volume'].values - df_off['Volume'].values
    ax.plot(df_off['datetime'], vol_diff, color=COLORS['diff'], lw=1.0, alpha=0.9, zorder=2)
    ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
    ax.axhline(np.mean(vol_diff), color=COLORS['mean_line'], ls='-', lw=2.0, alpha=0.8, zorder=3)
    style_axis(ax, ylabel='Volume Difference (m³)', xlabel='Year', title='(d) Volume Difference (ON − OFF)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    stats_text = f'mean: {np.mean(vol_diff):+.0f} m³\n std: {np.std(vol_diff):.0f} m³'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            va='bottom', ha='right', bbox=bbox_props, family='monospace')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    outfile = f'{OUTPUT_DIR}/Fig2_level_volume.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# FIGURE 3: EVAPORATION ANALYSIS (3-panel)
# ============================================================================
def plot_evaporation():
    """Generate evaporation analysis plot (Figure 3)"""
    print("\n" + "="*70)
    print("GENERATING: Evaporation Analysis")
    print("="*70)
    
    df_off = load_lake_csv(CSV_OFF)
    df_on = load_lake_csv(CSV_ON)
    
    # Align data
    df_off = df_off.set_index('datetime')
    df_on = df_on.set_index('datetime')
    common_start = max(df_off.index.min(), df_on.index.min())
    common_end = min(df_off.index.max(), df_on.index.max())
    df_off = df_off.loc[common_start:common_end].reset_index()
    df_on = df_on.loc[common_start:common_end].reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(hspace=0.25, left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='0.7', alpha=0.95, linewidth=0.5)

    # (a) Daily Evaporation
    ax = axes[0]
    ax.plot(df_off['datetime'], -df_off['Evaporation'], color=COLORS['hp_off'], 
            lw=1.0, alpha=0.85, label='HP OFF', zorder=2)
    ax.plot(df_on['datetime'], -df_on['Evaporation'], color=COLORS['hp_on'], 
            lw=1.0, alpha=0.85, label='HP ON', zorder=2)
    style_axis(ax, ylabel='Evaporation (m³/day)', title='(a) Daily Evaporation')
    ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.95, edgecolor='0.8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Evaporation Difference
    ax = axes[1]
    evap_diff = (-df_on['Evaporation'].values) - (-df_off['Evaporation'].values)
    ax.plot(df_off['datetime'], evap_diff, color=COLORS['diff'], lw=0.8, alpha=0.9, zorder=2)
    ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
    mean_evap = np.mean(evap_diff)
    ax.axhline(mean_evap, color=COLORS['mean_line'], ls='-', lw=2.0, alpha=0.8, zorder=3)
    ax.fill_between(df_off['datetime'], 0, evap_diff, 
                    where=(evap_diff > 0), alpha=0.15, color=COLORS['hp_off_fill'], zorder=1)
    ax.fill_between(df_off['datetime'], 0, evap_diff, 
                    where=(evap_diff < 0), alpha=0.15, color=COLORS['hp_on_fill'], zorder=1)
    style_axis(ax, ylabel='ΔEvaporation (m³/day)', title='(b) Evaporation Difference (ON − OFF)')
    # Add mean label as text annotation instead of legend to avoid overlap
    ax.text(0.01, 0.05, f'Mean: {mean_evap:+.1f} m³/day', transform=ax.transAxes, fontsize=9,
            va='bottom', ha='left', bbox=bbox_props, color=COLORS['mean_line'], fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Cumulative Evaporation Difference
    ax = axes[2]
    cumsum_evap_diff = np.cumsum(evap_diff)
    ax.plot(df_off['datetime'], cumsum_evap_diff/1e3, color=COLORS['diff'], lw=1.5, zorder=2)
    ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
    ax.fill_between(df_off['datetime'], 0, cumsum_evap_diff/1e3, 
                    alpha=0.15, color=COLORS['hp_off_fill'], zorder=1)
    style_axis(ax, ylabel='Cumulative ΔEvap (×10³ m³)', xlabel='Year', 
               title='(c) Cumulative Evaporation Difference')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    # Position stats in upper left since cumulative line trends upward to right
    ax.text(0.01, 0.05, f'Total: {cumsum_evap_diff[-1]/1e3:+.1f} ×10³ m³', 
            transform=ax.transAxes, fontsize=10, va='bottom', ha='left', bbox=bbox_props)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    outfile = f'{OUTPUT_DIR}/Fig3_evaporation.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# FIGURE 4: VOLUME vs EVAPORATION/OVERFLOW (4-panel)
# ============================================================================
def plot_volume_vs_evaporation():
    """Generate volume vs evaporation/overflow comparison (Figure 4)"""
    print("\n" + "="*70)
    print("GENERATING: Volume vs Evaporation Analysis")
    print("="*70)
    
    df_off = load_lake_csv(CSV_OFF)
    df_on = load_lake_csv(CSV_ON)
    
    # Align data
    df_off = df_off.set_index('datetime')
    df_on = df_on.set_index('datetime')
    common_start = max(df_off.index.min(), df_on.index.min())
    common_end = min(df_off.index.max(), df_on.index.max())
    df_off = df_off.loc[common_start:common_end].reset_index()
    df_on = df_on.loc[common_start:common_end].reset_index()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.28, left=0.08, right=0.95, top=0.96, bottom=0.07)
    
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='0.7', alpha=0.95, linewidth=0.5)

    # (a) Volume Difference
    ax = axes[0]
    vol_diff = df_on['Volume'].values - df_off['Volume'].values
    ax.plot(df_off['datetime'], vol_diff, color='#7B3294', lw=0.8, alpha=0.9, zorder=2)
    ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
    ax.axhline(np.mean(vol_diff), color=COLORS['mean_line'], ls='-', lw=2.0, alpha=0.8, zorder=3)
    style_axis(ax, ylabel='Volume Difference (m³)', title='(a) Lake Volume Difference (ON − OFF)')
    ax.text(0.01, 0.98, f'mean: {np.mean(vol_diff):+.0f} m³', transform=ax.transAxes,
            fontsize=9, va='top', ha='left', bbox=bbox_props, family='monospace')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Cumulative Evaporation Difference
    ax = axes[1]
    evap_diff = df_on['Evaporation'].values - df_off['Evaporation'].values
    cumsum_evap_diff = np.cumsum(evap_diff)
    ax.plot(df_off['datetime'], cumsum_evap_diff, color='#008837', lw=1.0, alpha=0.9, zorder=2)
    ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
    ax.fill_between(df_off['datetime'], 0, cumsum_evap_diff, alpha=0.15, color='#A6DBA0', zorder=1)
    style_axis(ax, ylabel='Cumulative ΔEvaporation (m³)', 
               title='(b) Cumulative Evaporation Difference (ON − OFF)')
    ax.text(0.01, 0.95, f'Total: {cumsum_evap_diff[-1]:+.0f} m³\n(+ve = HP ON evaporated less)', 
            transform=ax.transAxes, fontsize=9, va='top', ha='left', bbox=bbox_props)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Cumulative Overflow Difference
    ax = axes[2]
    if 'Overflow Vol' in df_off.columns:
        overflow_diff = df_on['Overflow Vol'].values - df_off['Overflow Vol'].values
        cumsum_overflow_diff = np.cumsum(overflow_diff)
        ax.plot(df_off['datetime'], cumsum_overflow_diff, color='#E66101', lw=1.0, alpha=0.9, zorder=2)
        ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
        ax.fill_between(df_off['datetime'], 0, cumsum_overflow_diff, alpha=0.15, color='#FDB863', zorder=1)
        style_axis(ax, ylabel='Cumulative ΔOverflow (m³)', 
                   title='(c) Cumulative Overflow Difference (ON − OFF)')
        ax.text(0.01, 0.95, f'Total: {cumsum_overflow_diff[-1]:+.0f} m³\n(+ve = HP ON had more overflow)', 
                transform=ax.transAxes, fontsize=9, va='top', ha='left', bbox=bbox_props)
    else:
        ax.text(0.5, 0.5, 'Overflow Vol not available in output', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        cumsum_overflow_diff = np.zeros_like(vol_diff)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) Overlay comparison
    ax = axes[3]
    ax.plot(df_off['datetime'], vol_diff, color='#7B3294', lw=0.8, alpha=0.8, 
            label='Vol. Diff', zorder=2)
    ax.plot(df_off['datetime'], cumsum_evap_diff, color='#008837', lw=1.0, alpha=0.8, 
            label='Σ Evap. Diff', zorder=2)
    
    if 'Overflow Vol' in df_off.columns:
        overflow_diff = df_on['Overflow Vol'].values - df_off['Overflow Vol'].values
        cumsum_overflow_loss = -np.cumsum(overflow_diff)
        ax.plot(df_off['datetime'], cumsum_overflow_loss, color='#E66101', lw=1.0, alpha=0.8, 
                label='Σ Overflow Loss', zorder=2)
        combined_effect = cumsum_evap_diff + cumsum_overflow_loss
        ax.plot(df_off['datetime'], combined_effect, color=COLORS['hp_on'], ls='--', lw=1.5, 
                alpha=0.9, label='Net Effect', zorder=3)
        correlation_combined = np.corrcoef(vol_diff, combined_effect)[0, 1]
        correlation_evap = np.corrcoef(vol_diff, cumsum_evap_diff)[0, 1]
        stats_text = f'r(Vol,Evap)={correlation_evap:.2f}  r(Vol,Net)={correlation_combined:.2f}'
        ax.text(0.01, 0.8, stats_text, transform=ax.transAxes, fontsize=9, 
                va='top', ha='left', bbox=bbox_props, family='monospace')
    
    ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
    style_axis(ax, ylabel='Difference (m³)', xlabel='Year', 
               title='(d) Volume Difference vs Cumulative Water Balance Components')
    ax.legend(loc='upper left', fontsize=9, ncol=4, framealpha=0.95, edgecolor='0.8')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    outfile = f'{OUTPUT_DIR}/Fig4_volume_vs_evaporation.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# FIGURE 5: SURFACE/MAX/MIN TEMPERATURE COMPARISON (3-panel)
# ============================================================================
def plot_temperature_timeseries():
    """Generate temperature time series comparison (Figure 5)"""
    print("\n" + "="*70)
    print("GENERATING: Temperature Time Series")
    print("="*70)
    
    df_off = load_lake_csv(CSV_OFF)
    df_on = load_lake_csv(CSV_ON)
    
    # Merge dataframes on datetime to align data
    df_merged = pd.merge(df_off, df_on, on='datetime', suffixes=('_off', '_on'), how='inner')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(hspace=0.28, left=0.08, right=0.95, top=0.96, bottom=0.08)
    
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='0.7', alpha=0.95, linewidth=0.5)

    temp_vars = [
        ('Surface Temp', '(a) Surface Temperature', '°C'),
        ('Max Temp', '(b) Maximum Temperature', '°C'),
        ('Min Temp', '(c) Minimum Temperature', '°C'),
    ]

    for i, (var, title, unit) in enumerate(temp_vars):
        ax = axes[i]
        
        # Plot with GMD colors
        ax.plot(df_merged['datetime'], df_merged[f'{var}_off'], 
                color=COLORS['hp_off'], lw=1.2, alpha=0.9, label='HP OFF', zorder=2)
        ax.plot(df_merged['datetime'], df_merged[f'{var}_on'], 
                color=COLORS['hp_on'], lw=1.2, alpha=0.9, label='HP ON', zorder=2)
        
        # Styling
        style_axis(ax, ylabel=f'Temperature ({unit})', title=title)
        
        # Legend on first panel only - upper left to avoid summer peaks
        if i == 0:
            ax.legend(loc='lower left', fontsize=7, ncol=2, 
                     framealpha=0.95, edgecolor='0.8')
        
        # Statistics - position based on panel to avoid data overlap
        diff = df_merged[f'{var}_on'].values - df_merged[f'{var}_off'].values
        stats_text = f'ΔT  mean: {np.nanmean(diff):+.2f}°C | min: {np.nanmin(diff):+.2f}°C | max: {np.nanmax(diff):+.2f}°C'
        
        # Position stats at lower right for all panels (compact single line format)
        ax.text(0.01, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',
                bbox=bbox_props, family='monospace')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Year')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

    outfile = f'{OUTPUT_DIR}/Fig5_temperature_timeseries.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# FIGURE 6: SEASONAL TEMPERATURE PROFILES (4-panel)
# ============================================================================
def plot_seasonal_profiles():
    """Generate seasonal temperature profiles (Figure 6)"""
    print("\n" + "="*70)
    print("GENERATING: Seasonal Temperature Profiles")
    print("="*70)
    
    dates_off, depths, temp_off, _ = load_netcdf_temp(NC_OFF)
    dates_on, _, temp_on, _ = load_netcdf_temp(NC_ON)
    
    # Align data - use minimum length
    min_len = min(len(dates_off), len(dates_on))
    dates = dates_off[:min_len]
    temp_off = temp_off[:min_len, :]
    temp_on = temp_on[:min_len, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.30, left=0.10, right=0.95, top=0.95, bottom=0.08)

    seasons = {
        '(a) Winter (DJF)': [12, 1, 2],
        '(b) Spring (MAM)': [3, 4, 5],
        '(c) Summer (JJA)': [6, 7, 8],
        '(d) Autumn (SON)': [9, 10, 11]
    }

    for i, (season_name, months) in enumerate(seasons.items()):
        ax = axes[i//2, i%2]
        
        month_array = np.array([d.month for d in dates])
        season_idx = np.isin(month_array, months)
        
        temp_off_season = np.nanmean(temp_off[season_idx, :], axis=0)
        temp_on_season = np.nanmean(temp_on[season_idx, :], axis=0)
        temp_off_std = np.nanstd(temp_off[season_idx, :], axis=0)
        temp_on_std = np.nanstd(temp_on[season_idx, :], axis=0)
        
        # Plot with GMD colors - mean lines
        ax.plot(temp_off_season, depths, color=COLORS['hp_off'], lw=2.0, 
                label='HP OFF', zorder=3)
        ax.plot(temp_on_season, depths, color=COLORS['hp_on'], lw=2.0, 
                label='HP ON', zorder=3)
        
        # Uncertainty bands (±1σ) - plot behind main lines
        ax.fill_betweenx(depths, temp_off_season - temp_off_std, temp_off_season + temp_off_std, 
                         alpha=0.15, color=COLORS['hp_off_fill'], zorder=2)
        ax.fill_betweenx(depths, temp_on_season - temp_on_std, temp_on_season + temp_on_std,
                         alpha=0.15, color=COLORS['hp_on_fill'], zorder=2)
        
        # Show extraction/injection depth
        ax.axhline(HP_EXTRACTION_ELEV, color=COLORS['extract'], ls='--', lw=1.8, alpha=0.8, 
                   label=f'HP depth', zorder=4)
        
        # Styling
        style_axis(ax, xlabel='Temperature (°C)', ylabel='Elevation (m)', title=season_name)
        ax.set_ylim([0, 50])
        ax.set_xlim([0, 25])
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Calculate ΔT at extraction depth
        idx_extract = np.argmin(np.abs(depths - HP_EXTRACTION_ELEV))
        delta_t = temp_on_season[idx_extract] - temp_off_season[idx_extract]
        
        # Legend and stats positioning
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='0.7', alpha=0.95, linewidth=0.5)
        
        # Position stats at lower left where there's no data (profiles are at x>4°C)
        # This avoids overlap with the profiles which extend into upper right
        if i == 0:  # Winter - first panel gets full legend
            ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='0.8')
        
        # Put ΔT annotation at lower right for all panels, avoiding legend in panel (a)
        ax.text(0.98, 0.03, f'ΔT at {HP_EXTRACTION_ELEV:.0f}m: {delta_t:+.1f}°C', 
                transform=ax.transAxes, fontsize=9, va='bottom', ha='right', 
                bbox=bbox_props, family='monospace')

    outfile = f'{OUTPUT_DIR}/Fig6_seasonal_profiles.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# FIGURE 7: TEMPERATURE DIFFERENCE TIME SERIES (3-panel)
# ============================================================================
def plot_temperature_difference():
    """Generate temperature difference time series (Figure 7)"""
    print("\n" + "="*70)
    print("GENERATING: Temperature Difference Time Series")
    print("="*70)
    
    df_off = load_lake_csv(CSV_OFF)
    df_on = load_lake_csv(CSV_ON)
    
    # Align data
    df_off = df_off.set_index('datetime')
    df_on = df_on.set_index('datetime')
    common_start = max(df_off.index.min(), df_on.index.min())
    common_end = min(df_off.index.max(), df_on.index.max())
    df_off = df_off.loc[common_start:common_end].reset_index()
    df_on = df_on.loc[common_start:common_end].reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(hspace=0.28, left=0.08, right=0.95, top=0.96, bottom=0.08)
    
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='0.7', alpha=0.95, linewidth=0.5)

    temp_vars = [
        ('Surface Temp', '(a) Surface Temperature Difference'),
        ('Max Temp', '(b) Maximum Temperature Difference'),
        ('Min Temp', '(c) Minimum Temperature Difference'),
    ]

    for i, (var, title) in enumerate(temp_vars):
        ax = axes[i]
        diff = df_on[var].values - df_off[var].values
        
        # Plot difference line
        ax.plot(df_off['datetime'], diff, color=COLORS['diff'], lw=0.8, alpha=0.9, zorder=2)
        
        # Zero reference line
        ax.axhline(0, color=COLORS['zero_line'], ls='--', lw=1.0, alpha=0.5, zorder=1)
        
        # Mean line  
        mean_diff = np.nanmean(diff)
        ax.axhline(mean_diff, color=COLORS['mean_line'], ls='-', lw=2.0, alpha=0.8, zorder=3)
        
        # Fill areas (cooling in blue, warming in red)
        ax.fill_between(df_off['datetime'], 0, diff, 
                        where=(diff > 0), alpha=0.20, color=COLORS['hp_off_fill'], zorder=1)
        ax.fill_between(df_off['datetime'], 0, diff, 
                        where=(diff < 0), alpha=0.20, color=COLORS['hp_on_fill'], zorder=1)
        
        # Styling
        style_axis(ax, ylabel='ΔTemperature (°C)', title=f'{title} (ON − OFF)')
        
        # Statistics box - compact single line at bottom right
        min_diff = np.nanmin(diff)
        max_diff = np.nanmax(diff)
        stats_text = f'mean: {mean_diff:+.2f}°C | min: {min_diff:+.2f}°C | max: {max_diff:+.2f}°C'
        # Position at lower right, with higher y on bottom panel to avoid x-axis label
        y_pos = 0.06 if i == 2 else 0.03
        ax.text(0.01, y_pos, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=bbox_props, family='monospace')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Year')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

    outfile = f'{OUTPUT_DIR}/Fig7_temperature_difference.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# FIGURE 8: TEMPERATURE AT KEY DEPTHS (4-panel)
# ============================================================================
def plot_temperature_depths():
    """Generate temperature at key depths (Figure 8)"""
    print("\n" + "="*70)
    print("GENERATING: Temperature at Key Depths")
    print("="*70)
    
    dates_off, depths, temp_off, _ = load_netcdf_temp(NC_OFF)
    dates_on, _, temp_on, _ = load_netcdf_temp(NC_ON)
    
    # Align data - use minimum length
    min_len = min(len(dates_off), len(dates_on))
    dates = dates_off[:min_len]
    temp_off = temp_off[:min_len, :]
    temp_on = temp_on[:min_len, :]
    
    # Find indices for key depths
    idx_40m = np.argmin(np.abs(depths - 40))
    idx_30m = np.argmin(np.abs(depths - 30))
    idx_extract = np.argmin(np.abs(depths - HP_EXTRACTION_ELEV))
    idx_bottom = np.argmin(np.abs(depths - 5))
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    plt.subplots_adjust(hspace=0.26, left=0.08, right=0.95, top=0.96, bottom=0.07)
    
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='0.7', alpha=0.95, linewidth=0.5)

    depth_labels = [
        (idx_40m, f'(a) Near Surface ({depths[idx_40m]:.0f} m above bottom)'),
        (idx_30m, f'(b) Mid-depth ({depths[idx_30m]:.0f} m above bottom)'),
        (idx_extract, f'(c) Heat Pump Depth ({depths[idx_extract]:.0f} m above bottom)'),
        (idx_bottom, f'(d) Near Bottom ({depths[idx_bottom]:.0f} m above bottom)')
    ]

    for i, (idx, title) in enumerate(depth_labels):
        ax = axes[i]
        
        # Get data for this depth
        data_off = temp_off[:, idx]
        data_on = temp_on[:, idx]
        
        # Plot data with GMD colors
        ax.plot(dates, data_off, color=COLORS['hp_off'], lw=1.2, 
                label='HP OFF', alpha=0.9, zorder=2)
        ax.plot(dates, data_on, color=COLORS['hp_on'], lw=1.2, 
                label='HP ON', alpha=0.9, zorder=2)
        
        # Styling
        style_axis(ax, ylabel='Temperature (°C)', title=title)
        
        # Legend only on first panel - position upper left to avoid summer peaks
        if i == 0:
            ax.legend(loc='upper left', fontsize=8, ncol=2, 
                     framealpha=0.98, edgecolor='0.8')
        
        # Statistics annotation
        diff = data_on - data_off
        mean_diff = np.nanmean(diff)
        min_diff = np.nanmin(diff)
        max_diff = np.nanmax(diff)
        
        # Stats text - single line format
        stats_text = f'ΔT mean: {mean_diff:+.2f}°C | min: {min_diff:+.2f}°C | max: {max_diff:+.2f}°C'
        
        # Configurable position (x, y in axes coordinates 0-1)
        stats_x, stats_y = 0.99, 0.97
        ax.text(stats_x, stats_y, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=bbox_props, family='monospace')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # X-axis formatting on bottom panel
    axes[-1].set_xlabel('Year')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

    outfile = f'{OUTPUT_DIR}/Fig8_temperature_depths.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

# ============================================================================
# STATISTICS CSV GENERATION
# ============================================================================
def generate_statistics_csv():
    """Generate temperature statistics comparison CSV file"""
    print("\n" + "="*70)
    print("GENERATING: Temperature Statistics CSV")
    print("="*70)
    
    # Load data
    df_off = load_lake_csv(CSV_OFF)
    df_on = load_lake_csv(CSV_ON)
    dates, depths, temp_off, _ = load_netcdf_temp(NC_OFF)
    _, _, temp_on, _ = load_netcdf_temp(NC_ON)
    
    # Find index for extraction/injection depth
    idx_extract = np.argmin(np.abs(depths - HP_EXTRACTION_ELEV))
    
    # Calculate statistics
    stats = [
        ('Mean Lake Temperature', np.nanmean(temp_off), np.nanmean(temp_on), '°C'),
        ('Mean Surface Temperature', df_off['Surface Temp'].mean(), df_on['Surface Temp'].mean(), '°C'),
        ('Max Temperature', np.nanmax(temp_off), np.nanmax(temp_on), '°C'),
        ('Min Temperature', np.nanmin(temp_off), np.nanmin(temp_on), '°C'),
        (f'Mean Temp at {HP_EXTRACTION_ELEV:.1f}m (extract/inject)', 
         np.nanmean(temp_off[:, idx_extract]), np.nanmean(temp_on[:, idx_extract]), '°C'),
        ('Mean Stratification (dT/dz)', df_off['Max dT/dz'].mean(), df_on['Max dT/dz'].mean(), '°C/m'),
    ]
    
    # Create DataFrame for CSV
    csv_data = []
    for name, off_val, on_val, unit in stats:
        diff = on_val - off_val
        csv_data.append({
            'Metric': name,
            'HP OFF': f'{off_val:.4f}',
            'HP ON': f'{on_val:.4f}',
            'Difference': f'{diff:.4f}',
            'Unit': unit
        })
    
    df_stats = pd.DataFrame(csv_data)
    
    # Save to CSV
    csv_file = f'{OUTPUT_DIR}/temperature_statistics.csv'
    df_stats.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")
    
    # Also print to console
    print("\n" + "="*90)
    print("TEMPERATURE STATISTICS TABLE")
    print("="*90)
    print(f"\n{'Metric':<45} {'HP OFF':>12} {'HP ON':>12} {'Difference':>12} {'Unit':>10}")
    print("-"*90)
    for name, off_val, on_val, unit in stats:
        diff = on_val - off_val
        print(f"{name:<45} {off_val:>12.4f} {on_val:>12.4f} {diff:>+12.4f} {unit:>10}")
    print("-"*90)
    
    return df_stats

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def generate_all_figures():
    """Generate all figures"""
    print("\n" + "="*70)
    print("MASTER SCRIPT - Heat Pump Impact Analysis")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Heat pump configuration:")
    print(f"  Flow rate: {HP_FLOW_RATE:,} m³/day")
    print(f"  Temperature change: {HP_TEMP_CHANGE}°C")
    print(f"  Extraction elevation: {HP_EXTRACTION_ELEV}m")
    print(f"  Injection elevation: {HP_INJECTION_ELEV}m")
    
    # Generate all figures
    plot_temperature_contour()       # Figure 1
    plot_level_volume()              # Figure 2
    plot_evaporation()               # Figure 3
    plot_volume_vs_evaporation()     # Figure 4
    plot_temperature_timeseries()    # Figure 5
    plot_seasonal_profiles()         # Figure 6
    plot_temperature_difference()    # Figure 7
    plot_temperature_depths()        # Figure 8
    
    # Generate statistics CSV
    generate_statistics_csv()
    
    print("\n" + "="*70)
    print("ALL OUTPUTS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nGenerated files in {OUTPUT_DIR}/:")
    print("  Fig1_temperature_contour.png     - Temperature contour (OFF vs ON)")
    print("  Fig2_level_volume.png            - Lake level and volume comparison")
    print("  Fig3_evaporation.png             - Evaporation analysis")
    print("  Fig4_volume_vs_evaporation.png   - Volume vs evaporation/overflow")
    print("  Fig5_temperature_timeseries.png  - Surface/Max/Min temperature")
    print("  Fig6_seasonal_profiles.png       - Seasonal temperature profiles")
    print("  Fig7_temperature_difference.png  - Temperature difference time series")
    print("  Fig8_temperature_depths.png      - Temperature at key depths")
    print("  temperature_statistics.csv       - Comparison statistics table")

if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 1:
        if '--temp-contour' in sys.argv:
            plot_temperature_contour()
        elif '--temp-analysis' in sys.argv:
            plot_temperature_timeseries()
            plot_seasonal_profiles()
            plot_temperature_difference()
            plot_temperature_depths()
        elif '--water-balance' in sys.argv:
            plot_level_volume()
            plot_evaporation()
            plot_volume_vs_evaporation()
        elif '--stats' in sys.argv:
            generate_statistics_csv()
        elif '--all' in sys.argv:
            generate_all_figures()
        else:
            print("Usage: python master_script.py [--all] [--temp-contour] [--temp-analysis] [--water-balance] [--stats]")
            print("  --all           Generate all figures and statistics (default)")
            print("  --temp-contour  Generate temperature contour plots only")
            print("  --temp-analysis Generate temperature analysis plots only")
            print("  --water-balance Generate water balance plots only")
            print("  --stats         Generate statistics CSV only")
    else:
        # Default: generate all figures
        generate_all_figures()
