#!/usr/bin/env python3
"""
Plot meteorological variables from met_hourly.csv
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Load the data
print("Loading met_hourly.csv...")
df = pd.read_csv('met_hourly.csv', parse_dates=['time'])
print(f"  Data loaded: {len(df)} rows")
print(f"  Period: {df['time'].min()} to {df['time'].max()}")
print(f"  Variables: {list(df.columns)}")

# Plot settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Create figure with 5 subplots (one for each variable)
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

variables = [
    ('ShortWave', 'Shortwave Radiation', 'W/m²', 'orange'),
    ('Cloud', 'Cloud Cover', 'fraction', 'gray'),
    ('AirTemp', 'Air Temperature', '°C', 'red'),
    ('RelHum', 'Relative Humidity', '%', 'blue'),
    ('WindSpeed', 'Wind Speed', 'm/s', 'green')
]

for i, (var, title, unit, color) in enumerate(variables):
    ax = axes[i]
    
    # For hourly data over 15 years, resample to daily for cleaner plots
    df_daily = df.set_index('time')[var].resample('D').mean()
    
    ax.plot(df_daily.index, df_daily.values, color=color, lw=0.5, alpha=0.8)
    ax.set_ylabel(f'{unit}')
    ax.set_title(f'({chr(97+i)}) {title}')
    
    # Add statistics
    stats_text = f'Mean: {df[var].mean():.2f}\nMin: {df[var].min():.2f}\nMax: {df[var].max():.2f}'
    ax.text(0.995, 0.02, stats_text, transform=ax.transAxes, fontsize=6,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[-1].set_xlabel('Year')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig('analysis_results/fig_met_hourly.png', dpi=300, bbox_inches='tight')
print("Saved: analysis_results/fig_met_hourly.png")

print("\nAll plots saved to analysis_results/")
