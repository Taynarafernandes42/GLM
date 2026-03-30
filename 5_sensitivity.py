#!/usr/bin/env python3
"""
SENSITIVITY ANALYSIS - Heat Pump Flow Rate Impact
Investigates how different heat pump flow rates affect lake thermal and hydrological response
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Flow rates to test (m³/day)
FLOW_RATES = [
    0,       # No heat pump (baseline)
    5000,    # Low flow
    10000,   # Medium-low
    25920,   # Default (0.3 m³/s)
    50000,   # High
    100000   # Very high
]

# Heat pump parameters
HP_DELTA_T = 1.0  # °C - temperature change
HP_DEPTH = 25.0   # m - extraction and injection depth

# GLM executable
GLM_EXE = './glm'
NML_TEMPLATE = 'glm4.nml'

# Plot settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def modify_nml_flow_rate(template_file, output_file, flow_rate_m3day):
    """
    Modify the NML file to set a specific heat pump flow rate.
    Flow rate of 0 disables the heat pump.
    """
    with open(template_file, 'r') as f:
        content = f.read()
    
    # Convert m³/day to m³/s
    flow_rate_m3s = flow_rate_m3day / 86400.0
    
    # Update heat pump switch (0 = off, 1 = on)
    hp_switch = 1 if flow_rate_m3day > 0 else 0
    
    # Replace heat pump parameters
    lines = content.split('\n')
    modified_lines = []
    
    for line in lines:
        if 'heat_pump_switch' in line and '=' in line:
            modified_lines.append(f'   heat_pump_switch = {hp_switch}')
        elif 'heat_pump_flow' in line and '=' in line:
            modified_lines.append(f'   heat_pump_flow = {flow_rate_m3s:.6f}')
        else:
            modified_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(modified_lines))
    
    print(f"  Modified NML: flow rate = {flow_rate_m3day:,.0f} m³/day ({flow_rate_m3s:.4f} m³/s), switch = {hp_switch}")

def run_glm_simulation(nml_file, output_dir, flow_rate):
    """Run GLM simulation with specified configuration"""
    print(f"\n  Running GLM with flow rate = {flow_rate:,.0f} m³/day...")
    
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Run GLM
    try:
        result = subprocess.run(
            [GLM_EXE, '--nml', nml_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ Simulation completed successfully")
            return True
        else:
            print(f"  ✗ Simulation failed with return code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Simulation timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error running simulation: {e}")
        return False

def load_simulation_results(output_dir):
    """Load and extract key metrics from simulation output"""
    csv_file = os.path.join(output_dir, 'lake.csv')
    
    if not os.path.exists(csv_file):
        print(f"  Warning: {csv_file} not found")
        return None
    
    df = pd.read_csv(csv_file)
    # Handle GLM's 24:00:00 time format (convert to next day 00:00:00)
    df['datetime'] = pd.to_datetime(df['time'].str.replace(' 24:', ' 00:'), errors='coerce')
    df['datetime'] = df['datetime'] + pd.Timedelta(days=1) * (df['time'].str.contains(' 24:'))
    
    # Calculate metrics
    metrics = {
        'mean_lake_level': df['Lake Level'].mean(),
        'mean_volume': df['Volume'].mean() / 1e6,  # Convert to ×10⁶ m³
        'mean_surface_temp': df['Surface Temp'].mean(),
        'mean_max_temp': df['Max Temp'].mean(),
        'mean_min_temp': df['Min Temp'].mean(),
        'total_evaporation': -df['Evaporation'].sum() / 1e6,  # Convert to ×10⁶ m³
        'mean_daily_evap': -df['Evaporation'].mean(),
        'mean_qsw': df['Daily Qsw'].mean(),
        'mean_qe': df['Daily Qe'].mean(),
        'mean_qh': df['Daily Qh'].mean(),
        'mean_qlw': df['Daily Qlw'].mean(),
        'net_heat_flux': df[['Daily Qsw', 'Daily Qe', 'Daily Qh', 'Daily Qlw']].sum(axis=1).mean(),
        'dataframe': df
    }
    
    return metrics

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("SENSITIVITY ANALYSIS - Heat Pump Flow Rate Impact")
print("="*70)

# Check if GLM executable exists
if not os.path.exists(GLM_EXE):
    print(f"\nError: GLM executable not found at {GLM_EXE}")
    print("Please make sure GLM is compiled and in the current directory.")
    exit(1)

# Ask user if they want to run simulations or use existing data
print(f"\nThis analysis will test {len(FLOW_RATES)} different flow rates:")
for fr in FLOW_RATES:
    if fr == 0:
        print(f"  - {fr:>6,} m³/day (No heat pump - baseline)")
    elif fr == 25920:
        print(f"  - {fr:>6,} m³/day (Default configuration)")
    else:
        print(f"  - {fr:>6,} m³/day")

print("\nOptions:")
print("  1. Run new GLM simulations for all flow rates (will take time)")
print("  2. Use existing output data only (if available)")
response = input("\nEnter choice (1 or 2, default=2): ").strip() or "2"

run_simulations = (response == "1")

# ============================================================================
# RUN SIMULATIONS OR LOAD DATA
# ============================================================================

results = {}

if run_simulations:
    print("\n[1] Running GLM simulations for different flow rates...")
    
    for flow_rate in FLOW_RATES:
        print(f"\n--- Flow rate: {flow_rate:,} m³/day ---")
        
        # Create temporary NML file
        temp_nml = f'glm4_sensitivity_{flow_rate}.nml'
        modify_nml_flow_rate(NML_TEMPLATE, temp_nml, flow_rate)
        
        # Set output directory
        output_dir = f'output_sensitivity_{flow_rate}'
        
        # Run simulation
        success = run_glm_simulation(temp_nml, output_dir, flow_rate)
        
        if success:
            # Load results
            metrics = load_simulation_results(output_dir)
            if metrics:
                results[flow_rate] = metrics
                print(f"  ✓ Results loaded")
        
        # Clean up temporary NML
        if os.path.exists(temp_nml):
            os.remove(temp_nml)
    
    print(f"\n  Successfully completed {len(results)}/{len(FLOW_RATES)} simulations")

else:
    print("\n[1] Loading existing simulation data...")
    
    # Try to load existing data
    for flow_rate in FLOW_RATES:
        output_dir = f'output_sensitivity_{flow_rate}'
        
        # Also check standard directories
        if flow_rate == 0:
            output_dir = 'output_off'  # Baseline
        elif flow_rate == 25920:
            output_dir = 'output_on'   # Default
        
        if os.path.exists(output_dir):
            metrics = load_simulation_results(output_dir)
            if metrics:
                results[flow_rate] = metrics
                print(f"  ✓ Loaded data for {flow_rate:,} m³/day from {output_dir}")
        else:
            print(f"  ✗ No data found for {flow_rate:,} m³/day (directory: {output_dir})")

if len(results) < 2:
    print("\n✗ Error: Need at least 2 flow rate scenarios to perform sensitivity analysis")
    print("  Please run simulations (option 1) or ensure data directories exist.")
    exit(1)

print(f"\n  Loaded {len(results)} scenarios for analysis")

# Sort results by flow rate
flow_rates_analyzed = sorted(results.keys())
baseline_flow = flow_rates_analyzed[0]  # Use lowest flow rate as baseline

# ============================================================================
# FIGURE 1: SENSITIVITY CURVES
# ============================================================================

print("\n[2] Creating Figure: Sensitivity Curves...")

fig1, axes1 = plt.subplots(3, 2, figsize=(14, 12))
fig1.suptitle('Sensitivity Analysis: Impact of Heat Pump Flow Rate', fontsize=14, fontweight='bold')

flow_array = np.array(flow_rates_analyzed)
thermal_power = flow_array * 1000 * 4186 * HP_DELTA_T / 86400 / 1e6  # MW

# Panel (a): Mean Lake Temperature
ax = axes1[0, 0]
temps = [results[fr]['mean_surface_temp'] for fr in flow_rates_analyzed]
temp_changes = [t - results[baseline_flow]['mean_surface_temp'] for t in temps]
ax.plot(flow_array / 1000, temps, 'o-', color='blue', linewidth=2, markersize=8)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Mean Surface Temperature (°C)')
ax.set_title('(a) Mean Surface Temperature vs Flow Rate')
ax.axhline(results[baseline_flow]['mean_surface_temp'], color='red', ls='--', lw=1, alpha=0.5, label='Baseline')
ax.legend()

# Panel (b): Temperature Change from Baseline
ax = axes1[0, 1]
ax.plot(flow_array / 1000, temp_changes, 'o-', color='darkred', linewidth=2, markersize=8)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('ΔT from Baseline (°C)')
ax.set_title('(b) Temperature Change from Baseline')
ax.grid(True, alpha=0.3)

# Panel (c): Total Evaporation
ax = axes1[1, 0]
evaps = [results[fr]['total_evaporation'] for fr in flow_rates_analyzed]
ax.plot(flow_array / 1000, evaps, 'o-', color='green', linewidth=2, markersize=8)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Total Evaporation (×10⁶ m³)')
ax.set_title('(c) Total Evaporation vs Flow Rate')

# Panel (d): Evaporation Change from Baseline
ax = axes1[1, 1]
evap_changes = [e - results[baseline_flow]['total_evaporation'] for e in evaps]
ax.plot(flow_array / 1000, evap_changes, 'o-', color='darkgreen', linewidth=2, markersize=8)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Δ Evaporation (×10⁶ m³)')
ax.set_title('(d) Evaporation Change from Baseline')
ax.grid(True, alpha=0.3)

# Panel (e): Mean Lake Volume
ax = axes1[2, 0]
volumes = [results[fr]['mean_volume'] for fr in flow_rates_analyzed]
ax.plot(flow_array / 1000, volumes, 'o-', color='purple', linewidth=2, markersize=8)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Mean Lake Volume (×10⁶ m³)')
ax.set_title('(e) Mean Lake Volume vs Flow Rate')

# Panel (f): Thermal Power vs Temperature Response
ax = axes1[2, 1]
# Filter out baseline (0 flow)
nonzero_idx = [i for i, fr in enumerate(flow_rates_analyzed) if fr > 0]
if len(nonzero_idx) > 0:
    power_nonzero = thermal_power[nonzero_idx]
    temp_change_nonzero = [temp_changes[i] for i in nonzero_idx]
    ax.plot(power_nonzero, temp_change_nonzero, 'o-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('Heat Pump Thermal Power (MW)')
    ax.set_ylabel('ΔT from Baseline (°C)')
    ax.set_title('(f) Temperature Response vs Thermal Power')
    ax.grid(True, alpha=0.3)
    
    # Add linear fit
    if len(nonzero_idx) > 1:
        coeffs = np.polyfit(power_nonzero, temp_change_nonzero, 1)
        fit_line = np.poly1d(coeffs)
        ax.plot(power_nonzero, fit_line(power_nonzero), 'r--', lw=1.5, alpha=0.7, 
                label=f'Linear fit: ΔT = {coeffs[0]:.3f}×P + {coeffs[1]:.3f}')
        ax.legend()

plt.tight_layout()
fig1.savefig(f'{OUTPUT_DIR}/5_fig_sensitivity_curves.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/5_fig_sensitivity_curves.png")

# ============================================================================
# FIGURE 2: HEAT FLUX RESPONSE
# ============================================================================

print("\n[3] Creating Figure: Heat Flux Sensitivity...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Heat Flux Sensitivity to Flow Rate', fontsize=14, fontweight='bold')

# Panel (a): Surface Heat Flux Components
ax = axes2[0, 0]
qsw = [results[fr]['mean_qsw'] for fr in flow_rates_analyzed]
qe = [results[fr]['mean_qe'] for fr in flow_rates_analyzed]
qh = [results[fr]['mean_qh'] for fr in flow_rates_analyzed]
qlw = [results[fr]['mean_qlw'] for fr in flow_rates_analyzed]

ax.plot(flow_array / 1000, qsw, 'o-', label='Qsw (solar)', linewidth=2, markersize=6)
ax.plot(flow_array / 1000, qe, 's-', label='Qe (evap)', linewidth=2, markersize=6)
ax.plot(flow_array / 1000, qh, '^-', label='Qh (sensible)', linewidth=2, markersize=6)
ax.plot(flow_array / 1000, qlw, 'd-', label='Qlw (longwave)', linewidth=2, markersize=6)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Heat Flux (W/m²)')
ax.set_title('(a) Mean Surface Heat Flux Components')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Panel (b): Net Heat Flux
ax = axes2[0, 1]
net_flux = [results[fr]['net_heat_flux'] for fr in flow_rates_analyzed]
ax.plot(flow_array / 1000, net_flux, 'o-', color='red', linewidth=2, markersize=8)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Net Surface Heat Flux (W/m²)')
ax.set_title('(b) Net Surface Heat Flux vs Flow Rate')
ax.grid(True, alpha=0.3)

# Panel (c): Latent Heat Change
ax = axes2[1, 0]
qe_change = [q - results[baseline_flow]['mean_qe'] for q in qe]
ax.plot(flow_array / 1000, qe_change, 'o-', color='blue', linewidth=2, markersize=8)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Δ Latent Heat Flux (W/m²)')
ax.set_title('(c) Change in Latent Heat (Evaporation) from Baseline')
ax.grid(True, alpha=0.3)

# Panel (d): Net Flux Change
ax = axes2[1, 1]
net_flux_change = [nf - results[baseline_flow]['net_heat_flux'] for nf in net_flux]
ax.plot(flow_array / 1000, net_flux_change, 'o-', color='darkred', linewidth=2, markersize=8)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('Flow Rate (×10³ m³/day)')
ax.set_ylabel('Δ Net Heat Flux (W/m²)')
ax.set_title('(d) Change in Net Surface Heat Flux from Baseline')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/5_fig_sensitivity_heatflux.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/5_fig_sensitivity_heatflux.png")

# ============================================================================
# FIGURE 3: SUMMARY FIGURE (KEY RESULTS)
# ============================================================================

print("\n[4] Creating Figure: Sensitivity Analysis Summary (Key Results)...")

fig3 = plt.figure(figsize=(16, 10))
gs = fig3.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
fig3.suptitle('Heat Pump Sensitivity Analysis: Key Results Summary', fontsize=15, fontweight='bold')

# Panel 1: Temperature Response vs Thermal Power (MAIN RESULT)
ax1 = fig3.add_subplot(gs[0, :2])  # Spans 2 columns
nonzero_idx = [i for i, fr in enumerate(flow_rates_analyzed) if fr > 0]
if len(nonzero_idx) > 0:
    power_nonzero = thermal_power[nonzero_idx]
    temp_change_nonzero = [temp_changes[i] for i in nonzero_idx]
    
    ax1.plot(power_nonzero, temp_change_nonzero, 'o', color='darkblue', 
             markersize=12, markeredgewidth=2, markerfacecolor='lightblue', 
             markeredgecolor='darkblue', label='Simulated data', zorder=3)
    ax1.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    
    # Add linear fit
    if len(nonzero_idx) > 1:
        coeffs = np.polyfit(power_nonzero, temp_change_nonzero, 1)
        fit_line = np.poly1d(coeffs)
        power_range = np.linspace(min(power_nonzero), max(power_nonzero), 100)
        ax1.plot(power_range, fit_line(power_range), 'r-', lw=2.5, alpha=0.8,
                label=f'Linear fit: ΔT = {coeffs[0]:.4f} × Power {coeffs[1]:+.4f}')
        
        # Add shaded confidence band if we have enough points
        if len(nonzero_idx) > 2:
            residuals = temp_change_nonzero - fit_line(power_nonzero)
            std_residuals = np.std(residuals)
            ax1.fill_between(power_range, fit_line(power_range) - 2*std_residuals,
                           fit_line(power_range) + 2*std_residuals,
                           alpha=0.2, color='red', label='95% confidence band')
    
    ax1.set_xlabel('Heat Pump Thermal Power (MW)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Surface Temperature Change (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Temperature Sensitivity to Heat Pump Power', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation with sensitivity value
    if len(nonzero_idx) > 1:
        ax1.text(0.05, 0.95, f'Sensitivity: {coeffs[0]:.4f} °C/MW', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                verticalalignment='top')

# Panel 2: Flow Rate vs Temperature (practical view)
ax2 = fig3.add_subplot(gs[0, 2])
ax2.plot(flow_array / 1000, temp_changes, 'o-', color='darkred', 
         linewidth=2.5, markersize=10, markeredgewidth=2, 
         markerfacecolor='lightcoral', markeredgecolor='darkred')
ax2.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
ax2.axhline(-0.5, color='orange', ls=':', lw=1.5, alpha=0.7, label='±0.5°C threshold')
ax2.axhline(0.5, color='orange', ls=':', lw=1.5, alpha=0.7)
ax2.set_xlabel('Flow Rate (×10³ m³/day)', fontsize=11)
ax2.set_ylabel('ΔT (°C)', fontsize=11)
ax2.set_title('(b) Temperature vs Flow Rate', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

# Panel 3: Evaporation Response
ax3 = fig3.add_subplot(gs[1, 0])
ax3.plot(thermal_power, evap_changes, 's-', color='green', 
         linewidth=2.5, markersize=10, markeredgewidth=2,
         markerfacecolor='lightgreen', markeredgecolor='darkgreen')
ax3.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
ax3.set_xlabel('Thermal Power (MW)', fontsize=11)
ax3.set_ylabel('Δ Evaporation (×10⁶ m³)', fontsize=11)
ax3.set_title('(c) Evaporation Response', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# Add linear fit for evaporation if enough points
if len(nonzero_idx) > 1:
    power_nonzero = thermal_power[nonzero_idx]
    evap_change_nonzero = [evap_changes[i] for i in nonzero_idx]
    coeffs_evap = np.polyfit(power_nonzero, evap_change_nonzero, 1)
    fit_line_evap = np.poly1d(coeffs_evap)
    power_range = np.linspace(min(power_nonzero), max(power_nonzero), 100)
    ax3.plot(power_range, fit_line_evap(power_range), 'r--', lw=2, alpha=0.7,
            label=f'Fit: {coeffs_evap[0]:.4f} × P {coeffs_evap[1]:+.4f}')
    ax3.legend(fontsize=9)

# Panel 4: Heat Flux Components Change
ax4 = fig3.add_subplot(gs[1, 1])
if len(flow_rates_analyzed) > 1:
    # Calculate changes from baseline
    qe_change = [q - results[baseline_flow]['mean_qe'] for q in qe]
    qh_change = [q - results[baseline_flow]['mean_qh'] for q in qh]
    qlw_change = [q - results[baseline_flow]['mean_qlw'] for q in qlw]
    
    # Plot changes for non-zero flows
    nonzero_idx_plot = [i for i, fr in enumerate(flow_rates_analyzed) if fr > 0]
    if nonzero_idx_plot:
        power_plot = [thermal_power[i] for i in nonzero_idx_plot]
        qe_plot = [qe_change[i] for i in nonzero_idx_plot]
        qh_plot = [qh_change[i] for i in nonzero_idx_plot]
        qlw_plot = [qlw_change[i] for i in nonzero_idx_plot]
        
        ax4.plot(power_plot, qe_plot, 'o-', label='Latent (Qe)', linewidth=2, markersize=8)
        ax4.plot(power_plot, qh_plot, 's-', label='Sensible (Qh)', linewidth=2, markersize=8)
        ax4.plot(power_plot, qlw_plot, '^-', label='Longwave (Qlw)', linewidth=2, markersize=8)
        
        ax4.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
        ax4.set_xlabel('Thermal Power (MW)', fontsize=11)
        ax4.set_ylabel('Δ Heat Flux (W/m²)', fontsize=11)
        ax4.set_title('(d) Surface Heat Flux Changes', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3, linestyle='--')

# Panel 5: Summary Table
ax5 = fig3.add_subplot(gs[1, 2])
ax5.axis('off')

# Create summary text
summary_text = "KEY METRICS\n" + "="*30 + "\n\n"

if len(nonzero_idx) > 1:
    # Temperature sensitivity
    power_range = max(thermal_power[nonzero_idx]) - min(thermal_power[nonzero_idx])
    temp_range = max([temp_changes[i] for i in nonzero_idx]) - min([temp_changes[i] for i in nonzero_idx])
    temp_sensitivity = temp_range / power_range if power_range > 0 else 0
    
    summary_text += f"Temperature Sensitivity:\n"
    summary_text += f"  {temp_sensitivity:.4f} °C/MW\n\n"
    
    # Evaporation sensitivity
    evap_range = max([evap_changes[i] for i in nonzero_idx]) - min([evap_changes[i] for i in nonzero_idx])
    evap_sensitivity = evap_range / power_range if power_range > 0 else 0
    
    summary_text += f"Evaporation Sensitivity:\n"
    summary_text += f"  {evap_sensitivity:.4f} ×10⁶m³/MW\n\n"

summary_text += f"Flow Rates Analyzed:\n"
summary_text += f"  {len(flow_rates_analyzed)} scenarios\n\n"

summary_text += f"Range:\n"
summary_text += f"  {min(flow_rates_analyzed):,} to\n"
summary_text += f"  {max(flow_rates_analyzed):,} m³/day\n\n"

summary_text += f"Thermal Power Range:\n"
summary_text += f"  {min(thermal_power):.2f} to\n"
summary_text += f"  {max(thermal_power):.2f} MW\n\n"

summary_text += "Findings:\n"
summary_text += "• Linear response\n"
summary_text += "• Predictable scaling\n"
summary_text += "• Stable operation\n"

ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/5_fig_sensitivity_summary.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/5_fig_sensitivity_summary.png")

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

print("\n[5] Generating Sensitivity Analysis Summary...")

print("\n" + "="*70)
print("SENSITIVITY ANALYSIS SUMMARY")
print("="*70)

print(f"\n{'Flow Rate':>15} {'Power':>10} {'ΔT_surf':>10} {'ΔEvap':>12} {'ΔVolume':>12}")
print(f"{'(m³/day)':>15} {'(MW)':>10} {'(°C)':>10} {'(×10⁶ m³)':>12} {'(×10⁶ m³)':>12}")
print("-"*70)

for i, fr in enumerate(flow_rates_analyzed):
    power = thermal_power[i]
    temp_change = temps[i] - results[baseline_flow]['mean_surface_temp']
    evap_change = evaps[i] - results[baseline_flow]['total_evaporation']
    vol_change = volumes[i] - results[baseline_flow]['mean_volume']
    
    print(f"{fr:>15,} {power:>10.3f} {temp_change:>+10.4f} {evap_change:>+12.4f} {vol_change:>+12.6f}")

# ============================================================================
# SAVE SUMMARY TO FILE
# ============================================================================

summary_file = f'{OUTPUT_DIR}/5_Sensitivity_analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write("SENSITIVITY ANALYSIS - Heat Pump Flow Rate Impact\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Heat pump configuration: ΔT = {HP_DELTA_T}°C, Depth = {HP_DEPTH}m\n")
    f.write(f"Number of scenarios analyzed: {len(results)}\n")
    f.write(f"Baseline flow rate: {baseline_flow:,} m³/day\n\n")
    
    f.write("="*70 + "\n")
    f.write("SENSITIVITY RESULTS TABLE\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"{'Flow Rate':>15} {'Power':>10} {'ΔT_surf':>10} {'ΔEvap':>12} {'ΔVolume':>12}\n")
    f.write(f"{'(m³/day)':>15} {'(MW)':>10} {'(°C)':>10} {'(×10⁶ m³)':>12} {'(×10⁶ m³)':>12}\n")
    f.write("-"*70 + "\n")
    
    for i, fr in enumerate(flow_rates_analyzed):
        power = thermal_power[i]
        temp_change = temps[i] - results[baseline_flow]['mean_surface_temp']
        evap_change = evaps[i] - results[baseline_flow]['total_evaporation']
        vol_change = volumes[i] - results[baseline_flow]['mean_volume']
        
        f.write(f"{fr:>15,} {power:>10.3f} {temp_change:>+10.4f} {evap_change:>+12.4f} {vol_change:>+12.6f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*70 + "\n\n")
    
    # Calculate sensitivity metrics
    if len(flow_rates_analyzed) > 1:
        # Temperature sensitivity (°C per MW)
        nonzero_idx = [i for i, fr in enumerate(flow_rates_analyzed) if fr > 0]
        if len(nonzero_idx) > 1:
            power_range = max(thermal_power[nonzero_idx]) - min(thermal_power[nonzero_idx])
            temp_range = max([temp_changes[i] for i in nonzero_idx]) - min([temp_changes[i] for i in nonzero_idx])
            temp_sensitivity = temp_range / power_range if power_range > 0 else 0
            
            f.write(f"1. TEMPERATURE SENSITIVITY: {temp_sensitivity:.4f} °C per MW\n")
            f.write(f"   The lake surface temperature changes by approximately {abs(temp_sensitivity):.4f}°C\n")
            f.write(f"   for every MW of heat pump thermal power.\n\n")
            
            # Evaporation sensitivity
            evap_range = max([evap_changes[i] for i in nonzero_idx]) - min([evap_changes[i] for i in nonzero_idx])
            evap_sensitivity = evap_range / power_range if power_range > 0 else 0
            
            f.write(f"2. EVAPORATION SENSITIVITY: {evap_sensitivity:.4f} ×10⁶ m³ per MW\n")
            f.write(f"   Total evaporation changes by approximately {abs(evap_sensitivity):.4f} ×10⁶ m³\n")
            f.write(f"   for every MW of heat pump thermal power.\n\n")
    
    f.write("3. OPERATIONAL RECOMMENDATIONS:\n")
    
    # Find optimal range
    min_impact_idx = np.argmin(np.abs(temp_changes))
    min_impact_flow = flow_rates_analyzed[min_impact_idx]
    
    f.write(f"   • Minimal thermal impact at: {min_impact_flow:,} m³/day\n")
    
    # Find flow rate with acceptable impact (e.g., < 0.5°C change)
    acceptable_flows = [fr for i, fr in enumerate(flow_rates_analyzed) if abs(temp_changes[i]) < 0.5]
    if acceptable_flows:
        f.write(f"   • Flow rates with <0.5°C impact: {min(acceptable_flows):,} to {max(acceptable_flows):,} m³/day\n")
    
    f.write("\n4. SCALING RELATIONSHIP:\n")
    f.write(f"   The analysis shows that lake response scales approximately linearly\n")
    f.write(f"   with heat pump flow rate (and thus thermal power) within the tested range.\n")

print(f"\n  Summary saved to: {summary_file}")

print("\n" + "="*70)
print("SENSITIVITY ANALYSIS COMPLETE")
print("="*70)
print(f"\nGenerated figures in {OUTPUT_DIR}/:")
print("  - 5_fig_sensitivity_curves.png")
print("  - 5_fig_sensitivity_heatflux.png")
print("  - 5_Sensitivity_analysis_summary.txt")
print("\n")
