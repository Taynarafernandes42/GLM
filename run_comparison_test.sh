#!/bin/bash

# Test script to run identical simulations with and without heat pump
# This will create a proper comparison for the heat pump effect

echo "=== GLM Heat Pump Comparison Test ==="
echo "This script runs two identical simulations:"
echo "1. Baseline (heat pump OFF)"  
echo "2. Heat pump (heat pump ON with 1°C cooling)"
echo

# Create output directories
echo "Creating output directories..."
mkdir -p output_baseline
mkdir -p output_heatpump

# Run baseline simulation (heat pump OFF)
echo "=== Running BASELINE simulation (heat pump OFF) ==="
echo "Command: ./glm glm4_baseline.nml"
echo "Output: ./output_baseline/"
echo
./glm glm4_baseline.nml

if [ $? -eq 0 ]; then
    echo "✓ Baseline simulation completed successfully"
else
    echo "✗ Baseline simulation failed"
    exit 1
fi

echo

# Run heat pump simulation (heat pump ON)
echo "=== Running HEAT PUMP simulation (heat pump ON) ==="
echo "Command: ./glm glm4_heatpump.nml"
echo "Output: ./output_heatpump/"
echo
./glm glm4_heatpump.nml

if [ $? -eq 0 ]; then
    echo "✓ Heat pump simulation completed successfully"
else
    echo "✗ Heat pump simulation failed"
    exit 1
fi

echo
echo "=== Comparison Ready ==="
echo "You can now compare:"
echo "  Baseline:  ./output_baseline/output.nc"
echo "  Heat pump: ./output_heatpump/output.nc"
echo
echo "Update your plotting script to use these files:"
echo "  nc_file_before = '/home/taynara/AED_Tools/GLM/output_baseline/output.nc'"
echo "  nc_file_after = '/home/taynara/AED_Tools/GLM/output_heatpump/output.nc'"
