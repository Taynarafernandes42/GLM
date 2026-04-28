/******************************************************************************
*                                                                             *
* glm_heatexchange.c                                                          *
*                                                                             *
* Developed by:                                                               *
* Taynara Fernandes                                                           *
* Matt Hipsey                                                                 *
* Casper Boon                                                                 *
*                                                                             *
* Helmholtz Centre for Environmental Research (UFZ)                           *
* Department of Lake Research (SEEFO)                                         *
*                                                                             *
******************************************************************************/

#include <stdio.h>
#include <math.h>
#include "glm.h"
#include "glm_types.h" 
#include "glm_const.h"
#include "glm_globals.h"
#include "glm_util.h"
#include "glm_input.h"
#include "glm_mixu.h"

// Storage for intercepted outflow
static AED_REAL stored_flow_rate  = 0.0;
static AED_REAL stored_temp       = 0.0;
static AED_REAL stored_salt       = 0.0;
static AED_REAL stored_Drawheight = 0.0;
static int stored_jday            = -1;

// Constraint violation tracking - per day and total
static int clamp_count_freeze     = 0;   // Clamped due to min temp (freezing)
static int clamp_count_hot        = 0;   // Clamped due to max temp
static int clamp_count_delta_t    = 0;   // Clamped due to max ΔT
static int clamp_count_flow       = 0;   // Clamped due to max flow
static int skip_count_cold_intake = 0;   // Skipped due to intake too cold
static int last_summary_jday      = -1;  // Track when we last printed summary

// Running totals for final report
static int total_clamp_freeze     = 0;
static int total_clamp_hot        = 0;
static int total_clamp_delta_t    = 0;
static int total_clamp_flow       = 0;
static int total_skip_cold        = 0;

// First/last day tracking for summary output
static int first_hp_jday          = -1;  // First day heat pump operated
static int last_hp_jday           = -1;  // Last day heat pump operated
static AED_REAL first_hp_flow     = 0.0;
static AED_REAL first_hp_extract_temp = 0.0;
static AED_REAL first_hp_inject_temp = 0.0;
static AED_REAL first_hp_delta_t  = 0.0;
static AED_REAL last_hp_flow      = 0.0;
static AED_REAL last_hp_extract_temp = 0.0;
static AED_REAL last_hp_inject_temp = 0.0;
static AED_REAL last_hp_delta_t   = 0.0;

static AED_REAL stored_WQ[MaxVars]; // WQ variables from outflow

/*****************************************************************************
 * Print daily constraint summary and reset counters                         *
 *****************************************************************************/
static void print_constraint_summary(int jday)
{
    int total_clamps = clamp_count_freeze + clamp_count_hot + clamp_count_delta_t + clamp_count_flow;
    
    if (total_clamps > 0 || skip_count_cold_intake > 0) {
        printf("Heat pump constraints [jday %d]: ", jday);
        
        if (clamp_count_freeze > 0) {
            printf("%d clamped (freezing), ", clamp_count_freeze);
        }
        if (clamp_count_hot > 0) {
            printf("%d clamped (too hot), ", clamp_count_hot);
        }
        if (clamp_count_delta_t > 0) {
            printf("%d clamped (ΔT limit), ", clamp_count_delta_t);
        }
        if (clamp_count_flow > 0) {
            printf("%d clamped (flow limit), ", clamp_count_flow);
        }
        if (skip_count_cold_intake > 0) {
            printf("%d skipped (intake cold), ", skip_count_cold_intake);
        }
        printf("\n");
    }
    
    // Update totals
    total_clamp_freeze += clamp_count_freeze;
    total_clamp_hot    += clamp_count_hot;
    total_clamp_delta_t += clamp_count_delta_t;
    total_clamp_flow   += clamp_count_flow;
    total_skip_cold    += skip_count_cold_intake;
    
    // Reset daily counters
    clamp_count_freeze = 0;
    clamp_count_hot    = 0;
    clamp_count_delta_t = 0;
    clamp_count_flow   = 0;
    skip_count_cold_intake = 0;
}

/*****************************************************************************
 * Physical constraint checking for heat pump operations                     *
 * Returns: 0 = OK, 1 = warning issued (clamped), 2 = critical error (skip)  *
 *****************************************************************************/
static int check_physical_constraints(int jday, AED_REAL *flow_rate, AED_REAL intake_temp, 
                                       AED_REAL *injection_temp, AED_REAL *delta_t)
{
    int status = 0;
    
    // Constraint 1: Check withdrawal temperature (system protection)
    if (intake_temp < heat_pump_min_withdraw_temp) {
        skip_count_cold_intake++;
        return 2;  // Critical - skip this injection entirely
    }
    
    // Constraint 2: Check flow rate limits
    if (*flow_rate > heat_pump_max_flow) {
        if (heat_pump_enforce_limits) {
            *flow_rate = heat_pump_max_flow;
        }
        clamp_count_flow++;
        status = 1;
    }
    
    // Constraint 3: Check ΔT magnitude
    if (fabs(*delta_t) > heat_pump_max_delta_t) {
        if (heat_pump_enforce_limits) {
            // Preserve sign, clamp magnitude
            *delta_t = (*delta_t > 0) ? heat_pump_max_delta_t : -heat_pump_max_delta_t;
            *injection_temp = intake_temp + *delta_t;
        }
        clamp_count_delta_t++;
        status = 1;
    }
    
    // Constraint 4: Check injection temperature - FREEZING (critical)
    if (*injection_temp < heat_pump_min_temp) {
        if (heat_pump_enforce_limits) {
            *injection_temp = heat_pump_min_temp;
            *delta_t = *injection_temp - intake_temp;
        }
        clamp_count_freeze++;
        status = 1;
    }
    
    // Constraint 5: Check injection temperature - TOO HOT (ecological concern)
    if (*injection_temp > heat_pump_max_temp) {
        if (heat_pump_enforce_limits) {
            *injection_temp = heat_pump_max_temp;
            *delta_t = *injection_temp - intake_temp;
        }
        clamp_count_hot++;
        status = 1;
    }
    
    // Constraint 6: Sanity check - injection volume vs layer volume
    // (Will be checked in heat_pump_insert_inflow after finding layer)
    
    return status;
}

/*****************************************************************************
 * Capture and store live flow data from GLM's output (called in glm_flow.c) *
 *****************************************************************************/
void heat_pump_capture_outflow(int jday, AED_REAL DrawHeight, AED_REAL vol, AED_REAL temp, AED_REAL salt, AED_REAL *wq_vars)
{
    // Only capture if heat pump is enabled
    if (heat_pump_switch <= 0) return;

    // Clear previous values when capturing new flow
    if (stored_jday != -1 && stored_jday != jday) {
        stored_flow_rate  = 0.0;
        stored_temp       = 0.0;
        stored_salt       = 0.0;
        stored_Drawheight = 0.0;
        // Clear WQ variables
        if (Num_WQ_Vars > 0 && wq_vars != NULL) {
            for (int wqidx = 0; wqidx < Num_WQ_Vars; wqidx++) {
                stored_WQ[wqidx] = 0.0;
            }
        }
    }

    // Store the captured flow data to modify heat pump backflow
    stored_flow_rate  = vol;
    stored_temp       = temp;
    stored_salt       = salt;
    stored_Drawheight = DrawHeight;
    stored_jday       = jday;
    
    // Store WQ variables
    if (Num_WQ_Vars > 0 && wq_vars != NULL) {
        for (int wqidx = 0; wqidx < Num_WQ_Vars; wqidx++) {
            stored_WQ[wqidx] = wq_vars[wqidx];
        }
    }
}
/*************************************************************************************
 * Insert heat pump water directly                                                   *
 * Called AFTER do_outflows() in glm_model.c for mass conservation                   *
 *************************************************************************************/
void heat_pump_insert_inflow(int jday) 
{
    // Only proceed if heat pump is enabled
    if (heat_pump_switch <= 0) return;

    // Check if the specified inflow index is valid
    if (heat_pump_inflow_idx < 0 || heat_pump_inflow_idx >= NumInf) {
        printf("ERROR: heat_pump_inflow_idx (%d) is out of range [0, %d]\n", 
               heat_pump_inflow_idx, NumInf-1);
        return;
    }
    
    // For Mode 1: require captured outflow data
    // For Mode 2: can operate independently using extraction layer temperature
    if (heat_pump_switch == 1) {
        if (stored_jday == -1 || stored_flow_rate <= 0.0) return;
    }
    
    // Variables for extraction temperature and salinity
    AED_REAL extraction_temp;
    AED_REAL extraction_salt;
    
    // Get extraction temperature: from captured outflow or from extraction layer
    if (stored_flow_rate > 0.0 && stored_jday != -1) {
        // Use captured outflow data
        extraction_temp = stored_temp;
        extraction_salt = stored_salt;
    } else if (heat_pump_switch == 2) {
        // Mode 2: Get temp from extraction layer (outflow elevation)
        if (heat_pump_outflow_idx < 0 || heat_pump_outflow_idx >= NumOut) {
            printf("ERROR: heat_pump_outflow_idx (%d) out of range for Mode 2\n", heat_pump_outflow_idx);
            return;
        }
        
        // Find extraction layer based on outflow elevation
        AED_REAL extract_elev = Outflows[heat_pump_outflow_idx].OLev;
        int Layer_extract;
        for (Layer_extract = botmLayer; Layer_extract <= surfLayer; Layer_extract++) {
            if (Lake[Layer_extract].Height >= extract_elev) break;
        }
        if (Layer_extract > surfLayer) Layer_extract = surfLayer;
        
        extraction_temp = Lake[Layer_extract].Temp;
        extraction_salt = Lake[Layer_extract].Salinity;
    } else {
        // Mode 1 without captured data - cannot proceed
        return;
    }

    // Calculate temperature change caused by the heat pump
    // Sign convention (physics standard):
    //   Positive = heat added to water (warming) - e.g., summer heat rejection
    //   Negative = heat removed from water (cooling) - e.g., winter heat extraction
    AED_REAL heated_temp;
    AED_REAL temp_change_value;
    AED_REAL step_duration_seconds = subdaily ? noSecs : SecsPerDay;
    AED_REAL flow_to_inject = stored_flow_rate;
    AED_REAL flow_rate_m3day_equiv;
    
    switch (heat_pump_switch) {
        case 1: {
            // Mode 1: Fixed temperature change (defined in .nml file)
            // Positive heat_pump_temp_change = warming, Negative = cooling
            temp_change_value = heat_pump_temp_change;
            heated_temp = extraction_temp + temp_change_value;
            break;
        }
        case 2: {
            // Mode 2: Heat flux-based with DYNAMIC FLOW RATE calculation
            // Target: achieve specified heat flux by adjusting flow rate
            // Q = Φ / (ρ × cp × ΔT)  [rearranged from Equation 2]
            // Positive Φ = heat added to water (warming)
            // Negative Φ = heat removed from water (cooling)
            
            // Use dynamic heat flux if available, otherwise use static value
            AED_REAL current_heat_flux = (heat_pump_dynamic_heat_flux != 0.0) ? 
                                        heat_pump_dynamic_heat_flux : heat_pump_heat_flux;
            
            // Skip if no heat flux specified
            if (fabs(current_heat_flux) < 1e-10) {
                return;  // No heat flux = no operation
            }
            
            // Determine maximum allowable ΔT based on temperature constraints
            AED_REAL max_delta_t;
            if (current_heat_flux < 0) {
                // Cooling mode (heat extraction): injection temp cannot go below min_temp
                // ΔT is negative, so max magnitude is (extraction_temp - min_temp)
                max_delta_t = -(extraction_temp - heat_pump_min_temp);
                // Also respect max_delta_t setting
                if (fabs(max_delta_t) > heat_pump_max_delta_t) {
                    max_delta_t = -heat_pump_max_delta_t;
                }
            } else {
                // Heating mode (heat rejection): injection temp cannot exceed max_temp
                max_delta_t = heat_pump_max_temp - extraction_temp;
                if (max_delta_t > heat_pump_max_delta_t) {
                    max_delta_t = heat_pump_max_delta_t;
                }
            }
            
            // Check for zero delta_t (temperature at constraint limit)
            if (fabs(max_delta_t) < 1e-6) {
                // Cannot change temperature - skip this timestep
                return;
            }
            
            // Calculate required flow rate to achieve target flux with this ΔT
            // Q (m³/s) = Φ (W) / (ρ (kg/m³) × cp (J/kg·K) × ΔT (K))
            AED_REAL required_flow_m3s = fabs(current_heat_flux) / (rho0 * SPHEAT * fabs(max_delta_t));
            AED_REAL required_flow_m3day = required_flow_m3s * SecsPerDay;
            
            // Apply flow rate limits
            if (required_flow_m3day > heat_pump_max_flow) {
                // Flow capped - actual flux will be less than target
                flow_to_inject = heat_pump_max_flow * step_duration_seconds / SecsPerDay;
                // Recalculate ΔT based on capped flow
                AED_REAL actual_flow_m3s = flow_to_inject / step_duration_seconds;
                temp_change_value = current_heat_flux / (rho0 * actual_flow_m3s * SPHEAT);
            } else {
                // Can achieve target flux
                flow_to_inject = required_flow_m3s * step_duration_seconds;
                temp_change_value = max_delta_t;
            }
            
            heated_temp = extraction_temp + temp_change_value;
            break;
        }
        default: {
            // Default to mode 1 behavior for backward compatibility
            temp_change_value = heat_pump_temp_change;
            heated_temp = extraction_temp + temp_change_value;
            break;
        }
    }
    
    // =========================================================================
    // PHYSICAL CONSTRAINTS CHECK
    // =========================================================================
    int constraint_status = check_physical_constraints(
        jday, &flow_to_inject, extraction_temp, &heated_temp, &temp_change_value);
    
    if (constraint_status == 2) {
        // Critical constraint violation - skip this injection
        stored_flow_rate = 0.0;  // Clear stored data
        return;
    }
    // If constraint_status == 1, values have been clamped and we continue
   
    // Get the injection elevation from the inflow configuration
    AED_REAL inject_elev = Inflows[heat_pump_inflow_idx].SubmElev;
    
    // Find the layer at injection elevation
    int Layer_inject;
    for (Layer_inject = botmLayer; Layer_inject <= surfLayer; Layer_inject++) {
        if (Lake[Layer_inject].Height >= inject_elev) break;
    }
    if (Layer_inject > surfLayer) Layer_inject = surfLayer;
    
    // Print daily summary if we've moved to a new day
    if (last_summary_jday == -1) {
        // First call - initialize
        last_summary_jday = jday;
    } else if (jday != last_summary_jday) {
        // Day changed - print summary for the previous day
        print_constraint_summary(last_summary_jday);
        last_summary_jday = jday;
    }
    
    // Additional constraint: Check if injection volume is reasonable compared to layer volume
    // (This is informational only - printed in the regular status output if relevant)
    
    // Calculate density of injected water
    AED_REAL inject_density = calculate_density(heated_temp, extraction_salt);
    
    // Directly inject into the lake layer
    // This combines the injected water properties with the existing layer
    Lake[Layer_inject].Temp = combine(Lake[Layer_inject].Temp, Lake[Layer_inject].LayerVol, Lake[Layer_inject].Density,
                                      heated_temp, flow_to_inject, inject_density);
    Lake[Layer_inject].Salinity = combine(Lake[Layer_inject].Salinity, Lake[Layer_inject].LayerVol, Lake[Layer_inject].Density,
                                          extraction_salt, flow_to_inject, inject_density);

    // Inject WQ variables
    if (Num_WQ_Vars > 0 && WQ_Vars != NULL) {
        for (int wqidx = 0; wqidx < Num_WQ_Vars; wqidx++) {
            _WQ_Vars(wqidx, Layer_inject) = combine_vol(_WQ_Vars(wqidx, Layer_inject), Lake[Layer_inject].LayerVol,
                                                        stored_WQ[wqidx], flow_to_inject);
        }
    }

    // Update layer density after mixing
    Lake[Layer_inject].Density = calculate_density(Lake[Layer_inject].Temp, Lake[Layer_inject].Salinity);
    
    // Add the volume back to the layer (mass conservation: outflow removed it, now we add it back)
    Lake[Layer_inject].LayerVol = Lake[Layer_inject].LayerVol + flow_to_inject;

    // Update cumulative volumes
    Lake[botmLayer].Vol1 = Lake[botmLayer].LayerVol;
    if (surfLayer != botmLayer) {
        for (int j = (botmLayer + 1); j <= surfLayer; j++) {
            Lake[j].Vol1 = Lake[j-1].Vol1 + Lake[j].LayerVol;
        }
    }

    // Recalculate layer heights from volumes (required for consistency)
    resize_internals(2, botmLayer);

    // Track first and last day info for summary
    if (first_hp_jday == -1) {
        // First day of heat pump operation - store and print
        first_hp_jday = jday;
        flow_rate_m3day_equiv = flow_to_inject * SecsPerDay / step_duration_seconds;
        first_hp_flow = flow_rate_m3day_equiv;
        first_hp_extract_temp = extraction_temp;
        first_hp_inject_temp = heated_temp;
        first_hp_delta_t = temp_change_value;
        
        printf("Heat pump FIRST operation at jday %d: Q=%.1f m³/d, T_extract=%.1f°C, T_inject=%.1f°C (ΔT=%.2f°C)\n",
               jday, flow_to_inject, extraction_temp, heated_temp, temp_change_value);
    }
    
    // Always update last day info (will be printed in final summary)
    last_hp_jday = jday;
    flow_rate_m3day_equiv = flow_to_inject * SecsPerDay / step_duration_seconds;
    last_hp_flow = flow_rate_m3day_equiv;
    last_hp_extract_temp = extraction_temp;
    last_hp_inject_temp = heated_temp;
    last_hp_delta_t = temp_change_value;
    
    // =========================================================================
    // DIAGNOSTIC OUTPUT - Track heat flux for verification
    // =========================================================================
    // Accumulate energy across ALL timesteps (e.g., 24 hourly steps per day)
    // Each timestep contributes: E_step = ρ × cp × V_step × ΔT
    // flow_to_inject is the volume injected during this timestep (m³)
    AED_REAL energy_this_step_j = rho0 * SPHEAT * flow_to_inject * fabs(temp_change_value);
    
    // Accumulate to daily and cumulative totals
    heat_pump_daily_flux += energy_this_step_j;
    heat_pump_daily_flow += flow_to_inject;
    heat_pump_cumulative_flux += energy_this_step_j;
    
    // Always update temperature values (use latest)
    heat_pump_daily_extract_temp = stored_temp;
    heat_pump_daily_inject_temp = heated_temp;
    
    // Clear stored data after injection to prevent double injection
    stored_flow_rate = 0.0;
}

/***********************************************************
 * Initialize heat pump system (called from glm_init.c)   *
 ***********************************************************/
void init_heat_pump() 
{
    // Heat pump initialization
}

/****************************************************************
 * Check heat pump configuration (called from glm_init.c)      *
 ****************************************************************/
void check_heat_pump_config() 
{
    if (heat_pump_switch > 0) {
        printf("\n=== HEAT PUMP MODULE CONFIGURATION ===\n");
        
        // Display mode-specific configuration
        if (heat_pump_switch == 1) {
            printf("Mode: Fixed ΔT (Mode 1)\n");
            printf("  Temperature change: %.2f°C\n", heat_pump_temp_change);
            printf("  Inflow index: %d, Outflow index: %d\n", 
                   heat_pump_inflow_idx, heat_pump_outflow_idx);
            
            // Validate mode 1 requirements
            if (heat_pump_temp_change == 0.0) {
                printf("  WARNING: heat_pump_temp_change is zero - no heating will occur\n");
            }
        } else if (heat_pump_switch == 2) {
            printf("Mode: Heat Flux (Mode 2)\n");
            printf("  Heat flux: %.0f W\n", heat_pump_heat_flux);
            printf("  Inflow index: %d, Outflow index: %d\n", 
                   heat_pump_inflow_idx, heat_pump_outflow_idx);
            
            // Validate mode 2 requirements  
            if (heat_pump_heat_flux == 0.0) {
                printf("  WARNING: heat_pump_heat_flux is zero - no heating will occur\n");
            }
            if (heat_pump_heat_flux < 0.0) {
                printf("  INFO: heat_pump_heat_flux is negative (%.0f W) - cooling mode\n", heat_pump_heat_flux);
            }
        } else {
            printf("Mode: Unknown (%d) - defaulting to Mode 1 behavior\n", heat_pump_switch);
            printf("  WARNING: Unsupported heat pump mode\n");
        }
        
        // Display physical constraints
        printf("\nPhysical Constraints:\n");
        printf("  Injection temp range: [%.1f°C, %.1f°C]\n", heat_pump_min_temp, heat_pump_max_temp);
        printf("  Maximum |ΔT|: %.1f°C\n", heat_pump_max_delta_t);
        printf("  Maximum flow rate: %.0f m³/day (%.2f m³/s)\n", 
               heat_pump_max_flow, heat_pump_max_flow / 86400.0);
        printf("  Minimum withdrawal temp: %.1f°C\n", heat_pump_min_withdraw_temp);
        printf("  Enforce limits: %s\n", heat_pump_enforce_limits ? "YES (clamp values)" : "NO (warnings only)");
               
        // Validate indices 
        if (heat_pump_inflow_idx < 0 || heat_pump_inflow_idx >= NumInf) {
            printf("\nERROR: heat_pump_inflow_idx (%d) is out of range [0, %d]\n", 
                   heat_pump_inflow_idx, NumInf-1);
        }
        if (heat_pump_outflow_idx < 0 || heat_pump_outflow_idx >= NumOut) {
            printf("ERROR: heat_pump_outflow_idx (%d) is out of range [0, %d]\n", 
                   heat_pump_outflow_idx, NumOut-1);
        }
        
        // Validate constraint ranges
        if (heat_pump_min_temp >= heat_pump_max_temp) {
            printf("ERROR: heat_pump_min_temp (%.1f) >= heat_pump_max_temp (%.1f)\n",
                   heat_pump_min_temp, heat_pump_max_temp);
        }
        if (heat_pump_max_delta_t <= 0) {
            printf("ERROR: heat_pump_max_delta_t must be positive (got %.1f)\n", heat_pump_max_delta_t);
        }
        if (heat_pump_max_flow <= 0) {
            printf("ERROR: heat_pump_max_flow must be positive (got %.1f)\n", heat_pump_max_flow);
        }
        
        // Respect the nml configuration for heat pump inflow (submerged or surface)
        // If subm_flag = .true. in nml, water is injected at subm_elev depth
        // If subm_flag = .false., plunge dynamics find neutral buoyancy level
        if (heat_pump_inflow_idx >= 0 && heat_pump_inflow_idx < NumInf) {
            printf("\nInflow configuration:\n");
            printf("  Inflow %d: SubmFlag = %s, SubmElev = %.1f m (from .nml)\n",
                   heat_pump_inflow_idx,
                   Inflows[heat_pump_inflow_idx].SubmFlag ? "TRUE (submerged injection)" : "FALSE (plunge dynamics)",
                   Inflows[heat_pump_inflow_idx].SubmElev);
        }
        
        // For outflow: respect the nml configuration (don't force dynamic)
        // If you want static extraction, ensure elev_idx_outflow is NOT set in .nml
        if (heat_pump_outflow_idx >= 0 && heat_pump_outflow_idx < NumOut) {
            // Debug: show what was read from nml
            printf("\nOutflow configuration:\n");
            printf("  Outflow %d: SubmElevDynamic = %s (from .nml)\n",
                   heat_pump_outflow_idx,
                   Outflows[heat_pump_outflow_idx].SubmElevDynamic ? "TRUE (dynamic depth)" : "FALSE (static depth)");
        }
        
        printf("=======================================\n\n");
    } else {
        printf("Heat pump module disabled (heat_pump_switch = %d)\n", heat_pump_switch);
    }
}

/****************************************************************
 * Report constraint violation statistics at end of simulation  *
 ****************************************************************/
void heat_pump_report_stats()
{
    // Print final day's constraint summary
    if (last_summary_jday != -1) {
        print_constraint_summary(last_summary_jday);
    }
    
    if (heat_pump_switch > 0 && first_hp_jday != -1) {
        printf("\n=== HEAT PUMP OPERATION SUMMARY ===\n");
        printf("First day (jday %d): Q=%.1f m³/d, T_extract=%.1f°C → T_inject=%.1f°C (ΔT=%.2f°C)\n",
               first_hp_jday, first_hp_flow, first_hp_extract_temp, first_hp_inject_temp, first_hp_delta_t);
        printf("Last day  (jday %d): Q=%.1f m³/d, T_extract=%.1f°C → T_inject=%.1f°C (ΔT=%.2f°C)\n",
               last_hp_jday, last_hp_flow, last_hp_extract_temp, last_hp_inject_temp, last_hp_delta_t);
    }
    
    int total = total_clamp_freeze + total_clamp_hot + total_clamp_delta_t + 
                total_clamp_flow + total_skip_cold;
    
    if (heat_pump_switch > 0 && total > 0) {
        printf("\n--- Constraint Events ---\n");
        printf("Total: %d\n", total);
        if (total_clamp_freeze > 0)
            printf("  Clamped (freezing risk):    %d timesteps\n", total_clamp_freeze);
        if (total_clamp_hot > 0)
            printf("  Clamped (too hot):          %d timesteps\n", total_clamp_hot);
        if (total_clamp_delta_t > 0)
            printf("  Clamped (ΔT exceeded):      %d timesteps\n", total_clamp_delta_t);
        if (total_clamp_flow > 0)
            printf("  Clamped (flow exceeded):    %d timesteps\n", total_clamp_flow);
        if (total_skip_cold > 0)
            printf("  Skipped (intake too cold):  %d timesteps\n", total_skip_cold);
    }
    
    if (heat_pump_switch > 0 && first_hp_jday != -1) {
        printf("===================================\n");
    }
}