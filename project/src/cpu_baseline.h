/**
 * @file    cpu_baseline.h
 * @brief   Sequential CPU implementations of all three pipeline
 *          stages for timing comparison.
 */

#ifndef CPU_BASELINE_H
#define CPU_BASELINE_H

#include "common.h"

/**
 * @brief  CPU sweep: generate response, smooth, find peak.
 */
void cpu_sweep(int n, float center_frac, float width_frac, int noise_seed,
               SweepResult* result);

/**
 * @brief  CPU gradient: compute margin scores, select best.
 */
void cpu_gradient(const float* smoothed, int n, OpPointResult* result);

/**
 * @brief  CPU Monte Carlo: sequential drift simulation.
 */
void cpu_monte_carlo(int num_scenarios, unsigned int base_seed,
                     MonteCarloResult* result);

#endif /* CPU_BASELINE_H */
