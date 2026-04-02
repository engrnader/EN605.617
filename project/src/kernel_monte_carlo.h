/**
 * @file    kernel_monte_carlo.h
 * @brief   Kernel 3: Monte Carlo robustness analysis with
 *          cuRAND and Thrust.
 */

#ifndef KERNEL_MONTE_CARLO_H
#define KERNEL_MONTE_CARLO_H

#include "common.h"

/**
 * @brief  Run Monte Carlo robustness simulation.
 *
 *         Each GPU thread simulates one drift scenario:
 *         sample random drift + noise with cuRAND, run the
 *         control loop, record pass/fail. Thrust aggregates.
 *
 * @param  num_scenarios  Total scenarios to simulate
 * @param  block_size     Threads per block
 * @param  num_blocks     Number of blocks
 * @param  seed           cuRAND base seed
 * @param  stream         CUDA stream
 * @param  result         Output: pass rate, percentiles, etc.
 */
void monte_carlo_pipeline(int num_scenarios, int block_size, int num_blocks,
                          unsigned long long seed, cudaStream_t stream,
                          MonteCarloResult* result);

#endif /* KERNEL_MONTE_CARLO_H */
