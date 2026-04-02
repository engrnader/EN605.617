/**
 * @file    kernel_sweep.h
 * @brief   Kernel 1: Parallel sweep, smoothing, and peak detection.
 */

#ifndef KERNEL_SWEEP_H
#define KERNEL_SWEEP_H

#include "common.h"

/**
 * @brief  Run the sweep pipeline for one channel.
 *
 * Generates a peak-shaped response, smooths it, and finds
 * the peak via parallel reduction. Can run on a user-supplied
 * CUDA stream for multi-channel concurrency.
 *
 * @param  d_raw        Device buffer for raw response [n]
 * @param  d_smoothed   Device buffer for smoothed data [n]
 * @param  n            Number of sweep steps
 * @param  block_size   Threads per block
 * @param  num_blocks   Number of blocks
 * @param  center_frac  Resonance center as fraction of n
 * @param  width_frac   Half-width as fraction of n
 * @param  noise_seed   Seed for deterministic noise
 * @param  stream       CUDA stream (0 for default)
 * @param  result       Output: peak index and value
 */
void sweep_pipeline(float* d_raw, float* d_smoothed, int n, int block_size,
                    int num_blocks, float center_frac, float width_frac,
                    int noise_seed, cudaStream_t stream, SweepResult* result);

#endif /* KERNEL_SWEEP_H */
