/**
 * @file    kernel_gradient.h
 * @brief   Kernel 2: Gradient analysis, margin scoring, and
 *          operating point selection.
 */

#ifndef KERNEL_GRADIENT_H
#define KERNEL_GRADIENT_H

#include "common.h"

/**
 * @brief  Compute gradient and margin scores, then select
 *         the best operating point for one channel.
 *
 * @param  d_smoothed   Smoothed response from Kernel 1 [n]
 * @param  n            Number of sweep steps
 * @param  block_size   Threads per block
 * @param  num_blocks   Number of blocks
 * @param  stream       CUDA stream
 * @param  result       Output: best index, margin, gradient
 */
void gradient_pipeline(const float* d_smoothed, int n, int block_size,
                       int num_blocks, cudaStream_t stream,
                       OpPointResult* result);

#endif /* KERNEL_GRADIENT_H */
