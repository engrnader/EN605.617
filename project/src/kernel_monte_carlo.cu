/**
 * @file    kernel_monte_carlo.cu
 * @brief   Kernel 3: Monte Carlo robustness analysis.
 *
 *          Each GPU thread runs one drift scenario:
 *            1. cuRAND samples drift rate + per-step noise
 *            2. Proportional controller loop for N iterations
 *            3. Record peak deviation and pass/fail
 *
 *          Thrust aggregation:
 *            count_if   -> pass/fail ratio
 *            reduce     -> mean deviation
 *            max_element-> worst case
 *            sort       -> P95, P99 percentiles
 */

#include <curand_kernel.h>
#include <math.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "kernel_monte_carlo.h"

/* ---------------------------------------------------------------
 * Device kernels
 * --------------------------------------------------------------- */

/**
 * @brief  Initialize one cuRAND state per thread.
 */
__global__ void kern_init_curand(curandState* states, unsigned long long seed,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * @brief  Monte Carlo control loop simulation.
 *
 *         Each thread: sample drift, run loop, record result.
 */
__global__ void kern_monte_carlo(curandState* states, float* max_devs,
                                 int* pass_fail, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    curandState local_state = states[idx];

    float drift =
        curand_uniform(&local_state) * 2.0f * MC_DRIFT_MAX - MC_DRIFT_MAX;

    float deviation = 0.0f;
    float peak_dev  = 0.0f;
    int   passed    = 1;

    for (int step = 0; step < MC_LOOP_ITERS; step++) {
        float noise      = curand_normal(&local_state) * MC_NOISE_STDDEV;
        float correction = MC_CTRL_GAIN * deviation;
        deviation += drift + noise - correction;

        float abs_dev = fabsf(deviation);
        if (abs_dev > peak_dev)
            peak_dev = abs_dev;
        if (abs_dev > MC_FAIL_THRESHOLD)
            passed = 0;
    }

    max_devs[idx]  = peak_dev;
    pass_fail[idx] = passed;
    states[idx]    = local_state;
}

/* ---------------------------------------------------------------
 * Thrust functor
 * --------------------------------------------------------------- */

struct is_pass {
    __host__ __device__ bool operator()(int x) const {
        return x == 1;
    }
};

/* ---------------------------------------------------------------
 * Thrust aggregation
 * --------------------------------------------------------------- */

/**
 * @brief  Compute statistics from simulation results.
 */
static void thrust_aggregate(float* d_max_devs, int* d_pass_fail, int n,
                             MonteCarloResult* result) {
    thrust::device_ptr<int>   t_pf(d_pass_fail);
    thrust::device_ptr<float> t_dev(d_max_devs);

    result->pass_count = thrust::count_if(t_pf, t_pf + n, is_pass());

    float total = thrust::reduce(t_dev, t_dev + n, 0.0f, thrust::plus<float>());
    result->avg_dev = total / (float)n;

    thrust::device_ptr<float> max_ptr = thrust::max_element(t_dev, t_dev + n);
    result->worst_dev                 = *max_ptr;

    thrust::sort(t_dev, t_dev + n);
    int idx_95      = (int)(0.95f * (float)n);
    int idx_99      = (int)(0.99f * (float)n);
    result->p95_dev = *(t_dev + idx_95);
    result->p99_dev = *(t_dev + idx_99);
}

/* ---------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------- */

void monte_carlo_pipeline(int num_scenarios, int block_size, int num_blocks,
                          unsigned long long seed, cudaStream_t stream,
                          MonteCarloResult* result) {
    int    n           = num_scenarios;
    size_t state_bytes = (size_t)n * sizeof(curandState);
    size_t float_bytes = (size_t)n * sizeof(float);
    size_t int_bytes   = (size_t)n * sizeof(int);

    curandState* d_states;
    float*       d_max_devs;
    int*         d_pass_fail;
    CUDA_CHECK(cudaMalloc(&d_states, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_max_devs, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_pass_fail, int_bytes));

    kern_init_curand<<<num_blocks, block_size, 0, stream>>>(d_states, seed, n);

    kern_monte_carlo<<<num_blocks, block_size, 0, stream>>>(
        d_states, d_max_devs, d_pass_fail, n);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    thrust_aggregate(d_max_devs, d_pass_fail, n, result);

    result->num_scenarios = n;
    result->pass_pct      = 100.0f * (float)result->pass_count / (float)n;

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_max_devs));
    CUDA_CHECK(cudaFree(d_pass_fail));
}
