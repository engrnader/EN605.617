/**
 * @file    monte_carlo.cu
 * @brief   CUDA Advanced Libraries - Monte Carlo Robustness Analysis
 *
 *          Simulates closed-loop controller robustness under random
 *          environmental drift. Each GPU thread runs one complete
 *          drift scenario: sample random parameters with cuRAND,
 *          simulate a control loop, and record pass/fail.
 *
 *          Libraries demonstrated:
 *            cuRAND  - per-thread RNG (device API)
 *            Thrust  - count_if, reduce, max_element, sort
 *
 * @usage   ./monte_carlo <num_scenarios> <block_size>
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <chrono>

/* ---------------------------------------------------------------
 * Constants
 * --------------------------------------------------------------- */
#define LOOP_ITERATIONS 200  /* control loop steps per scenario */
#define DRIFT_RATE_MAX 0.03f /* max abs drift rate per step     */
#define NOISE_STDDEV 0.15f   /* Gaussian noise standard dev     */
#define FAIL_THRESHOLD 3.0f  /* deviation that causes lock loss */
#define CONTROLLER_GAIN 0.3f /* proportional correction gain    */
#define NUM_RUNS 3           /* timed runs for benchmarking     */

/* ---------------------------------------------------------------
 * Error checking macro
 * --------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

/* ---------------------------------------------------------------
 * Kernels
 * --------------------------------------------------------------- */

/**
 * @brief  Initialize one cuRAND state per thread.
 *
 * @param  states      Output array of RNG states [n]
 * @param  seed        Base seed for reproducibility
 * @param  n           Number of states to initialize
 */
__global__ void kernel_init_curand(curandState* states, unsigned long long seed,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    /* each thread gets a unique sequence number */
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * @brief  Monte Carlo control loop simulation.
 *
 *         Each thread samples a random drift rate and runs
 *         LOOP_ITERATIONS steps of a proportional controller.
 *         At each step: deviation += drift + noise - correction.
 *         Records max absolute deviation and pass/fail.
 *
 * @param  states       cuRAND states (read/write) [n]
 * @param  max_devs     Output: peak |deviation| per thread [n]
 * @param  pass_fail    Output: 1 = pass, 0 = fail [n]
 * @param  n            Number of scenarios
 */
__global__ void kernel_monte_carlo(curandState* states, float* max_devs,
                                   int* pass_fail, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    curandState local_state = states[idx];

    /* sample drift rate: uniform in [-DRIFT_RATE_MAX, +max] */
    float drift =
        curand_uniform(&local_state) * 2.0f * DRIFT_RATE_MAX - DRIFT_RATE_MAX;

    float deviation = 0.0f;
    float peak_dev  = 0.0f;
    int   passed    = 1;

    for (int step = 0; step < LOOP_ITERATIONS; step++) {
        float noise      = curand_normal(&local_state) * NOISE_STDDEV;
        float correction = CONTROLLER_GAIN * deviation;
        deviation += drift + noise - correction;

        float abs_dev = fabsf(deviation);
        if (abs_dev > peak_dev)
            peak_dev = abs_dev;
        if (abs_dev > FAIL_THRESHOLD)
            passed = 0;
    }

    max_devs[idx]  = peak_dev;
    pass_fail[idx] = passed;
    states[idx]    = local_state;
}

/* ---------------------------------------------------------------
 * Thrust functors
 * --------------------------------------------------------------- */

/** @brief Predicate: returns true if scenario passed. */
struct is_pass {
    __host__ __device__ bool operator()(int x) const {
        return x == 1;
    }
};

/* ---------------------------------------------------------------
 * GPU pipeline
 * --------------------------------------------------------------- */

/**
 * @brief  Allocate device arrays for the simulation.
 */
static void alloc_device(curandState** d_states, float** d_max_devs,
                         int** d_pass_fail, int n) {
    size_t state_bytes = (size_t)n * sizeof(curandState);
    size_t float_bytes = (size_t)n * sizeof(float);
    size_t int_bytes   = (size_t)n * sizeof(int);
    CUDA_CHECK(cudaMalloc(d_states, state_bytes));
    CUDA_CHECK(cudaMalloc(d_max_devs, float_bytes));
    CUDA_CHECK(cudaMalloc(d_pass_fail, int_bytes));
}

/**
 * @brief  Free device arrays.
 */
static void free_device(curandState* d_states, float* d_max_devs,
                        int* d_pass_fail) {
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_max_devs));
    CUDA_CHECK(cudaFree(d_pass_fail));
}

/**
 * @brief  Run Thrust aggregation on GPU results.
 *
 * @param  d_max_devs   Device array of peak deviations [n]
 * @param  d_pass_fail  Device array of pass/fail flags [n]
 * @param  n            Number of scenarios
 * @param  pass_count   Output: number of passing scenarios
 * @param  avg_dev      Output: mean peak deviation
 * @param  worst_dev    Output: max peak deviation
 * @param  p95_dev      Output: 95th percentile deviation
 * @param  p99_dev      Output: 99th percentile deviation
 */
static void thrust_aggregate(float* d_max_devs, int* d_pass_fail, int n,
                             int* pass_count, float* avg_dev, float* worst_dev,
                             float* p95_dev, float* p99_dev) {
    thrust::device_ptr<int>   t_pf(d_pass_fail);
    thrust::device_ptr<float> t_dev(d_max_devs);

    *pass_count = thrust::count_if(t_pf, t_pf + n, is_pass());

    float total = thrust::reduce(t_dev, t_dev + n, 0.0f, thrust::plus<float>());
    *avg_dev    = total / (float)n;

    thrust::device_ptr<float> max_ptr = thrust::max_element(t_dev, t_dev + n);
    *worst_dev                        = *max_ptr;

    /* sort for percentile analysis */
    thrust::sort(t_dev, t_dev + n);
    int idx_95 = (int)(0.95f * (float)n);
    int idx_99 = (int)(0.99f * (float)n);
    *p95_dev   = *(t_dev + idx_95);
    *p99_dev   = *(t_dev + idx_99);
}

/**
 * @brief  Run the full GPU Monte Carlo pipeline.
 */
static void run_gpu(int n, int block_size, int num_blocks, int run_idx) {
    curandState* d_states;
    float*       d_max_devs;
    int*         d_pass_fail;
    alloc_device(&d_states, &d_max_devs, &d_pass_fail, n);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    unsigned long long seed = 42ULL + (unsigned long long)run_idx;

    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    kernel_init_curand<<<num_blocks, block_size>>>(d_states, seed, n);

    kernel_monte_carlo<<<num_blocks, block_size>>>(d_states, d_max_devs,
                                                   d_pass_fail, n);

    int   pass_count;
    float avg_dev, worst_dev, p95_dev, p99_dev;
    thrust_aggregate(d_max_devs, d_pass_fail, n, &pass_count, &avg_dev,
                     &worst_dev, &p95_dev, &p99_dev);

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    float pass_pct = 100.0f * (float)pass_count / (float)n;
    printf(
        "  GPU Run %-2d | %8.3f ms | "
        "Pass: %6.2f%% | Avg: %5.3f | "
        "P95: %5.3f | P99: %5.3f | "
        "Worst: %5.3f\n",
        run_idx, elapsed_ms, pass_pct, avg_dev, p95_dev, p99_dev, worst_dev);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    free_device(d_states, d_max_devs, d_pass_fail);
}

/* ---------------------------------------------------------------
 * CPU baseline
 * --------------------------------------------------------------- */

/**
 * @brief  Simulate one scenario on the CPU.
 */
static float cpu_simulate_one(float drift, float* max_dev_out,
                              unsigned int* rng_state) {
    float deviation = 0.0f;
    float peak_dev  = 0.0f;

    for (int step = 0; step < LOOP_ITERATIONS; step++) {
        /* simple LCG-based normal approximation */
        float u1 = 0.0f, u2 = 0.0f;
        *rng_state = (*rng_state) * 1664525u + 1013904223u;
        u1         = (float)(*rng_state) / 4294967296.0f;
        *rng_state = (*rng_state) * 1664525u + 1013904223u;
        u2         = (float)(*rng_state) / 4294967296.0f;
        /* Box-Muller transform */
        if (u1 < 1e-10f)
            u1 = 1e-10f;
        float noise =
            sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2) * NOISE_STDDEV;

        float correction = CONTROLLER_GAIN * deviation;
        deviation += drift + noise - correction;

        float abs_dev = fabsf(deviation);
        if (abs_dev > peak_dev)
            peak_dev = abs_dev;
    }
    *max_dev_out = peak_dev;
    return (peak_dev <= FAIL_THRESHOLD) ? 1.0f : 0.0f;
}

/**
 * @brief  Run the full CPU baseline sequentially.
 */
static void run_cpu(int n, int run_idx) {
    auto t0 = std::chrono::steady_clock::now();

    int    pass_count = 0;
    float  total_dev  = 0.0f;
    float  worst_dev  = 0.0f;
    float* all_devs   = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        unsigned int rng = (unsigned int)(i + run_idx * n);
        float        drift =
            ((float)(rng % 10000) / 10000.0f) * 2.0f * DRIFT_RATE_MAX -
            DRIFT_RATE_MAX;
        float max_dev;
        float passed = cpu_simulate_one(drift, &max_dev, &rng);
        pass_count += (int)passed;
        total_dev += max_dev;
        all_devs[i] = max_dev;
        if (max_dev > worst_dev)
            worst_dev = max_dev;
    }

    auto   t1 = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    float avg_dev  = total_dev / (float)n;
    float pass_pct = 100.0f * (float)pass_count / (float)n;

    printf(
        "  CPU Run %-2d | %8.3f ms | "
        "Pass: %6.2f%% | Avg: %5.3f | "
        "Worst: %5.3f\n",
        run_idx, elapsed_ms, pass_pct, avg_dev, worst_dev);

    free(all_devs);
}

/* ---------------------------------------------------------------
 * Main
 * --------------------------------------------------------------- */

/**
 * @brief  Parse command line arguments.
 */
static void parse_args(int argc, char** argv, int* n, int* block_size) {
    *n          = (1 << 16); /* default: 65536 scenarios */
    *block_size = 256;

    if (argc >= 2)
        *n = atoi(argv[1]);
    if (argc >= 3)
        *block_size = atoi(argv[2]);
}

/**
 * @brief  Validate and adjust launch parameters.
 *
 * @return Number of blocks, or -1 on error.
 */
static int validate_params(int* n, int block_size) {
    if (block_size <= 0 || block_size > 1024) {
        fprintf(stderr, "Error: block_size must be in [1, 1024]\n");
        return -1;
    }
    if (block_size % 32 != 0) {
        printf(
            "Warning: block_size %d is not a "
            "multiple of 32 (warp size)\n",
            block_size);
    }
    int num_blocks = *n / block_size;
    if (*n % block_size != 0) {
        num_blocks++;
        *n = num_blocks * block_size;
        printf("Warning: scenarios rounded up to %d\n", *n);
    }
    return num_blocks;
}

/**
 * @brief  Print device info and simulation parameters.
 */
static void print_config(int n, int block_size, int num_blocks) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Device            : %s\n", prop.name);
    printf("\n");
    printf("Monte Carlo Robustness Analysis\n");
    printf("  Scenarios       : %d\n", n);
    printf("  Block size      : %d\n", block_size);
    printf("  Num blocks      : %d\n", num_blocks);
    printf("  Loop iterations : %d\n", LOOP_ITERATIONS);
    printf("  Fail threshold  : %.1f\n", FAIL_THRESHOLD);
    printf("  Controller gain : %.2f\n", CONTROLLER_GAIN);
    printf("  Runs            : %d\n", NUM_RUNS);
    printf("\n");
}

int main(int argc, char** argv) {
    int n, block_size;
    parse_args(argc, argv, &n, &block_size);

    int num_blocks = validate_params(&n, block_size);
    if (num_blocks < 0)
        return EXIT_FAILURE;

    print_config(n, block_size, num_blocks);

    printf("--- GPU (cuRAND + Thrust) ---\n");
    for (int run = 1; run <= NUM_RUNS; run++)
        run_gpu(n, block_size, num_blocks, run);

    printf("\n--- CPU Baseline ---\n");
    for (int run = 1; run <= NUM_RUNS; run++) run_cpu(n, run);

    printf("\n");
    return EXIT_SUCCESS;
}
