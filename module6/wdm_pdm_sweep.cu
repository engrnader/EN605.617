/**
 * @file    wdm_pdm_sweep.cu
 * @brief   CUDA Streams and Events - WDM Ring Resonator PDM Sweep Pipeline
 *
 *          Simulates the operating point search for a 4-channel silicon
 * photonic WDM link. Each ring resonator is thermally tuned by a PDM heater.
 *          Sweeping the PDM setpoint while reading the monitor photodetector
 * (MPD) current reveals a Lorentzian resonance peak - the operating point.
 *
 *          Three-stage pipeline per channel:
 *            Stage 1 - kernel_generate_response : Lorentzian response + noise
 *            Stage 2 - kernel_smooth_response   : uniform moving average
 * stencil Stage 3 - kernel_find_peak         : parallel reduction (max search)
 *
 *          Two modes benchmarked:
 *            WITH STREAMS    : all 4 channels run concurrently via CUDA streams
 *            WITHOUT STREAMS : all 4 channels run sequentially (baseline)
 *
 * @usage   ./wdm_pdm_sweep <total_threads> <block_size>
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Constants
 */
#define NUM_STREAMS 4         /* one stream per WDM ring resonator channel */
#define SMOOTH_RADIUS 8       /* moving average half-windows in samples */
#define NUM_RUNS 2            /* timed runs per mode (rbric requirement) */
#define PEAK_AMPLITUDE 1.0f   /* Lorentzian peak amplitude (normalized) */
#define NOISE_AMPLITUDE 0.04f /* simulated MPD measurement noise amplitude */

/**
 * Ring resonator parameters as fractions of the total sweep range.
 * RR_CENTER_FRAC      - resonant PDM setpoint / num_steps
 * RR_HALFWIDTH_FRAC   - Lorentzian half-width at half-maximum / num_steps
 *
 * @note: The 4 values in RR_CENTER_FRAC represent where wach of the 4 ring
 * resonators has its resonance peak in the sweep range. Evenly distributed
 * across the PDM range. One ring per WDM channel.
 */
static const float RR_CENTER_FRAC[NUM_STREAMS] = {0.15f, 0.38f, 0.62f, 0.82f};
static const float RR_HALFWIDTH_FRAC[NUM_STREAMS] = {0.018f, 0.022f, 0.020f,
                                                     0.024f};

/**
 * Error Checking
 */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0);

/**
 * Kernels
 */
/**
 * @brief  Generate the Lorentzian resonance response for a PDM sweep.
 *
 *         Models the MPD current as a function of heater setpoint:
 *           L(x) = A / (1 + ((x - x0) / gamma)^2) + noise
 *
 *         One thread per PDM step; all steps evaluated in parallel.
 *
 * @param  response   Output array [num_steps]
 * @param  num_steps  Total PDM sweep steps
 * @param  x0         Resonant setpoint as a fraction of num_steps
 * @param  gamma      Half-width at half-maximum as a fraction of num_steps
 * @param  noise_seed Seed for deterministic per-run noise variation
 */

__global__ void kernel_generate_response(float* response, int num_steps,
                                         float x0, float gamma,
                                         int noise_seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_steps)
        return;

    /* normalized to [0,1] */
    float x = (float)idx / (float)num_steps;
    /* how far the current step is from the resonant center, divided by the
     * half-width */
    float delta = (x - x0) / gamma;
    /* when delta = 0 (exactly at resonance), signal = PEAK_AMPLITUDE / 1.0
     * = 1.0 */
    float signal = PEAK_AMPLITUDE / (1.0f + delta * delta);
    /* feed in different noise each time */
    float noise = NOISE_AMPLITUDE * sinf((float)(idx * noise_seed) * 0.037f);

    response[idx] = signal + noise;
}

/**
 * @brief  Smooth the resonance response curve with a uniform moving average.
 *
 *         Reduces MPD measurement noise before peak detection. Each output
 *         sample is the mean of the (2 * radius + 1) nearest input samples.
 *         Boundary samples use a reduced window with no zero-padding.
 *
 * @param  smoothed   Output smoothed array [num_steps]
 * @param  raw        Input raw response array [num_steps]
 * @param  num_steps  Array length
 * @param  radius     Stencil half-width in samples
 */
__global__ void kernel_smooth_response(float* smoothed, const float* raw,
                                       int num_steps, int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_steps)
        return;

    float sum   = 0.0f;
    int   count = 0;

    for (int k = -radius; k <= radius; k++) {
        int j = idx + k;
        if (j >= 0 && j < num_steps) {
            sum += raw[j];
            count++;
        }
    }
    smoothed[idx] = sum / (float)count;
}

/**
 * @brief  Parallel reduction to find the maximum value and its PDM index.
 *
 *         Each block reduces its segment to one (value, index) pair written
 *         to d_block_vals / d_block_idxs. The CPU completes the final
 *         reduction across the small number of block results.
 *
 *         Shared memory layout: [ float[blockDim.x] | int[blockDim.x] ]
 *
 * @param  d_block_vals  Per-block maximum values  [num_blocks]
 * @param  d_block_idxs  Per-block maximum indices [num_blocks]
 * @param  data          Input smoothed response   [num_steps]
 * @param  num_steps     Array length
 */
__global__ void kernel_find_peak(float* d_block_vals, int* d_block_idxs,
                                 const float* data, int num_steps) {
    /* decleares synamic shared memory. */
    extern __shared__ float s_vals[];
    /* floats for values and ints for indices. Cast a pointer offset into the
     * same shared memory block */
    int* s_idxs = (int*)(s_vals + blockDim.x);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* load into shared memory; out-of-bounds lanes get sentinel value */
    s_vals[tid] = (idx < num_steps) ? data[idx] : -1.0f;
    s_idxs[tid] = (idx < num_steps) ? idx : -1;
    __syncthreads();

    /* tree reduction: keep the larger (value, index) pair at each step */
    /* Half the active threads with each iteration until thread 0 holds the
     * block's maximum */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_vals[tid + stride] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + stride];
            s_idxs[tid] = s_idxs[tid + stride];
        }
        __syncthreads();
    }

    /* thread 0 writes this block's winner to global memory */
    if (tid == 0) {
        d_block_vals[blockIdx.x] = s_vals[0];
        d_block_idxs[blockIdx.x] = s_idxs[0];
    }
}

/**
 * Host Helpers
 */

/**
 * @brief Final CPU-side reduction over per-block peak results.
 */
static void find_global_peak(const float* block_vals, const int* block_idxs,
                             int num_blocks, float* out_val, int* out_idx) {
    *out_val = -1.0f;
    *out_idx = -1;

    for (int i = 0; i < num_blocks; i++) {
        if (block_vals[i] > *out_val) {
            *out_val = block_vals[i];
            *out_idx = block_idxs[i];
        }
    }
}

/**
 * @brief  Print the unified results table header. Called once before all runs.
 */
static void print_results_header(void) {
    printf("%-4s  %-7s  %-7s  %-6s  %-9s  %-5s  %-9s  %-11s  %-13s  %-10s\n",
           "Run", "Streams", "Threads", "Blocks", "Steps", "Ring", "Peak PDM",
           "Peak Val", "Expect PDM", "Time(ms)");
    printf("%s\n",
           "-------------------------------------------------------------------"
           "----------------");
}

/* ***************************************************************************
 * Pipeline: WITHOUT STREAMS (sequential baseline)
 * ***************************************************************************/

/**
 * @brief  Run the 4-channel pipeline sequentially on the default stream.
 *
 *         Rings are processed one at a time. Serves as a timing baseline to
 *         quantify the speedup gained from concurrent stream execution.
 *
 * @param  num_steps          Total PDM sweep steps per channel
 * @param  threads_per_block  Threads per CUDA block
 * @param  num_blocks         Number of CUDA blocks
 * @param  run_idx            Run index used to vary noise seed
 */
static void run_without_streams(int num_steps, int threads_per_block,
                                int num_blocks, int run_idx) {
    size_t response_bytes  = (size_t)num_steps * sizeof(float);
    size_t block_val_bytes = (size_t)num_blocks * sizeof(float);
    size_t block_idx_bytes = (size_t)num_blocks * sizeof(int);
    size_t shared_peak_bytes =
        (size_t)threads_per_block * (sizeof(float) + sizeof(int));

    float* h_peak_vals[NUM_STREAMS];
    int*   h_peak_idxs[NUM_STREAMS];
    float* d_raw[NUM_STREAMS];
    float* d_smoothed[NUM_STREAMS];
    float* d_block_vals[NUM_STREAMS];
    int*   d_block_idxs[NUM_STREAMS];

    cudaEvent_t ev_start, ev_stop;

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaMallocHost(&h_peak_vals[s], block_val_bytes));
        CUDA_CHECK(cudaMallocHost(&h_peak_idxs[s], block_idx_bytes));
        CUDA_CHECK(cudaMalloc(&d_raw[s], response_bytes));
        CUDA_CHECK(cudaMalloc(&d_smoothed[s], response_bytes));
        CUDA_CHECK(cudaMalloc(&d_block_vals[s], block_val_bytes));
        CUDA_CHECK(cudaMalloc(&d_block_idxs[s], block_idx_bytes));
    }

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    /* process each ring sequentially on the default stream */
    for (int s = 0; s < NUM_STREAMS; s++) {
        int seed = (run_idx * NUM_STREAMS) + s + 1;

        kernel_generate_response<<<num_blocks, threads_per_block>>>(
            d_raw[s], num_steps, RR_CENTER_FRAC[s], RR_HALFWIDTH_FRAC[s], seed);

        kernel_smooth_response<<<num_blocks, threads_per_block>>>(
            d_smoothed[s], d_raw[s], num_steps, SMOOTH_RADIUS);

        kernel_find_peak<<<num_blocks, threads_per_block, shared_peak_bytes>>>(
            d_block_vals[s], d_block_idxs[s], d_smoothed[s], num_steps);

        CUDA_CHECK(cudaMemcpy(h_peak_vals[s], d_block_vals[s], block_val_bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_peak_idxs[s], d_block_idxs[s], block_idx_bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    for (int s = 0; s < NUM_STREAMS; s++) {
        float peak_val;
        int   peak_idx;
        find_global_peak(h_peak_vals[s], h_peak_idxs[s], num_blocks, &peak_val,
                         &peak_idx);
        int expected_pdm = (int)(RR_CENTER_FRAC[s] * (float)num_steps);
        printf(
            "%-4d  %-7s  %-7d  %-6d  %-9d  %-5d  %-9d  %-11.4f  %-13d  "
            "%-10.4f\n",
            run_idx, "w/o", threads_per_block, num_blocks, num_steps, s,
            peak_idx, peak_val, expected_pdm, elapsed_ms);
    }

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaFreeHost(h_peak_vals[s]));
        CUDA_CHECK(cudaFreeHost(h_peak_idxs[s]));
        CUDA_CHECK(cudaFree(d_raw[s]));
        CUDA_CHECK(cudaFree(d_smoothed[s]));
        CUDA_CHECK(cudaFree(d_block_vals[s]));
        CUDA_CHECK(cudaFree(d_block_idxs[s]));
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

/* ***************************************************************************
 * Pipeline: WITH STREAMS (concurrent, pipelined)
 * ***************************************************************************/

/**
 * @brief  Run the 4-channel PDM sweep pipeline using NUM_STREAMS CUDA streams.
 *
 *         All four channels are processed concurrently. Within each stream,
 *         cudaEvents gate the three stages in order. Async D2H transfers
 * overlap with compute on other streams. Overall time measured with cudaEvents.
 *
 * @param  num_steps          Total PDM sweep steps per channel
 * @param  threads_per_block  Threads per CUDA block
 * @param  num_blocks         Number of CUDA blocks
 * @param  run_idx            Run index used to vary noise seed
 */
static void run_with_streams(int num_steps, int threads_per_block,
                             int num_blocks, int run_idx) {
    size_t response_bytes  = (size_t)num_steps * sizeof(float);
    size_t block_val_bytes = (size_t)num_blocks * sizeof(float);
    size_t block_idx_bytes = (size_t)num_blocks * sizeof(int);
    size_t shared_peak_bytes =
        (size_t)threads_per_block * (sizeof(float) + sizeof(int));

    float* h_peak_vals[NUM_STREAMS];
    int*   h_peak_idxs[NUM_STREAMS];
    float* d_raw[NUM_STREAMS];
    float* d_smoothed[NUM_STREAMS];
    float* d_block_vals[NUM_STREAMS];
    int*   d_block_idxs[NUM_STREAMS];

    cudaStream_t stream[NUM_STREAMS];
    cudaEvent_t  ev_gen_done[NUM_STREAMS];
    cudaEvent_t  ev_smooth_done[NUM_STREAMS];
    cudaEvent_t  ev_start, ev_stop;

    /* pinned host memory enables asynchronous DMA transfers */
    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaMallocHost(&h_peak_vals[s], block_val_bytes));
        CUDA_CHECK(cudaMallocHost(&h_peak_idxs[s], block_idx_bytes));
        CUDA_CHECK(cudaMalloc(&d_raw[s], response_bytes));
        CUDA_CHECK(cudaMalloc(&d_smoothed[s], response_bytes));
        CUDA_CHECK(cudaMalloc(&d_block_vals[s], block_val_bytes));
        CUDA_CHECK(cudaMalloc(&d_block_idxs[s], block_idx_bytes));
        CUDA_CHECK(cudaStreamCreate(&stream[s]));
        CUDA_CHECK(cudaEventCreate(&ev_gen_done[s]));
        CUDA_CHECK(cudaEventCreate(&ev_smooth_done[s]));
    }
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    /* queue all 4 channel pipelines - GPU overlaps their execution */
    for (int s = 0; s < NUM_STREAMS; s++) {
        int seed = (run_idx * NUM_STREAMS) + s + 1;

        /* stage 1: evaluate Lorentzian response for every PDM step */
        kernel_generate_response<<<num_blocks, threads_per_block, 0,
                                   stream[s]>>>(
            d_raw[s], num_steps, RR_CENTER_FRAC[s], RR_HALFWIDTH_FRAC[s], seed);
        CUDA_CHECK(cudaEventRecord(ev_gen_done[s], stream[s]));

        /* stage 2: smooth - starts only after generation event fires */
        CUDA_CHECK(cudaStreamWaitEvent(stream[s], ev_gen_done[s], 0));
        kernel_smooth_response<<<num_blocks, threads_per_block, 0, stream[s]>>>(
            d_smoothed[s], d_raw[s], num_steps, SMOOTH_RADIUS);
        CUDA_CHECK(cudaEventRecord(ev_smooth_done[s], stream[s]));

        /* stage 3: find peak - starts only after smooth event fires */
        CUDA_CHECK(cudaStreamWaitEvent(stream[s], ev_smooth_done[s], 0));
        kernel_find_peak<<<num_blocks, threads_per_block, shared_peak_bytes,
                           stream[s]>>>(d_block_vals[s], d_block_idxs[s],
                                        d_smoothed[s], num_steps);

        /* async D2H: overlaps with compute on other streams */
        CUDA_CHECK(cudaMemcpyAsync(h_peak_vals[s], d_block_vals[s],
                                   block_val_bytes, cudaMemcpyDeviceToHost,
                                   stream[s]));
        CUDA_CHECK(cudaMemcpyAsync(h_peak_idxs[s], d_block_idxs[s],
                                   block_idx_bytes, cudaMemcpyDeviceToHost,
                                   stream[s]));
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaStreamSynchronize(stream[s]));
        float peak_val;
        int   peak_idx;
        find_global_peak(h_peak_vals[s], h_peak_idxs[s], num_blocks, &peak_val,
                         &peak_idx);
        int expected_pdm = (int)(RR_CENTER_FRAC[s] * (float)num_steps);
        printf(
            "%-4d  %-7s  %-7d  %-6d  %-9d  %-5d  %-9d  %-11.4f  %-13d  "
            "%-10.4f\n",
            run_idx, "w/", threads_per_block, num_blocks, num_steps, s,
            peak_idx, peak_val, expected_pdm, elapsed_ms);
    }

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaStreamDestroy(stream[s]));
        CUDA_CHECK(cudaEventDestroy(ev_gen_done[s]));
        CUDA_CHECK(cudaEventDestroy(ev_smooth_done[s]));
        CUDA_CHECK(cudaFreeHost(h_peak_vals[s]));
        CUDA_CHECK(cudaFreeHost(h_peak_idxs[s]));
        CUDA_CHECK(cudaFree(d_raw[s]));
        CUDA_CHECK(cudaFree(d_smoothed[s]));
        CUDA_CHECK(cudaFree(d_block_vals[s]));
        CUDA_CHECK(cudaFree(d_block_idxs[s]));
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

/* ***************************************************************************
 * Main
 * ***************************************************************************/

int main(int argc, char** argv) {
    int total_threads = (1 << 20); /* default: 1M threads */
    int block_size    = 256;

    if (argc >= 2)
        total_threads = atoi(argv[1]);
    if (argc >= 3)
        block_size = atoi(argv[2]);

    if (block_size <= 0 || block_size > 1024) {
        fprintf(stderr, "Error: block_size must be in [1, 1024]\n");
        return EXIT_FAILURE;
    }
    if (block_size % 32 != 0) {
        printf("Warning: block_size %d is not a multiple of 32 (warp size)\n",
               block_size);
    }

    /* round up to an even multiple of block_size */
    int num_blocks = total_threads / block_size;
    if (total_threads % block_size != 0) {
        num_blocks++;
        total_threads = num_blocks * block_size;
        printf("Warning: total_threads rounded up to %d\n", total_threads);
    }

    int num_steps = total_threads; /* one thread evaluates one PDM step */

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device              : %s\n", prop.name);
    printf("Concurrent kernels  : %d\n", prop.concurrentKernels);
    printf("\n");
    printf("WDM PDM Sweep Pipeline\n");
    printf("  Total threads : %d\n", total_threads);
    printf("  Block size    : %d\n", block_size);
    printf("  Num blocks    : %d\n", num_blocks);
    printf("  Sweep steps   : %d per ring\n", num_steps);
    printf("  Channels      : %d\n", NUM_STREAMS);
    printf("  Runs per mode : %d\n", NUM_RUNS);

    printf("\n");
    print_results_header();

    for (int run = 1; run <= NUM_RUNS; run++) {
        run_without_streams(num_steps, block_size, num_blocks, run);
        run_with_streams(num_steps, block_size, num_blocks, run);
    }

    printf("\n");
    return EXIT_SUCCESS;
}
