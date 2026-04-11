/**
 * @file    main.cu
 * @brief   Multi-Channel Sensor Calibration Pipeline
 *
 *          Orchestrates the three-kernel GPU pipeline:
 *            Kernel 1: Parallel sweep + peak detection
 *            Kernel 2: Gradient + margin + op. point selection
 *            Kernel 3: Monte Carlo robustness analysis
 *
 *          Each channel runs on its own CUDA stream.
 *          CPU baselines run sequentially for comparison.
 *
 * @usage   ./calibration <sweep_steps> <block_size> [mc_scenarios]
 */

#include <stdio.h>

#include <chrono>

#include "common.h"
#include "cpu_baseline.h"
#include "kernel_gradient.h"
#include "kernel_monte_carlo.h"
#include "kernel_sweep.h"

/* ---------------------------------------------------------------
 * Argument parsing
 * --------------------------------------------------------------- */

typedef struct {
    int sweep_steps;  /* DAC sweep points per channel    */
    int block_size;   /* CUDA threads per block           */
    int mc_scenarios; /* Monte Carlo scenarios per channel*/
} Config;

/**
 * @brief  Parse and validate command line arguments.
 */
static int parse_args(int argc, char** argv, Config* cfg) {
    cfg->sweep_steps  = (1 << 16); /* 65536 */
    cfg->block_size   = 256;
    cfg->mc_scenarios = (1 << 14); /* 16384 */

    if (argc >= 2)
        cfg->sweep_steps = atoi(argv[1]);
    if (argc >= 3)
        cfg->block_size = atoi(argv[2]);
    if (argc >= 4)
        cfg->mc_scenarios = atoi(argv[3]);

    if (cfg->block_size <= 0 || cfg->block_size > 1024) {
        fprintf(stderr, "Error: block_size must be in [1, 1024]\n");
        return -1;
    }
    if (cfg->block_size % 32 != 0)
        printf("Warning: block_size %d not multiple of 32\n", cfg->block_size);
    return 0;
}

/**
 * @brief  Round n up to a multiple of block_size.
 */
static int round_up(int n, int block_size) {
    int nb = n / block_size;
    if (n % block_size != 0) {
        nb++;
        n = nb * block_size;
    }
    return n;
}

/* ---------------------------------------------------------------
 * Print helpers
 * --------------------------------------------------------------- */

static void print_device_info(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device : %s\n", prop.name);
    printf("SMs    : %d\n", prop.multiProcessorCount);
    printf("\n");
}

static void print_config(const Config* cfg) {
    printf("Pipeline Configuration\n");
    printf("  Channels      : %d\n", NUM_CHANNELS);
    printf("  Sweep steps   : %d\n", cfg->sweep_steps);
    printf("  Block size    : %d\n", cfg->block_size);
    printf("  MC scenarios  : %d per channel\n", cfg->mc_scenarios);
    printf("  Runs          : %d\n", NUM_RUNS);
    printf("\n");
}

static void print_sweep_result(int ch, const SweepResult* r, int n) {
    float expected = CHANNEL_CENTER_FRAC[ch] * (float)n;
    printf(
        "  Ch %d | Peak idx: %6d (expected: %6.0f) "
        "| Val: %.4f\n",
        ch, r->peak_idx, expected, r->peak_val);
}

static void print_oppoint_result(int ch, const OpPointResult* r) {
    printf(
        "  Ch %d | Op.point: %6d | Margin: %.4f "
        "| Grad: %+.6f\n",
        ch, r->best_idx, r->margin, r->gradient);
}

static void print_mc_result(int ch, const MonteCarloResult* r) {
    printf(
        "  Ch %d | Pass: %6.2f%% | Avg: %.3f "
        "| P95: %.3f | P99: %.3f | Worst: %.3f\n",
        ch, r->pass_pct, r->avg_dev, r->p95_dev, r->p99_dev, r->worst_dev);
}

/* ---------------------------------------------------------------
 * GPU pipeline: all channels with CUDA streams
 * --------------------------------------------------------------- */

/**
 * @brief  Run full GPU pipeline for all channels.
 */
static void run_gpu_pipeline(const Config* cfg, int run_idx) {
    int    n_sw   = round_up(cfg->sweep_steps, cfg->block_size);
    int    n_mc   = round_up(cfg->mc_scenarios, cfg->block_size);
    int    nb_sw  = n_sw / cfg->block_size;
    int    nb_mc  = n_mc / cfg->block_size;
    size_t fbytes = (size_t)n_sw * sizeof(float);

    cudaStream_t streams[NUM_CHANNELS];
    float*       d_raw[NUM_CHANNELS];
    float*       d_smoothed[NUM_CHANNELS];

    SweepResult      sw_res[NUM_CHANNELS];
    OpPointResult    op_res[NUM_CHANNELS];
    MonteCarloResult mc_res[NUM_CHANNELS];

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* allocate per-channel buffers and streams */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        CUDA_CHECK(cudaStreamCreate(&streams[ch]));
        CUDA_CHECK(cudaMalloc(&d_raw[ch], fbytes));
        CUDA_CHECK(cudaMalloc(&d_smoothed[ch], fbytes));
    }

    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    /* Kernel 1 + 2: sweep and gradient per channel */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        int seed = run_idx * NUM_CHANNELS + ch + 1;

        sweep_pipeline(d_raw[ch], d_smoothed[ch], n_sw, cfg->block_size, nb_sw,
                       CHANNEL_CENTER_FRAC[ch], CHANNEL_WIDTH_FRAC[ch], seed,
                       streams[ch], &sw_res[ch]);

        gradient_pipeline(d_smoothed[ch], n_sw, cfg->block_size, nb_sw,
                          streams[ch], &op_res[ch]);
    }

    /* Kernel 3: Monte Carlo per channel */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        unsigned long long mc_seed =
            42ULL + (unsigned long long)(run_idx * NUM_CHANNELS + ch);
        monte_carlo_pipeline(n_mc, cfg->block_size, nb_mc, mc_seed, streams[ch],
                             &mc_res[ch]);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* print results */
    printf("GPU Run %d (%.3f ms)\n", run_idx, elapsed_ms);
    printf("  -- Kernel 1: Sweep + Peak --\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        print_sweep_result(ch, &sw_res[ch], n_sw);

    printf("  -- Kernel 2: Gradient + Op.Point --\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        print_oppoint_result(ch, &op_res[ch]);

    printf("  -- Kernel 3: Monte Carlo --\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++) print_mc_result(ch, &mc_res[ch]);
    printf("\n");

    /* cleanup */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        CUDA_CHECK(cudaFree(d_raw[ch]));
        CUDA_CHECK(cudaFree(d_smoothed[ch]));
        CUDA_CHECK(cudaStreamDestroy(streams[ch]));
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

/* ---------------------------------------------------------------
 * CPU baseline: sequential, all channels
 * --------------------------------------------------------------- */

/**
 * @brief  Run full CPU baseline for all channels.
 */
static void run_cpu_pipeline(const Config* cfg, int run_idx) {
    auto t0 = std::chrono::steady_clock::now();

    SweepResult      sw_res[NUM_CHANNELS];
    OpPointResult    op_res[NUM_CHANNELS];
    MonteCarloResult mc_res[NUM_CHANNELS];

    /* CPU needs smoothed data for Kernel 2 */
    int    n        = cfg->sweep_steps;
    float* raw      = (float*)malloc(n * sizeof(float));
    float* smoothed = (float*)malloc(n * sizeof(float));

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        int seed = run_idx * NUM_CHANNELS + ch + 1;
        cpu_sweep(n, CHANNEL_CENTER_FRAC[ch], CHANNEL_WIDTH_FRAC[ch], seed,
                  &sw_res[ch]);

        /* regenerate smoothed for gradient */
        for (int i = 0; i < n; i++) {
            float x   = (float)i / (float)n;
            float d   = (x - CHANNEL_CENTER_FRAC[ch]) / CHANNEL_WIDTH_FRAC[ch];
            float sig = PEAK_AMPLITUDE / (1.0f + d * d);
            float noise = NOISE_AMPLITUDE * sinf((float)(i * seed) * 0.037f);
            raw[i]      = sig + noise;
        }
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            int   cnt = 0;
            for (int k = -SMOOTH_RADIUS; k <= SMOOTH_RADIUS; k++) {
                int j = i + k;
                if (j >= 0 && j < n) {
                    sum += raw[j];
                    cnt++;
                }
            }
            smoothed[i] = sum / (float)cnt;
        }
        cpu_gradient(smoothed, n, &op_res[ch]);

        unsigned int mc_seed =
            42u + (unsigned int)(run_idx * NUM_CHANNELS + ch);
        cpu_monte_carlo(cfg->mc_scenarios, mc_seed, &mc_res[ch]);
    }

    free(raw);
    free(smoothed);

    auto   t1 = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("CPU Run %d (%.3f ms)\n", run_idx, elapsed_ms);
    printf("  -- Kernel 1: Sweep + Peak --\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        print_sweep_result(ch, &sw_res[ch], cfg->sweep_steps);

    printf("  -- Kernel 2: Gradient + Op.Point --\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        print_oppoint_result(ch, &op_res[ch]);

    printf("  -- Kernel 3: Monte Carlo --\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++) print_mc_result(ch, &mc_res[ch]);
    printf("\n");
}

/* ---------------------------------------------------------------
 * Main
 * --------------------------------------------------------------- */

int main(int argc, char** argv) {
    Config cfg;
    if (parse_args(argc, argv, &cfg) < 0)
        return EXIT_FAILURE;

    print_device_info();
    print_config(&cfg);

    printf("=== GPU Pipeline (CUDA Streams) ===\n\n");
    for (int run = 1; run <= NUM_RUNS; run++) run_gpu_pipeline(&cfg, run);

    printf("=== CPU Baseline (Sequential) ===\n\n");
    for (int run = 1; run <= NUM_RUNS; run++) run_cpu_pipeline(&cfg, run);

    return EXIT_SUCCESS;
}
