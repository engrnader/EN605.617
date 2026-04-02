/**
 * @file    common.h
 * @brief   Shared constants, macros, and types for the
 *          multi-channel sensor calibration pipeline.
 */

#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------------------------------------------------------------
 * Pipeline constants
 * --------------------------------------------------------------- */
#define NUM_CHANNELS 4        /* sensor channels (CUDA streams) */
#define SMOOTH_RADIUS 8       /* moving average half-window     */
#define PEAK_AMPLITUDE 1.0f   /* normalized response amplitude  */
#define NOISE_AMPLITUDE 0.04f /* measurement noise amplitude    */

/* Kernel 2: gradient and margin scoring */
#define GRADIENT_RADIUS 4      /* central difference half-width  */
#define EDGE_MARGIN_FRAC 0.05f /* ignore peaks near sweep edges  */

/* Kernel 3: Monte Carlo */
#define MC_LOOP_ITERS 200      /* control loop steps per scenario*/
#define MC_DRIFT_MAX 0.03f     /* max abs drift rate per step    */
#define MC_NOISE_STDDEV 0.15f  /* Gaussian noise standard dev    */
#define MC_FAIL_THRESHOLD 3.0f /* deviation causing lock loss    */
#define MC_CTRL_GAIN 0.3f      /* proportional controller gain   */

/* Benchmarking */
#define NUM_RUNS 2 /* timed runs per mode            */

/* ---------------------------------------------------------------
 * Per-channel sensor parameters
 * --------------------------------------------------------------- */
static const float CHANNEL_CENTER_FRAC[NUM_CHANNELS] = {0.15f, 0.38f, 0.62f,
                                                        0.82f};
static const float CHANNEL_WIDTH_FRAC[NUM_CHANNELS]  = {0.018f, 0.022f, 0.020f,
                                                        0.024f};

/* ---------------------------------------------------------------
 * Result structures
 * --------------------------------------------------------------- */

/** Output from Kernel 1: sweep + peak detection. */
typedef struct {
    int   peak_idx; /* DAC index of detected peak      */
    float peak_val; /* detector value at peak           */
} SweepResult;

/** Output from Kernel 2: operating point selection. */
typedef struct {
    int   best_idx; /* DAC index of selected op. point  */
    float margin;   /* stability margin score           */
    float gradient; /* gradient magnitude at op. point  */
} OpPointResult;

/** Output from Kernel 3: Monte Carlo statistics. */
typedef struct {
    int   num_scenarios;
    int   pass_count;
    float pass_pct;
    float avg_dev;
    float worst_dev;
    float p95_dev;
    float p99_dev;
} MonteCarloResult;

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

#endif /* COMMON_H */
