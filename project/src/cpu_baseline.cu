/**
 * @file    cpu_baseline.cu
 * @brief   Sequential CPU implementations of all three pipeline
 *          stages. Used for GPU vs CPU timing comparison.
 *
 *          .cu extension so nvcc compiles it alongside GPU code.
 *          All functions here run on the host only.
 */

#include <math.h>
#include <stdlib.h>

#include <algorithm>

#include "cpu_baseline.h"

/* ---------------------------------------------------------------
 * CPU Kernel 1: Sweep + smooth + find peak
 * --------------------------------------------------------------- */

/**
 * @brief  Generate peak-shaped response on CPU.
 */
static void cpu_generate(float* response, int n, float x0, float gamma,
                         int noise_seed) {
    for (int i = 0; i < n; i++) {
        float x      = (float)i / (float)n;
        float delta  = (x - x0) / gamma;
        float signal = PEAK_AMPLITUDE / (1.0f + delta * delta);
        float noise  = NOISE_AMPLITUDE * sinf((float)(i * noise_seed) * 0.037f);
        response[i]  = signal + noise;
    }
}

/**
 * @brief  Smooth response on CPU.
 */
static void cpu_smooth(float* out, const float* in, int n, int radius) {
    for (int i = 0; i < n; i++) {
        float sum   = 0.0f;
        int   count = 0;
        for (int k = -radius; k <= radius; k++) {
            int j = i + k;
            if (j >= 0 && j < n) {
                sum += in[j];
                count++;
            }
        }
        out[i] = sum / (float)count;
    }
}

/**
 * @brief  Find peak on CPU (linear scan).
 */
static void cpu_find_peak(const float* data, int n, SweepResult* result) {
    result->peak_val = -1.0f;
    result->peak_idx = -1;
    for (int i = 0; i < n; i++) {
        if (data[i] > result->peak_val) {
            result->peak_val = data[i];
            result->peak_idx = i;
        }
    }
}

void cpu_sweep(int n, float center_frac, float width_frac, int noise_seed,
               SweepResult* result) {
    float* raw      = (float*)malloc(n * sizeof(float));
    float* smoothed = (float*)malloc(n * sizeof(float));

    cpu_generate(raw, n, center_frac, width_frac, noise_seed);
    cpu_smooth(smoothed, raw, n, SMOOTH_RADIUS);
    cpu_find_peak(smoothed, n, result);

    free(raw);
    free(smoothed);
}

/* ---------------------------------------------------------------
 * CPU Kernel 2: Gradient + margin scoring
 * --------------------------------------------------------------- */

void cpu_gradient(const float* smoothed, int n, OpPointResult* result) {
    result->best_idx = -1;
    result->margin   = -1.0f;
    result->gradient = 0.0f;

    for (int i = 0; i < n; i++) {
        int left  = i - GRADIENT_RADIUS;
        int right = i + GRADIENT_RADIUS;
        if (left < 0)
            left = 0;
        if (right >= n)
            right = n - 1;

        float grad = (smoothed[right] - smoothed[left]) / (float)(right - left);

        float frac   = (float)i / (float)n;
        float edge_w = 1.0f;
        if (frac < EDGE_MARGIN_FRAC || frac > (1.0f - EDGE_MARGIN_FRAC))
            edge_w = 0.0f;

        float margin = smoothed[i] * edge_w / (1.0f + fabsf(grad));

        if (margin > result->margin) {
            result->margin   = margin;
            result->best_idx = i;
            result->gradient = grad;
        }
    }
}

/* ---------------------------------------------------------------
 * CPU Kernel 3: Monte Carlo
 * --------------------------------------------------------------- */

/**
 * @brief  Simple LCG for CPU baseline RNG.
 */
static unsigned int lcg_next(unsigned int state) {
    return state * 1664525u + 1013904223u;
}

/**
 * @brief  Box-Muller normal sample from LCG state.
 */
static float lcg_normal(unsigned int* state) {
    *state   = lcg_next(*state);
    float u1 = (float)(*state) / 4294967296.0f;
    *state   = lcg_next(*state);
    float u2 = (float)(*state) / 4294967296.0f;
    if (u1 < 1e-10f)
        u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/**
 * @brief  Simulate one drift scenario on CPU.
 */
static float cpu_sim_one(float drift, float* peak_dev_out, unsigned int* rng) {
    float dev = 0.0f, peak = 0.0f;
    for (int s = 0; s < MC_LOOP_ITERS; s++) {
        float noise      = lcg_normal(rng) * MC_NOISE_STDDEV;
        float correction = MC_CTRL_GAIN * dev;
        dev += drift + noise - correction;
        float ad = fabsf(dev);
        if (ad > peak)
            peak = ad;
    }
    *peak_dev_out = peak;
    return (peak <= MC_FAIL_THRESHOLD) ? 1.0f : 0.0f;
}

void cpu_monte_carlo(int num_scenarios, unsigned int base_seed,
                     MonteCarloResult* result) {
    int   pass_count = 0;
    float total_dev  = 0.0f;
    float worst_dev  = 0.0f;
    int   n          = num_scenarios;

    float* devs = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        /* mix seed so different base_seeds produce distinct sequences */
        unsigned int rng =
            lcg_next(base_seed ^ (unsigned int)(i * 2654435761u));
        float drift = ((float)(rng % 10000) / 10000.0f) * 2.0f * MC_DRIFT_MAX -
                      MC_DRIFT_MAX;
        float peak_dev;
        float passed = cpu_sim_one(drift, &peak_dev, &rng);
        pass_count += (int)passed;
        total_dev += peak_dev;
        devs[i] = peak_dev;
        if (peak_dev > worst_dev)
            worst_dev = peak_dev;
    }

    std::sort(devs, devs + n);

    result->num_scenarios = n;
    result->pass_count    = pass_count;
    result->pass_pct      = 100.0f * (float)pass_count / (float)n;
    result->avg_dev       = total_dev / (float)n;
    result->worst_dev     = worst_dev;
    result->p95_dev       = devs[(int)(0.95f * n)];
    result->p99_dev       = devs[(int)(0.99f * n)];

    free(devs);
}
