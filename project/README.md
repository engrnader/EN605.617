# GPU-Accelerated Sensor Calibration and Robustness Analysis

Final project for EN605.617 (Introduction to GPU Programming).

A three-kernel CUDA pipeline that calibrates a resonant sensor and
stress-tests it against drift, benchmarked against a sequential CPU
baseline.

## What it does

1. **Kernel 1 (Sweep + Peak):** generates a noisy Lorentzian response
   across 65,536 DAC setpoints, smooths it with a stencil, and finds the
   peak via tree-based parallel reduction in shared memory.
2. **Kernel 2 (Gradient + Margin):** computes a central-difference
   gradient, scores each point by stability margin, and selects the
   best operating point.
3. **Kernel 3 (Monte Carlo):** runs 16,384 control-loop simulations in
   parallel using per-thread cuRAND state, then aggregates pass rate,
   mean, P95, P99, and worst-case deviation with Thrust.

A CPU baseline runs the identical algorithms sequentially for
correctness validation and speedup measurement.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc on PATH)
- C++17 compiler (MSVC on Windows, g++ on Linux)

## Build

### Linux / WSL

```
cd src
make
```

### Windows (x64 Native Tools Command Prompt)

```
cd src
run.bat
```

`run.bat` compiles and then executes the demo with default parameters.

### Manual compile

```
nvcc -O2 --std=c++17 -lineinfo main.cu kernel_sweep.cu kernel_gradient.cu kernel_monte_carlo.cu cpu_baseline.cu -o calibration.exe
```

## Run

```
./calibration.exe <sweep_steps> <block_size> [mc_scenarios]
```

Arguments:

| Argument       | Description                                    | Example |
|----------------|------------------------------------------------|---------|
| `sweep_steps`  | Points per frequency sweep                     | 65536   |
| `block_size`   | CUDA threads per block (multiple of 32)        | 256     |
| `mc_scenarios` | Monte Carlo scenarios per run (optional)       | 16384   |

Default demo:

```
./calibration.exe 65536 256 16384
```

## Expected output

The program prints device info, then runs the GPU pipeline twice
followed by the CPU baseline twice. Each run reports per-kernel
results and elapsed time. Representative timings on an RTX 2060:

- GPU steady-state: about 11.5 ms per run
- CPU baseline: about 275 to 325 ms per run
- Speedup: roughly 25 to 30 times

## Source layout

| File                   | Role                                                        |
|------------------------|-------------------------------------------------------------|
| `main.cu`              | Driver: arg parsing, GPU/CPU orchestration, output          |
| `kernel_sweep.cu`      | Kernel 1: response generation, smoothing, peak reduction    |
| `kernel_gradient.cu`   | Kernel 2: gradient, margin scoring, operating point         |
| `kernel_monte_carlo.cu`| Kernel 3: cuRAND init, control loop, Thrust aggregation     |
| `cpu_baseline.cu`      | Sequential CPU reference for all three stages               |
| `common.h`             | Shared constants, result structures, `CUDA_CHECK` macro     |
| `Makefile`             | Linux build                                                 |
| `run.bat`              | Windows build and demo                                      |

## Author

Nader Mahfouz, Johns Hopkins University, Spring 2026.
