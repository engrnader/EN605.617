# Module 8: Monte Carlo Robustness Analysis

CUDA advanced libraries assignment using cuRAND (Module 7) and Thrust (Module 8).

## Build and Run

```
nvcc -O2 --std=c++17 -lineinfo monte_carlo.cu -o monte_carlo.exe
monte_carlo.exe <num_scenarios> <block_size>
```

Or use the batch script:
```
run.bat 65536 256
```

## Examples

```
monte_carlo.exe 65536 256    # 64K scenarios
monte_carlo.exe 1048576 256  # 1M scenarios
```
