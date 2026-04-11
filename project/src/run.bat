@echo off
nvcc -O2 --std=c++17 -lineinfo main.cu kernel_sweep.cu kernel_gradient.cu kernel_monte_carlo.cu cpu_baseline.cu -o calibration.exe
calibration.exe 65536 256 16384
