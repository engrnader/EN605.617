@echo off
nvcc -O2 --std=c++17 -lineinfo monte_carlo.cu -o monte_carlo.exe
monte_carlo.exe 65536 256
