@echo off
nvcc -lineinfo assignment.cu -o assignment.exe || exit /b 1
del /f results.dat 2>nul
for %%b in (64 128 256 512) do assignment.exe %%b >> results.dat
type results.dat
