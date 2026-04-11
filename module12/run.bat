@echo off
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CL_INC=%CUDA_PATH%\include
set CL_LIB=%CUDA_PATH%\lib\x64

echo *** Building ***
g++ -std=c++11 -w -DCL_TARGET_OPENCL_VERSION=300 -I. -I"%CL_INC%" assignment.cpp -o assignment.exe -L"%CL_LIB%" -lOpenCL
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b 1
)

echo.
echo *** Run 1: Default (16 elements, workgroup 4, offset 10, buffer copy) ***
assignment.exe

echo.
echo *** Run 2: Custom size and offset ***
assignment.exe --size 32 --workgroup 8 --offset 5

echo.
echo *** Run 3: Memory mapping ***
assignment.exe --size 16 --workgroup 4 --offset 10 --useMap
