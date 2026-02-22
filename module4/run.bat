@echo off

echo Building...
nvcc -lineinfo assignment.cu -o assignment.exe
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo *** Run 1: blockSize=64 ***
assignment.exe 64

echo.
echo *** Run 2: blockSize=128 ***
assignment.exe 128

echo.
echo *** Run 3: blockSize=256 ***
assignment.exe 256

echo.
echo *** Run 4: blockSize=512 ***
assignment.exe 512
