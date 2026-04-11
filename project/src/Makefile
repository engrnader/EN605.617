NVCC = nvcc
NVCC_FLAGS = -O2 --std=c++17 -lineinfo

SRCS = main.cu kernel_sweep.cu kernel_gradient.cu \
       kernel_monte_carlo.cu cpu_baseline.cu
TARGET = calibration.exe

all: $(TARGET)

$(TARGET): $(SRCS) common.h kernel_sweep.h kernel_gradient.h \
           kernel_monte_carlo.h cpu_baseline.h
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
