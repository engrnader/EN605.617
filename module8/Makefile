NVCC = nvcc
NVCC_FLAGS = -O2 --std=c++17 -lineinfo

all: monte_carlo.exe

monte_carlo.exe: monte_carlo.cu
	$(NVCC) $(NVCC_FLAGS) monte_carlo.cu -o monte_carlo.exe

clean:
	rm -f monte_carlo.exe

.PHONY: all clean
