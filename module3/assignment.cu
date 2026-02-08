/* Based on the work of Andrew Krepps */
/* Module 3 Assignment: CPU vs GPU comparison with and without branching */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Scalar used in the simple arithmetic operation */
#define SCALAR 3

/***************************************************************************
 * GPU Kernels
 ***************************************************************************/

/* GPU kernel: element-wise computation WITHOUT conditional branching.
   Each thread computes: output[i] = input[i] * input[i] + SCALAR */
__global__ void gpu_no_branch(const float *input, float *output, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		output[idx] = input[idx] * input[idx] + SCALAR;
	}
}

/* GPU kernel: element-wise computation WITH conditional branching.
   Even-indexed elements get squared + SCALAR, odd-indexed get halved - SCALAR.
   This causes warp divergence since adjacent threads take different paths. */
__global__ void gpu_with_branch(const float *input, float *output, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		if (idx % 2 == 0) {
			output[idx] = input[idx] * input[idx] + SCALAR;
		} else {
			output[idx] = input[idx] * 0.5f - SCALAR;
		}
	}
}

/***************************************************************************
 * CPU Functions
 ***************************************************************************/

/* CPU version: same computation as gpu_no_branch */
void cpu_no_branch(const float *input, float *output, int n)
{
	for (int i = 0; i < n; i++) {
		output[i] = input[i] * input[i] + SCALAR;
	}
}

/* CPU version: same computation as gpu_with_branch */
void cpu_with_branch(const float *input, float *output, int n)
{
	for (int i = 0; i < n; i++) {
		if (i % 2 == 0) {
			output[i] = input[i] * input[i] + SCALAR;
		} else {
			output[i] = input[i] * 0.5f - SCALAR;
		}
	}
}

/***************************************************************************
 * Timing helpers
 ***************************************************************************/

/* Record a CUDA event timestamp and return it */
cudaEvent_t record_cuda_event(void)
{
	cudaEvent_t event;
	cudaEventCreate(&event);
	cudaEventRecord(event, 0);
	return event;
}

/***************************************************************************
 * GPU execution wrapper
 ***************************************************************************/

/* Runs a GPU kernel, measures execution time using cudaEvents, and copies
   the result back to host_output. Returns elapsed time in milliseconds. */
typedef void (*KernelFunc)(const float*, float*, int);

float run_gpu_kernel(KernelFunc kernel, const float *host_input,
                     float *host_output, int n, int numBlocks, int blockSize)
{
	size_t bytes = n * sizeof(float);

	float *d_input  = NULL;
	float *d_output = NULL;
	cudaMalloc((void **)&d_input, bytes);
	cudaMalloc((void **)&d_output, bytes);

	cudaMemcpy(d_input, host_input, bytes, cudaMemcpyHostToDevice);

	/* Time only the kernel execution */
	cudaEvent_t start = record_cuda_event();
	kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
	cudaEvent_t stop = record_cuda_event();
	cudaEventSynchronize(stop);

	float elapsed_ms = 0.0f;
	cudaEventElapsedTime(&elapsed_ms, start, stop);

	cudaMemcpy(host_output, d_output, bytes, cudaMemcpyDeviceToHost);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	cudaFree(d_output);

	return elapsed_ms;
}

/***************************************************************************
 * CPU execution wrapper
 ***************************************************************************/

typedef void (*CpuFunc)(const float*, float*, int);

/* Runs a CPU function and measures wall-clock time. Returns elapsed ms. */
float run_cpu_function(CpuFunc func, const float *input, float *output, int n)
{
	clock_t start = clock();
	func(input, output, n);
	clock_t end = clock();

	return (float)(end - start) / (CLOCKS_PER_SEC / 1000.0f);
}

/***************************************************************************
 * Verification: compare CPU and GPU output arrays
 ***************************************************************************/

int verify_results(const float *cpu_out, const float *gpu_out, int n,
                   const char *label)
{
	const float epsilon = 1e-4f;
	for (int i = 0; i < n; i++) {
		float diff = cpu_out[i] - gpu_out[i];
		if (diff < 0) diff = -diff;
		if (diff > epsilon) {
			printf("  [MISMATCH] %s at index %d: cpu=%.4f gpu=%.4f\n",
			       label, i, cpu_out[i], gpu_out[i]);
			return 0;
		}
	}
	printf("  [VERIFIED] %s: CPU and GPU results match\n", label);
	return 1;
}

/***************************************************************************
 * Main
 ***************************************************************************/

int main(int argc, char **argv)
{
	/* Parse command line arguments (from starter code) */
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads / blockSize;

	/* Validate: round up if not evenly divisible */
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks * blockSize;
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	int n = totalThreads;

	printf("+------------------+-----------+\n");
	printf("| Parameter        | Value     |\n");
	printf("+------------------+-----------+\n");
	printf("| Array size (N)   | %-9d |\n", n);
	printf("| Block size       | %-9d |\n", blockSize);
	printf("| Num blocks       | %-9d |\n", numBlocks);
	printf("+------------------+-----------+\n");
	printf("\n");

	/* Allocate host memory */
	size_t bytes = n * sizeof(float);
	float *h_input      = (float *)malloc(bytes);
	float *h_cpu_out    = (float *)malloc(bytes);
	float *h_gpu_out    = (float *)malloc(bytes);

	if (!h_input || !h_cpu_out || !h_gpu_out) {
		fprintf(stderr, "Error: failed to allocate host memory\n");
		return 1;
	}

	/* Initialize input data */
	for (int i = 0; i < n; i++) {
		h_input[i] = (float)(i % 1000) * 0.01f;
	}

	/* ---- Run all 4 variants and collect timings ---- */

	/* 1. CPU without branching */
	float cpu_no_branch_ms = run_cpu_function(cpu_no_branch, h_input,
	                                          h_cpu_out, n);

	/* 2. GPU without branching */
	float gpu_no_branch_ms = run_gpu_kernel(gpu_no_branch, h_input,
	                                        h_gpu_out, n, numBlocks, blockSize);
	verify_results(h_cpu_out, h_gpu_out, n, "No-Branch");

	/* 3. CPU with branching */
	float cpu_branch_ms = run_cpu_function(cpu_with_branch, h_input,
	                                       h_cpu_out, n);

	/* 4. GPU with branching */
	float gpu_branch_ms = run_gpu_kernel(gpu_with_branch, h_input,
	                                     h_gpu_out, n, numBlocks, blockSize);
	verify_results(h_cpu_out, h_gpu_out, n, "With-Branch");

	/* ---- Print timing results ---- */
	printf("\n+------------------------+------------+\n");
	printf("| Variant                | Time (ms)  |\n");
	printf("+------------------------+------------+\n");
	printf("| CPU without branching  | %10.4f |\n", cpu_no_branch_ms);
	printf("| GPU without branching  | %10.4f |\n", gpu_no_branch_ms);
	printf("| CPU with branching     | %10.4f |\n", cpu_branch_ms);
	printf("| GPU with branching     | %10.4f |\n", gpu_branch_ms);
	printf("+------------------------+------------+\n");
	printf("\n");

	/* Free host memory */
	free(h_input);
	free(h_cpu_out);
	free(h_gpu_out);

	return 0;
}
