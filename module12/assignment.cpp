/**
 * @file    assignment.cpp
 * @brief   OpenCL program demonstrating buffers, sub-buffers,
 *          vector types (int4), memory mapping, multiple
 *          kernels, and event dependencies.
 *
 * @usage   ./assignment [--size N] [--workgroup N]
 *                       [--offset N] [--useMap]
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define DEFAULT_PLATFORM 0
#define DEFAULT_NUM_ELEMENTS 16
#define DEFAULT_WORKGROUP_SIZE 4
#define DEFAULT_OFFSET 10
#define VECTOR_WIDTH 4
#define MAX_PRINT_ELEMENTS 4

/* ---------------------------------------------------------------
 * Configuration
 * --------------------------------------------------------------- */

struct Config {
    int  platform;
    int  numElements;
    int  workgroupSize;
    int  offset;
    bool useMap;
};

/* ---------------------------------------------------------------
 * Error checking
 * --------------------------------------------------------------- */

inline void checkErr(cl_int err, const char* name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/* ---------------------------------------------------------------
 * Argument parsing
 * --------------------------------------------------------------- */

static void parse_args(int argc, char** argv, Config* cfg) {
    cfg->platform      = DEFAULT_PLATFORM;
    cfg->numElements   = DEFAULT_NUM_ELEMENTS;
    cfg->workgroupSize = DEFAULT_WORKGROUP_SIZE;
    cfg->offset        = DEFAULT_OFFSET;
    cfg->useMap        = false;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--platform" && i + 1 < argc)
            cfg->platform = atoi(argv[++i]);
        else if (arg == "--size" && i + 1 < argc)
            cfg->numElements = atoi(argv[++i]);
        else if (arg == "--workgroup" && i + 1 < argc)
            cfg->workgroupSize = atoi(argv[++i]);
        else if (arg == "--offset" && i + 1 < argc)
            cfg->offset = atoi(argv[++i]);
        else if (arg == "--useMap")
            cfg->useMap = true;
        else {
            std::cout << "Usage: " << argv[0] << " [--size N] [--workgroup N]"
                      << " [--offset N] [--useMap]" << std::endl;
            exit(0);
        }
    }
}

/* ---------------------------------------------------------------
 * Print int4 array (up to MAX_PRINT_ELEMENTS)
 * --------------------------------------------------------------- */

static void print_int4_array(const int* data, int numElements, int totalInts,
                             int numDevices) {
    for (int d = 0; d < numDevices; d++) {
        std::cout << "  Device " << d << ": ";
        int limit = std::min(numElements, MAX_PRINT_ELEMENTS);
        for (int e = 0; e < limit; e++) {
            int b = (d * totalInts) + (e * VECTOR_WIDTH);
            std::cout << "(" << data[b] << "," << data[b + 1] << ","
                      << data[b + 2] << "," << data[b + 3] << ") ";
        }
        if (numElements > MAX_PRINT_ELEMENTS)
            std::cout << "...";
        std::cout << std::endl;
    }
}

/* ---------------------------------------------------------------
 * Setup: platform, device, context, program
 * --------------------------------------------------------------- */

static cl_context setup_opencl(const Config* cfg, cl_platform_id** platformIDs,
                               cl_device_id** deviceIDs, cl_uint* numDevices,
                               cl_program* program) {
    cl_int  errNum;
    cl_uint numPlatforms;

    /* platform enumeration */
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        errNum != CL_SUCCESS ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    *platformIDs = new cl_platform_id[numPlatforms];
    errNum       = clGetPlatformIDs(numPlatforms, *platformIDs, NULL);
    checkErr(errNum, "clGetPlatformIDs");

    /* device enumeration */
    errNum = clGetDeviceIDs((*platformIDs)[cfg->platform], CL_DEVICE_TYPE_ALL,
                            0, NULL, numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
        checkErr(errNum, "clGetDeviceIDs");

    *deviceIDs = new cl_device_id[*numDevices];
    errNum = clGetDeviceIDs((*platformIDs)[cfg->platform], CL_DEVICE_TYPE_ALL,
                            *numDevices, *deviceIDs, NULL);
    checkErr(errNum, "clGetDeviceIDs");

    std::cout << "Platforms: " << numPlatforms << ", Devices: " << *numDevices
              << std::endl;

    /* context */
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platformIDs)[cfg->platform], 0};
    cl_context context =
        clCreateContext(props, *numDevices, *deviceIDs, NULL, NULL, &errNum);
    checkErr(errNum, "clCreateContext");

    /* read and build program */
    std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");
    std::string srcProg(std::istreambuf_iterator<char>(srcFile),
                        (std::istreambuf_iterator<char>()));
    const char* src = srcProg.c_str();
    size_t      len = srcProg.length();

    *program = clCreateProgramWithSource(context, 1, &src, &len, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    errNum =
        clBuildProgram(*program, *numDevices, *deviceIDs, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(*program, (*deviceIDs)[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        std::cerr << "Build error:" << std::endl << log;
        checkErr(errNum, "clBuildProgram");
    }

    return context;
}

/* ---------------------------------------------------------------
 * Create buffers and sub-buffers
 * --------------------------------------------------------------- */

static cl_mem create_buffers(cl_context context, int totalInts,
                             cl_uint              numDevices,
                             std::vector<cl_mem>* subBuffers) {
    cl_int errNum;
    size_t bufSize = sizeof(int) * totalInts * numDevices;

    cl_mem mainBuf =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bufSize, NULL, &errNum);
    checkErr(errNum, "clCreateBuffer");

    for (unsigned d = 0; d < numDevices; d++) {
        cl_buffer_region region = {(size_t)(totalInts * d * sizeof(int)),
                                   (size_t)(totalInts * sizeof(int))};
        cl_mem           sub =
            clCreateSubBuffer(mainBuf, CL_MEM_READ_WRITE,
                              CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
        checkErr(errNum, "clCreateSubBuffer");
        subBuffers->push_back(sub);
    }

    return mainBuf;
}

/* ---------------------------------------------------------------
 * Create queues and kernels per device
 * --------------------------------------------------------------- */

static void create_queues_and_kernels(
    cl_context context, cl_program program, cl_device_id* deviceIDs,
    cl_uint numDevices, const std::vector<cl_mem>& subBuffers, int offset,
    std::vector<cl_command_queue>* queues, std::vector<cl_kernel>* sqKernels,
    std::vector<cl_kernel>* offKernels) {
    cl_int errNum;
    for (unsigned d = 0; d < numDevices; d++) {
        cl_command_queue q =
            clCreateCommandQueue(context, deviceIDs[d], 0, &errNum);
        checkErr(errNum, "clCreateCommandQueue");
        queues->push_back(q);

        cl_kernel sq = clCreateKernel(program, "square", &errNum);
        checkErr(errNum, "clCreateKernel(square)");
        errNum = clSetKernelArg(sq, 0, sizeof(cl_mem), &subBuffers[d]);
        checkErr(errNum, "clSetKernelArg(square)");
        sqKernels->push_back(sq);

        cl_kernel off = clCreateKernel(program, "add_offset", &errNum);
        checkErr(errNum, "clCreateKernel(add_offset)");
        errNum = clSetKernelArg(off, 0, sizeof(cl_mem), &subBuffers[d]);
        errNum |= clSetKernelArg(off, 1, sizeof(int), &offset);
        checkErr(errNum, "clSetKernelArg(add_offset)");
        offKernels->push_back(off);
    }
}

/* ---------------------------------------------------------------
 * Write data to device (buffer copy or map)
 * --------------------------------------------------------------- */

static void write_to_device(cl_command_queue queue, cl_mem buffer,
                            const int* data, size_t bytes, bool useMap) {
    cl_int errNum;
    if (useMap) {
        cl_int* ptr =
            (cl_int*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE, 0,
                                        bytes, 0, NULL, NULL, &errNum);
        checkErr(errNum, "clEnqueueMapBuffer(write)");
        for (size_t i = 0; i < bytes / sizeof(int); i++) ptr[i] = data[i];
        errNum = clEnqueueUnmapMemObject(queue, buffer, ptr, 0, NULL, NULL);
        checkErr(errNum, "clEnqueueUnmapMemObject");
    } else {
        errNum = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, bytes, data, 0,
                                      NULL, NULL);
        checkErr(errNum, "clEnqueueWriteBuffer");
    }
}

/* ---------------------------------------------------------------
 * Read data from device (buffer copy or map)
 * --------------------------------------------------------------- */

static void read_from_device(cl_command_queue queue, cl_mem buffer, int* data,
                             size_t bytes, bool useMap) {
    cl_int errNum;
    if (useMap) {
        cl_int* ptr =
            (cl_int*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                                        bytes, 0, NULL, NULL, &errNum);
        checkErr(errNum, "clEnqueueMapBuffer(read)");
        for (size_t i = 0; i < bytes / sizeof(int); i++) data[i] = ptr[i];
        errNum = clEnqueueUnmapMemObject(queue, buffer, ptr, 0, NULL, NULL);
        checkErr(errNum, "clEnqueueUnmapMemObject");
        clFinish(queue);
    } else {
        errNum = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bytes, data, 0,
                                     NULL, NULL);
        checkErr(errNum, "clEnqueueReadBuffer");
    }
}

/* ---------------------------------------------------------------
 * Execute both kernels with event dependency
 * --------------------------------------------------------------- */

static void run_kernels(const std::vector<cl_command_queue>& queues,
                        const std::vector<cl_kernel>&        sqKernels,
                        const std::vector<cl_kernel>&        offKernels,
                        size_t globalSize, size_t localSize,
                        cl_uint numDevices) {
    cl_int                errNum;
    std::vector<cl_event> events;

    for (unsigned d = 0; d < numDevices; d++) {
        cl_event ev1, ev2;

        errNum = clEnqueueNDRangeKernel(queues[d], sqKernels[d], 1, NULL,
                                        &globalSize, &localSize, 0, NULL, &ev1);
        checkErr(errNum, "clEnqueueNDRangeKernel(sq)");

        /* add_offset waits on square via event */
        errNum = clEnqueueNDRangeKernel(queues[d], offKernels[d], 1, NULL,
                                        &globalSize, &localSize, 1, &ev1, &ev2);
        checkErr(errNum, "clEnqueueNDRangeKernel(off)");

        events.push_back(ev2);
        clReleaseEvent(ev1);
    }

    clWaitForEvents(events.size(), &events[0]);
    for (size_t i = 0; i < events.size(); i++) clReleaseEvent(events[i]);
}

/* ---------------------------------------------------------------
 * Verify results: expected = input^2 + offset
 * --------------------------------------------------------------- */

static bool verify_results(const int* data, int numElements, int offset) {
    bool pass  = true;
    int  limit = std::min(numElements, MAX_PRINT_ELEMENTS);
    for (int e = 0; e < limit; e++) {
        int b = e * VECTOR_WIDTH;
        for (int c = 0; c < VECTOR_WIDTH; c++) {
            int orig     = b + c;
            int expected = (orig * orig) + offset;
            if (data[b + c] != expected) {
                std::cerr << "  FAIL [" << e << "][" << c << "]: expected "
                          << expected << ", got " << data[b + c] << std::endl;
                pass = false;
            }
        }
    }
    return pass;
}

/* ---------------------------------------------------------------
 * Cleanup
 * --------------------------------------------------------------- */

static void cleanup(cl_context context, cl_program program, cl_mem mainBuffer,
                    std::vector<cl_mem>&           subBuffers,
                    std::vector<cl_command_queue>& queues,
                    std::vector<cl_kernel>&        sqKernels,
                    std::vector<cl_kernel>&        offKernels,
                    cl_platform_id* platformIDs, cl_device_id* deviceIDs,
                    int* data) {
    for (size_t d = 0; d < queues.size(); d++) {
        clReleaseKernel(sqKernels[d]);
        clReleaseKernel(offKernels[d]);
        clReleaseCommandQueue(queues[d]);
        clReleaseMemObject(subBuffers[d]);
    }
    clReleaseMemObject(mainBuffer);
    clReleaseProgram(program);
    clReleaseContext(context);
    delete[] data;
    delete[] platformIDs;
    delete[] deviceIDs;
}

/* ---------------------------------------------------------------
 * Main
 * --------------------------------------------------------------- */

int main(int argc, char** argv) {
    Config cfg;
    parse_args(argc, argv, &cfg);

    std::cout << "OpenCL: Buffers, Sub-Buffers, Vectors" << std::endl
              << "  Elements: " << cfg.numElements
              << ", Workgroup: " << cfg.workgroupSize
              << ", Offset: " << cfg.offset
              << ", Map: " << (cfg.useMap ? "yes" : "no") << std::endl;

    cl_platform_id* platformIDs;
    cl_device_id*   deviceIDs;
    cl_uint         numDevices;
    cl_program      program;

    cl_context context =
        setup_opencl(&cfg, &platformIDs, &deviceIDs, &numDevices, &program);

    int    totalInts = cfg.numElements * VECTOR_WIDTH;
    size_t bytes     = sizeof(int) * totalInts * numDevices;
    int*   data      = new int[totalInts * numDevices];
    for (int i = 0; i < totalInts * (int)numDevices; i++) data[i] = i;

    std::cout << "Input:" << std::endl;
    print_int4_array(data, cfg.numElements, totalInts, numDevices);

    std::vector<cl_mem> subBuffers;
    cl_mem              mainBuf =
        create_buffers(context, totalInts, numDevices, &subBuffers);

    std::vector<cl_command_queue> queues;
    std::vector<cl_kernel>        sqKernels, offKernels;
    create_queues_and_kernels(context, program, deviceIDs, numDevices,
                              subBuffers, cfg.offset, &queues, &sqKernels,
                              &offKernels);

    write_to_device(queues[numDevices - 1], mainBuf, data, bytes, cfg.useMap);

    run_kernels(queues, sqKernels, offKernels, (size_t)cfg.numElements,
                (size_t)cfg.workgroupSize, numDevices);

    read_from_device(queues[numDevices - 1], mainBuf, data, bytes, cfg.useMap);

    std::cout << "Output (x^2 + " << cfg.offset << "):" << std::endl;
    print_int4_array(data, cfg.numElements, totalInts, numDevices);

    if (verify_results(data, cfg.numElements, cfg.offset))
        std::cout << "Verification: PASS" << std::endl;

    cleanup(context, program, mainBuf, subBuffers, queues, sqKernels,
            offKernels, platformIDs, deviceIDs, data);

    return 0;
}
