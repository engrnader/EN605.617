/**
 * @file    assignment.cl
 * @brief   OpenCL kernels using int4 vector types.
 *
 *          Kernel 1: square each component of an int4 vector
 *          Kernel 2: add a constant offset to each component
 */

__kernel void square(__global int4* buffer) {
    size_t id = get_global_id(0);
    buffer[id] = buffer[id] * buffer[id];
}

__kernel void add_offset(__global int4* buffer, int offset) {
    size_t id = get_global_id(0);
    buffer[id] = buffer[id] + (int4)(offset);
}
