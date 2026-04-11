# Module 12: OpenCL Buffers, Sub-Buffers, and Vector Types

OpenCL program that squares int4 vectors then adds an offset, using sub-buffers to partition work across devices. Supports buffer copy and memory-mapped transfers.

## Build and Run

**Windows (run.bat):**
```
run.bat
```

**Linux (Makefile):**
```
make && ./assignment
```

## Options

```
./assignment [--size N] [--workgroup N] [--offset N] [--useMap]
```
