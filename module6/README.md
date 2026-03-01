# Module 6 - CUDA Streams and Events

Simulates a 4-channel WDM ring resonator PDM sweep pipeline using CUDA streams
and events. Three kernels per channel: Lorentzian response generation, moving
average smoothing, and parallel reduction peak detection. Benchmarks concurrent
stream execution against a sequential baseline.

## Build and Run

```bash
make
./wdm_pdm_sweep <total_threads> <block_size>
```

## Example

```bash
./wdm_pdm_sweep 1048576 256
./wdm_pdm_sweep 1048576 128
./wdm_pdm_sweep 1048576 512
```
