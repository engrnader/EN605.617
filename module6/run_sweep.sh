# Run WDM PDM sweep with 3 block sizes for assignment submission.

TOTAL=1048576

for BS in 128 256 512; do
    ./wdm_pdm_sweep $TOTAL $BS
    echo ""
done
