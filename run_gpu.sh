#!/bin/bash
# PhysisML — launch on the Intel Arc A370M GPU
# Uses the physisml_gpu conda env with IPEX XPU

CONDA_PYTHON="/home/speleoalex/miniforge3/envs/physisml_gpu/bin/python"

if [ ! -f "$CONDA_PYTHON" ]; then
  echo "Conda env 'physisml_gpu' not found."
  echo "See: docs/it/setup/gpu_intel_arc.md"
  exit 1
fi

# Activate oneAPI (required for XPU)
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

echo "GPU: Intel Arc A370M (XPU)"
echo "Running: $@"
exec "$CONDA_PYTHON" "$@"