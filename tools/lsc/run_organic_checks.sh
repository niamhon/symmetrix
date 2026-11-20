#!/bin/bash -l
# Run within the user's existing one-H100 allocation. Never submit a new job.
set -euo pipefail
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
repo=/mnt/home/noneill/Programs/symmetrix-dev
export LSC_ORGANIC_DATA="${LSC_ORGANIC_DATA:-/mnt/home/noneill/ceph/symmetrix-lsc-validation/organic-stage3}"
export LSC_LAMMPS_BINARY="$repo/.lsc-reference-logs/kokkos-charge-sync-20261119/bin/lmp-fixed"
export LSC_LAMMPS_LAUNCHER='/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun --bind-to none -np 1'
export LSC_REQUIRE_LAMMPS=1 LSC_RUN_ORGANIC=1 LSC_ORGANIC_REUSE_RESULTS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
: "${SLURM_JOB_ID:?Run this script inside the existing GPU allocation}"
pilot_cache="/tmp/noneill-organic-pilot-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$pilot_cache/cache" CUDA_CACHE_PATH="$pilot_cache/cuda" TMPDIR="$pilot_cache/tmp" MPLCONFIGDIR="$pilot_cache/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR" "$MPLCONFIGDIR"
label="${1:-run1}"
if (($#)); then shift; fi
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
work="$LSC_ORGANIC_DATA/checks/$label"
log="$LSC_ORGANIC_DATA/results/pytest-$label.log"
xml="$LSC_ORGANIC_DATA/results/pytest-$label.xml"
if [[ -e "$work" || -e "$log" || -e "$xml" ]]; then
    echo 'Preserving earlier results: choose a new run label.' >&2
    exit 2
fi
cd "$repo"
tests=(
    symmetrix/test/lsc/test_organic_pilot.py
    symmetrix/test/lsc/test_cuda_compact_first.py
    symmetrix/test/lsc/test_cuda_derivatives.py
    symmetrix/test/lsc/test_lammps_lsc_kokkos.py::test_device_pair_outputs
    symmetrix/test/lsc/test_lammps_lsc_kokkos_charges.py::test_charge_sync_after_host_rebuild
    symmetrix/test/lsc/test_lammps_lsc_kokkos_charges.py::test_atomic_without_charge_storage
)
if (($#)); then tests=("$@"); fi
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader > "$LSC_ORGANIC_DATA/results/gpu-$label.txt"
code=0
python -m pytest "${tests[@]}" -q -rA --basetemp="$work" --junitxml="$xml" 2>&1 | tee "$log" || code=$?
python tools/lsc/organic_pilot.py audit
exit "$code"
