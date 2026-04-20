#!/bin/bash
# select_gpu_device wrapper script

export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export TRITON_HOME="/dev/shm/$USER/.cache/triton/triton_${RANK}"
export TRITON_CACHE_DIR="/dev/shm/$USER/.cache/triton/triton_${RANK}"
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/$USER/.cache/torchinductor/torchinductor_${RANK}"
TRACE_DIR="traces/baseline"
mkdir -p $TRACE_DIR

 
cmd="nsys profile -o $TRACE_DIR/plexustrace_${SLURM_JOB_ID}_${RANK} -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop $*"

echo $cmd
exec $cmd
