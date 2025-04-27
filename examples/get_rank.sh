#!/bin/bash
# select_gpu_device wrapper script
ulimit -c 0
export RANK=${SLURM_PROCID}
exec $*
