#!/bin/bash
# select_gpu_device wrapper script
ulimit -c 0
export RANK=${FLUX_TASK_RANK}
exec "$@"