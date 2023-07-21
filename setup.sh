#!/bin/bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Script to be sourced on launch of the Gradient Notebook

# called from root folder in container
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXIT_CODE=0
echo "Graphcore setup - Starting notebook setup"
DETECTED_NUMBER_OF_IPUS=$(python .gradient/available_ipus.py)
if [[ "$1" == "test" ]]; then
    IPU_ARG="${DETECTED_NUMBER_OF_IPUS}"
else
    IPU_ARG=${1:-"${DETECTED_NUMBER_OF_IPUS}"}
fi
echo "Graphcore setup - Detected ${DETECTED_NUMBER_OF_IPUS} IPUs"
if [[ "${DETECTED_NUMBER_OF_IPUS}" == "0" ]]; then
    echo "=============================================================================="
    echo "                         IPU ERROR  DETECTED"
    echo "=============================================================================="
    echo "Connection to IPUs timed-out. This error indicates a problem with the "
    echo "hardware you are running on. Please contact Paperspace Support at "
    echo " https://docs.paperspace.com/contact-support/ "
    echo " referencing the Notebook ID: ${PAPERSPACE_METRIC_WORKLOAD_ID:-unknown}"
    echo "=============================================================================="
    exit 255
fi
# Check the state of the partition
GC_INFO_OUTPUT=$(timeout 5 gc-info -l 2>&1)
if [[ "$(echo ${GC_INFO_OUTPUT} | grep 'Partition.* \[active\]')" ]]
then
    echo "Graphcore setup - Partition check - passed"
elif [[ "$(echo ${GC_INFO_OUTPUT} | grep 'partition is not ACTIVE')" ]]
then
    echo "=============================================================================="
    echo "                         IPU ERROR  DETECTED"
    echo "=============================================================================="
    echo " IPU Partition is not active. This error indicates a problem with the "
    echo "hardware you are running on. Please contact Paperspace Support at "
    echo " https://docs.paperspace.com/contact-support/ "
    echo " referencing the Notebook ID: ${PAPERSPACE_METRIC_WORKLOAD_ID:-unknown}"
    echo "=============================================================================="
    gc-info -l
    exit 254
else
    echo "[WARNING] IPU Partition in an unrecognised state - Notebook will start normally but"
    echo "[WARNING] you may encounter hardware related errors. Get in touch with Paperspace and/or"
    echo "[WARNING] Graphcore support if you encounter unexpected behaviours or errors."
    EXIT_CODE=253
fi

export NUM_AVAILABLE_IPU=${IPU_ARG}
export GRAPHCORE_POD_TYPE="pod${IPU_ARG}"
export POPLAR_EXECUTABLE_CACHE_DIR="/tmp/exe_cache/${SDK_VERSION}"
export DATASETS_DIR="/tmp/dataset_cache"
export CHECKPOINT_DIR="/tmp/checkpoints"
export PERSISTENT_CHECKPOINT_DIR="/storage/ipu-checkpoints/"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CACHE_DIR="/tmp"


# mounted public dataset directory (path in the container)
# in the Paperspace environment this would be ="/datasets"
export PUBLIC_DATASETS_DIR="/datasets"

export HUGGINGFACE_HUB_CACHE="/tmp/huggingface_caches"
export TRANSFORMERS_CACHE="/tmp/huggingface_caches/checkpoints"
export HF_DATASETS_CACHE="/tmp/huggingface_caches/datasets"

# Set framework specific variables
export POPTORCH_CACHE_DIR="${POPLAR_EXECUTABLE_CACHE_DIR}"
export POPTORCH_LOG_LEVEL=ERR
export RDMAV_FORK_SAFE=1
export POPART_PRELOAD_POPEF="full-preload"

# Logger specific vars
export TIER_TYPE=$(python .gradient/check_tier.py)
export FIREHOSE_STREAM_NAME="paperspacenotebook_production"
export GCLOGGER_CONFIG="${PUBLIC_DATASETS_DIR}/gcl"
export REPO_FRAMEWORK="Hugging Face"

echo "Graphcore setup - Spawning dataset preparation process"
nohup /notebooks/.gradient/prepare-datasets.sh ${@} & tail -f nohup.out &

export PIP_DISABLE_PIP_VERSION_CHECK=1 CACHE_DIR=/tmp
echo "Graphcore setup - Starting Jupyter kernel"
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True \
            --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True \
            --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True

exit $EXIT_CODE