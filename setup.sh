#!/bin/bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Script to be sourced on launch of the Gradient Notebook

# called from root folder in container
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Graphcore setup - Starting notebook setup"
DETECTED_NUMBER_OF_IPUS=$(python .gradient/available_ipus.py)
IPU_ARG="${DETECTED_NUMBER_OF_IPUS}"
# if [[ "$1" == "test" ]]; then
# else
#     IPU_ARG=${1:-"${DETECTED_NUMBER_OF_IPUS}"}
# fi
echo "Graphcore setup - Detected ${DETECTED_NUMBER_OF_IPUS} IPUs"
if [[ "${DETECTED_NUMBER_OF_IPUS}" == "0" ]]; then
    echo "=============================================================================="
    echo "                         ERROR  DETECTED"
    echo "=============================================================================="
    echo "Connection to IPUs timed-out. This error indicates a problem with the "
    echo "hardware you are running on. Please contact Paperspace Support at "
    echo " https://docs.paperspace.com/contact-support/ "
    echo " referencing the Notebook ID: ${PAPERSPACE_METRIC_WORKLOAD_ID:-unknown}"
    echo "=============================================================================="
    exit -1
fi

export NUM_AVAILABLE_IPU=${IPU_ARG}
export GRAPHCORE_POD_TYPE="pod${IPU_ARG}"
export POPLAR_EXECUTABLE_CACHE_DIR="/tmp/exe_cache"
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

# Logger specific vars
export TIER_TYPE=$(python .gradient/check_tier.py)
export FIREHOSE_STREAM_NAME="paperspacenotebook_production"
export GCLOGGER_CONFIG="${PUBLIC_DATASETS_DIR}/gcl"
export REPO_FRAMEWORK="Hugging Face"

export APP_PORT=7865

echo "Graphcore setup - Spawning dataset preparation process"
nohup /notebooks/.gradient/prepare-datasets.sh ${@} & tail -f nohup.out &

export PIP_DISABLE_PIP_VERSION_CHECK=1 CACHE_DIR=/tmp
echo "Graphcore setup - Starting Jupyter kernel"
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True \
            --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True \
            --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True
