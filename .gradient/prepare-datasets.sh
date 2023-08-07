#! /usr/bin/env bash
set -euxo pipefail
run-tests() {
    echo "PAPERSPACE-AUTOMATED-TESTING: Started testing"
    if [ "${8}" == "unset" ]; then
        EXAMPLES_UTILS_REV=latest_stable
    else
        EXAMPLES_UTILS_REV=${8}
    fi
    python -m pip install gradient
    python -m pip install "examples-utils[jupyter] @ git+https://github.com/graphcore/examples-utils@${EXAMPLES_UTILS_REV}"

    # set variable matching the standard Paperspace entry point
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export HUGGING_FACE_HUB_TOKEN=${7}
    export VIRTUAL_ENV="/some/fake/venv/GC-automated-paperspace-test-${4}"

    LOG_FOLDER="${5}/log_${4}_$(date +'%Y-%m-%d-%H_%M_%S')"
    mkdir -p ${LOG_FOLDER}
    TEST_CONFIG_FILE="${6}"
    # Run the health check script
    HEALTH_CHECK_LOG_FOLDER="/storage/graphcore_health_checks"
    python -m graphcore_cloud_tools.paperspace_utils.health_check --log-folder ${HEALTH_CHECK_LOG_FOLDER}
    # Copy the health check logs to local log folder
    HEALTH_CHECK_LOG_FILE=$(find ${HEALTH_CHECK_LOG_FOLDER} -type f | sort -n | tail -1)
    cp ${HEALTH_CHECK_LOG_FILE} ${LOG_FOLDER}

    cd /notebooks/
    echo "PAPERSPACE-AUTOMATED-TESTING: starting platform_assessment testing"
    python -m examples_utils platform_assessment --spec ${TEST_CONFIG_FILE} "${@:9}" \
        --log-dir $LOG_FOLDER \
        --gc-monitor \
        --cloning-directory /tmp/clones \
        --additional-metrics

    exit_code=$?
    tar -czvf "${LOG_FOLDER}.tar.gz" ${LOG_FOLDER}
    echo "PAPERSPACE-AUTOMATED-TESTING: Testing complete with exit code ${exit_code}"
    echo "Shutting down notebook"

    sleep 5
    gradient apiKey ${1}
    gradient notebooks stop --id ${PAPERSPACE_METRIC_WORKLOAD_ID}
    echo "Notebook Stopped"
}

if [ ! "$(command -v fuse-overlayfs)" ]; then
    echo "fuse-overlayfs not found installing - please update to our latest image"
    apt update -y
    apt install -o DPkg::Lock::Timeout=120 -y psmisc libfuse3-dev fuse-overlayfs
fi

python -m pip install "graphcore-cloud-tools[logger] @ git+https://github.com/graphcore/graphcore-cloud-tools@v0.1"

echo "Starting preparation of datasets"
python -m graphcore_cloud_tools paperspace symlinks --path "$( dirname -- "${BASH_SOURCE[0]}" )"/symlink_config.json

echo "Finished running prepare-datasets.sh"
# Run automated test if specified
if [[ "${1:-}" == 'test' ]]; then
    ARGS="${@:2}"
elif [[ "${2:-}" == 'test' ]]; then
    ARGS="${@:3}"
fi
[ -n "${ARGS+x}" ] && run-tests $ARGS

echo "Finished running setup.sh."
