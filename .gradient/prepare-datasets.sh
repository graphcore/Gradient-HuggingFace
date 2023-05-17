#! /usr/bin/env bash
set -euxo pipefail
run-tests() {

    # set variable matching the standard Paperspace entry point
    export PIP_DISABLE_PIP_VERSION_CHECK=1

    export VIRTUAL_ENV="/some/fake/venv/GC-automated-paperspace-test-${4}"
    LOG_FOLDER="${5}/log_${4}_$(date +'%Y-%m-%d-%H_%M_%S')"
    mkdir -p ${LOG_FOLDER}
    TEST_CONFIG_FILE="${6}"

    cd /notebooks/
    python -m examples_utils platform_assessment --spec ${TEST_CONFIG_FILE} "${@:9}" \
        --log-dir $LOG_FOLDER \
        --gc-monitor \
        --cloning-directory /tmp/clones \
        --additional-metrics

    exit_code=$?
    tar -czvf "${LOG_FOLDER}.tar.gz" ${LOG_FOLDER}
    echo "PAPERSPACE-AUTOMATED-TESTING: Testing complete with exit code ${exit_code}"
}

if [ ! "$(command -v fuse-overlayfs)" ]; then
    echo "fuse-overlayfs not found installing - please update to our latest image"
    apt update -y
    apt install -o DPkg::Lock::Timeout=120 -y psmisc libfuse3-dev fuse-overlayfs
fi


[ "${8}" == "unset" ] && EXAMPLES_UTILS_REV=latest_stable || EXAMPLES_UTILS_REV=${8}

python -m pip install "examples-utils[jupyter] @ git+https://github.com/graphcore/examples-utils@${EXAMPLES_UTILS_REV}" --use-feature=fast-deps
python -m pip install gradient


mkdir -p ${PERSISTENT_CHECKPOINT_DIR}
echo "Starting preparation of datasets"
python -m examples_utils paperspace symlinks --path "$( dirname -- "${BASH_SOURCE[0]}" )"/symlink_config.json
# /notebooks/.gradient/symlink_datasets_and_caches.py

# pre-install the correct version of optimum for this release
python3 -m pip install "optimum-graphcore>=0.5, <0.6"

echo "Finished running setup.sh."

# Run automated test if specified
if [[ "${1:-}" == 'test' ]]; then
    run-tests "${@:2}"
elif [[ "${2:-}" == 'test' ]]; then
    run-tests "${@:3}"
fi

echo "Test finished shutting down notebook"
sleep 5
gradient apiKey ${1}
gradient notebooks stop --id ${PAPERSPACE_METRIC_WORKLOAD_ID}