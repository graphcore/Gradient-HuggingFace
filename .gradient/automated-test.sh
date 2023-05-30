#! /usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# The entry point for the automated testing on the Paperspace platform
# this script is meant to be launched in the docker image as the entry point.
#
# Arguments:
# 1: Gradient API key
# 2: Dataset ID
# 3: Version ID
# 4: Either the runtime in which we are running or 'upload-reports'
# 5: Folder in which to save/look for tar.gz report archives
# 6: Examples utils spec file to process and benchmark
# 7: Huggingface token
# 8: examples-utils revision
# @:9 other arguments are passed to the `examples_utils platform_assesment` command

set -x

upload_report() {
    # Uploads files to a gradient dataset
    python -m pip install gradient

    gradient apiKey ${1}

    for file in `find ${5} -name "*.tar.gz"`
    do
        echo uploading $file
        gradient datasets files put --id ${2}:${3} --source-path $file
    done
    gradient datasets versions commit --id ${2}:${3}

    echo Committed version: ${2}:${3}
}

run_tests(){
    git submodule update -j 5 --init
    # set variable matching the standard Paperspace entry point
    export PIP_DISABLE_PIP_VERSION_CHECK=1

    export VIRTUAL_ENV="/some/fake/venv/GC-automated-paperspace-test-${4}"
    LOG_FOLDER="${5}/log_${4}_$(date +'%Y-%m-%d-%H_%M_%S')"
    TEST_CONFIG_FILE="${6}"
    mkdir -p ${LOG_FOLDER}
    # Run the health check script
    # HEALTH_CHECK_LOG_FOLDER="/storage/graphcore_health_checks"
    # python -m examples_utils.paperspace_utils.health_check --log-folder ${HEALTH_CHECK_LOG_FOLDER}
    # # Copy the health check logs to local log folder
    # HEALTH_CHECK_LOG_FILE=$(find ${HEALTH_CHECK_LOG_FOLDER} -type f | sort -n | tail -1)
    # cp ${HEALTH_CHECK_LOG_FILE} ${LOG_FOLDER}
    cd /notebooks/
    stdbuf -oL -eL python -m examples_utils platform_assessment --spec ${TEST_CONFIG_FILE} "${@:9}" \
        --log-dir $LOG_FOLDER \
        --gc-monitor \
        --cloning-directory /tmp/clones \
        --additional-metrics

    tar -czvf "${LOG_FOLDER}.tar.gz" ${LOG_FOLDER}
    echo "PAPERSPACE-AUTOMATED-TESTING: Testing complete"
}
# Prep the huggingface token
export HUGGING_FACE_HUB_TOKEN=${7}
if [ "${8}" == "unset" ]; then
    EXAMPLES_UTILS_REV=latest_stable
else
    EXAMPLES_UTILS_REV=${8}
fi

python -m pip install "examples-utils[jupyter] @ git+https://github.com/graphcore/examples-utils@${EXAMPLES_UTILS_REV}"
#python -m pip install gradient
# In sh single equal is needed for string compare.
if [ "${4}" = "upload-reports" ]
then
    echo "Uploading report"
    upload_report ${@}
else
    echo "Running tests"
    run_tests ${@}
fi
# Make the notebook stop itself
echo "Test finished shutting down notebook"
sleep 5
gradient apiKey ${1}
gradient notebooks stop --id ${PAPERSPACE_METRIC_WORKLOAD_ID}
