#! /usr/bin/env bash
set -uxo pipefail

if [ ! "$(command -v fuse-overlayfs)" ]
then
    echo "fuse-overlayfs not found installing - please update to our latest image"
    apt update -y
    apt install -o DPkg::Lock::Timeout=120 -y psmisc libfuse3-dev fuse-overlayfs
fi

mkdir -p ${PERSISTENT_CHECKPOINT_DIR}
if [[ "${1:-}" == 'ngrok' ]]; then
    wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
    tar -xvf ngrok-v3-stable-linux-amd64.tgz
    /notebooks/ngrok config add-authtoken ${2}
    /notebooks/ngrok http ${APP_PORT} | tee ngrok-live.out &
fi
echo "Starting preparation of datasets"
/notebooks/.gradient/symlink_datasets_and_caches.py


echo "Finished running setup.sh."
# Run automated test if specified
if [[ "${1:-}" == 'test' ]]; then
    /notebooks/.gradient/automated-test.sh "${@:2}"
elif [[ "${2:-}" == 'test' ]]; then
    /notebooks/.gradient/automated-test.sh "${@:3}"
fi
