#!/usr/bin/env bash
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

###
# Here run your tests
# JUnit XML files with the pattern `*report.xml` will be published as a test report
###

# cd to app root directory and add `utils/examples_tests` to PYTHONPATH
cd "$(dirname "$0")"/..
export PYTHONPATH=$(cd ../../../utils; pwd):$PYTHONPATH

# Run tests
python3 -m pytest --junitxml=report.xml
