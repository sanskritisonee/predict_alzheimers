#!/bin/bash

set -euo pipefail

# This script sends a study to Orthanc/StoreSCP.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-4242}"
AET="${AET:-HIPPOAI}"
TEST_STUDY_DIR="${TEST_STUDY_DIR:-/data/TestVolumes/Study1}"

storescu "${HOST}" "${PORT}" -v -aec "${AET}" +r +sd "${TEST_STUDY_DIR}"
