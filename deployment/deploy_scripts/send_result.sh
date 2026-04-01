#!/bin/bash

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-4242}"
AET="${AET:-HIPPOAI}"
REPORT_DCM_PATH="${REPORT_DCM_PATH:-/datadrive/out/report.dcm}"

storescu "${HOST}" "${PORT}" -v -aec "${AET}" "${REPORT_DCM_PATH}"