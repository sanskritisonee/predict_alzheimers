#!/bin/bash

set -euo pipefail

ORTHANC_URL="${ORTHANC_URL:-http://localhost:8042}"
ROUTE_SCRIPT="${ROUTE_SCRIPT:-route_dicoms.lua}"
SCP_PORT="${SCP_PORT:-106}"
AET="${AET:-HIPPOAI}"
STUDIES_DIR="${STUDIES_DIR:-./studies}"

mkdir -p "${STUDIES_DIR}"

curl -X POST "${ORTHANC_URL}/tools/execute-script" --data-binary "@${ROUTE_SCRIPT}" -v
storescp "${SCP_PORT}" -v -aet "${AET}" -od "${STUDIES_DIR}" --sort-on-study-uid st