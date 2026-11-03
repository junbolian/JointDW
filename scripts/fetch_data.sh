#!/usr/bin/env bash
# Download the data of release v1.0 into data/, verify its SHA-256 and unpack it.
# JOINTDW_DATA_URL overrides the location of the release asset (a directory URL, for example file:///path).
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${JOINTDW_DATA_URL:-https://github.com/junbolian/JointDW/releases/download/v1.0}"
SHA256=8d34a69a2e5ad3c23a7c7c6a200377fbfa084b6f4e8e9275d696e4e584282d4f
mkdir -p data
curl -fsSL -o data/jointdw_data.tar.gz "$URL/jointdw_data.tar.gz"
echo "$SHA256  data/jointdw_data.tar.gz" | sha256sum -c -
tar -xzf data/jointdw_data.tar.gz
echo "data/ is ready"
