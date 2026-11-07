#!/usr/bin/env bash
# Download the data of release v1.1 into data/, verify its SHA-256 and unpack it.
# JOINTDW_DATA_URL overrides the location of the release asset (a directory URL, for example file:///path).
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${JOINTDW_DATA_URL:-https://github.com/junbolian/JointDW/releases/download/v1.1}"
NAME=jointdw_data_v1.1.tar.gz
SHA256=e7493282474adc1a7c663081c974dd42c79154a8a29ef813ccc31e15a33ebfed
mkdir -p data
curl -fsSL -o "data/$NAME" "$URL/$NAME"
echo "$SHA256  data/$NAME" | sha256sum -c -
tar -xzf "data/$NAME"
echo "data/ is ready"
