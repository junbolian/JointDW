#!/usr/bin/env bash
# Download the data of release v1.2.1 into data/, check it against the SHA-256 recorded below and unpack it.
# JOINTDW_DATA_URL overrides the location of the release asset (a directory URL, for example file:///path).
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${JOINTDW_DATA_URL:-https://github.com/junbolian/JointDW/releases/download/v1.2.1}"
NAME=jointdw_data_v1.2.1.tar.gz
EXPECTED=cf63f063a95aeb9ec8d997a5cb0038a29edcfdccc87a9ff9415679c7f66bae29
mkdir -p data
curl -fsSL -o "data/$NAME" "$URL/$NAME"
python - "data/$NAME" "$EXPECTED" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for block in iter(lambda: fh.read(1 << 20), b""):
        h.update(block)
if h.hexdigest() != sys.argv[2]:
    sys.exit(f"{sys.argv[1]}: SHA-256 {h.hexdigest()} differs from the recorded {sys.argv[2]}")
print(f"{sys.argv[1]}: SHA-256 as recorded")
PY
tar -xzf "data/$NAME"
echo "data/ is ready"
