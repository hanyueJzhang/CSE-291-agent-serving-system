#!/usr/bin/env bash
# Download the Mooncake FAST'25 toolagent trace into workload/traces/ (gitignored).
# Pinned to the upstream commit that added it; the checksum guards against a changed file.
set -euo pipefail
cd "$(dirname "$0")/.."
TRACE_COMMIT=3cca71daccf2a7afb8fe3f0295358f70e3a69fdb   # kvcache-ai/Mooncake "add FAST25 release", 2025-02-21
TRACE_URL=https://raw.githubusercontent.com/kvcache-ai/Mooncake/$TRACE_COMMIT/FAST25-release/traces/toolagent_trace.jsonl
TRACE_SHA256=48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71
OUT=workload/traces/toolagent_trace.jsonl

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

mkdir -p workload/traces
if [ -f "$OUT" ] && [ "$(sha256 "$OUT")" = "$TRACE_SHA256" ]; then echo "$OUT exists"; exit 0; fi
curl -fsSL "$TRACE_URL" -o "$OUT.tmp"
if [ "$(sha256 "$OUT.tmp")" != "$TRACE_SHA256" ]; then
  echo "checksum mismatch for $TRACE_URL" >&2; rm -f "$OUT.tmp"; exit 1
fi
mv "$OUT.tmp" "$OUT"
echo "downloaded $OUT"
