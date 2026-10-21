#!/bin/bash
# retry N CMD [ARGS...] — run CMD up to N times with linear backoff.
# Used by the Dockerfile for network operations that can flake (proxy resets).
n=$1; shift
for i in $(seq 1 "$n"); do
  "$@" && exit 0
  s=$((i * 10))
  echo "retry: attempt $i/$n failed ($*) — sleeping ${s}s" >&2
  sleep "$s"
done
echo "retry: all $n attempts failed for: $*" >&2
exit 1
