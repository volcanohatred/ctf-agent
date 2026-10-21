#!/bin/bash
# apt-install PKGS... — apt-get install with retries (apt cache resumes
# partial downloads between attempts). Fails the build only if all retries
# are exhausted.
for i in 1 2 3 4 5; do
  apt-get install -y "$@" && exit 0
  echo "apt-install: attempt $i failed — retrying in 10s" >&2
  sleep 10
done
echo "apt-install: all retries failed for: $*" >&2
exit 1
