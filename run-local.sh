#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
trap "exit 1" HUP INT QUIT ABRT SEGV

docker build --pull --tag fake-news-detector:local .
docker run --user root --rm -it fake-news-detector:local
