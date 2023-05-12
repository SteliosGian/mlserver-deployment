#!/usr/bin/env bash

MODE=${1:-inference}

set -euo pipefail
IFS=$'\n\t'
trap "exit 1" HUP INT QUIT ABRT SEGV

docker build --pull --tag fake-news-detector:local .

if [[ $MODE == "inference" ]]
then 
    docker run --user root --rm -it -p 8080:8080 fake-news-detector:local
elif [[ $MODE == "training" ]] 
then
    docker run --user root --rm -it -p 8080:8080 --mount type=bind,source=./src/inference/temp_model,target=/app/inference/temp_model fake-news-detector:local python fake_news_detector/train.py
else
    echo "Invalid argument. Only inference/training are allowed."
fi
