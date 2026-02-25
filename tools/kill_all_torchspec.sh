#!/bin/bash

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    echo "Killing torchspec inference workers..."
    pgrep -f 'torchspec\.target\.remote_backend' | xargs -r kill -9
    echo "Killing mooncake master..."
    pgrep -f 'mooncake_master' | xargs -r kill -9

else
    nvidia-smi

    echo "Killing torchspec inference workers..."
    pgrep -f 'torchspec\.target\.remote_backend' | xargs -r kill -9
    echo "Killing mooncake master..."
    pgrep -f 'mooncake_master' | xargs -r kill -9

    if [ $# -gt 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y lsof
        else
            apt-get update
            apt-get install -y lsof
        fi
        kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}') 2>/dev/null
        lsof /dev/nvidia* | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi

    nvidia-smi
fi
