#!/bin/bash

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Parse command line arguments
# Usage: ./build_conda.sh [MODE]
#   1       - Create new micromamba env and install (default)
#   current - Install into current environment
#   0       - Skip env creation and installation
MODE="${1:-1}"

if [ "$MODE" = "1" ]; then
    if ! command -v micromamba &> /dev/null; then
        echo "Error: micromamba is not installed."
        echo "Please install it first: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
        exit 1
    fi

    # Initialize micromamba for this script
    export MAMBA_EXE="${MAMBA_EXE:-$(command -v micromamba)}"
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "$("$MAMBA_EXE" shell hook --shell bash)"

    micromamba create -n torchspec python=3.12 uv -c conda-forge -y
elif [ "$MODE" = "current" ]; then
    echo "Using current environment: $(python3 --version), $(which python3)"
else
    echo "Skipping micromamba setup (mode=0)"
fi

SGLANG_VERSION="${SGLANG_VERSION:-v0.5.8.post1}"
SGLANG_COMMIT=0f2df9370a1de1b4fb11b071d39ab3ce2287a350
SGLANG_FOLDER_NAME="_sglang"

# Install sglang inside the conda environment
if [ ! -d "$PROJECT_ROOT/$SGLANG_FOLDER_NAME" ]; then
    git clone https://github.com/sgl-project/sglang.git "$PROJECT_ROOT/$SGLANG_FOLDER_NAME"
fi

# Avoid pythonpath conflict, because we are using the offline engine.
cd "$PROJECT_ROOT/$SGLANG_FOLDER_NAME"
git checkout $SGLANG_COMMIT
git reset --hard HEAD

cd "$PROJECT_ROOT"

if [ "$MODE" = "1" ]; then
    micromamba run -n torchspec pip install -e "${SGLANG_FOLDER_NAME}/python[all]"
    micromamba run -n torchspec uv pip install -e ".[dev]"

    echo "torchspec environment setup complete!"
    echo "Activate with: micromamba activate torchspec"
elif [ "$MODE" = "current" ]; then
    pip install -e "${SGLANG_FOLDER_NAME}/python[all]"
    pip install -e ".[dev]"

    echo "torchspec installed into current environment!"
else
    echo "Skipping package installation (mode=0)"
    echo "Please install packages manually:"
    echo "  pip install -e \"${SGLANG_FOLDER_NAME}/python[all]\""
    echo "  pip install -e \".[dev]\""
fi

cd "$PROJECT_ROOT/$SGLANG_FOLDER_NAME"

# Apply sglang patch (matches Docker build behavior)
git apply "$PROJECT_ROOT/patches/sglang/$SGLANG_VERSION/sglang.patch"
