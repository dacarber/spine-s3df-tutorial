#!/bin/bash
# ----------------------------------------------------------------------------
# spine_container.sh -- run a command INSIDE the SPINE container.
#
# The container (.sif file) has everything SPINE needs pre-installed:
# python, PyTorch + CUDA, MinkowskiEngine, ROOT, LArCV, SPINE itself, Jupyter.
# You never install those yourself.
#
# Usage:
#     spine_container.sh spine --version           # run one command
#     spine_container.sh python my_script.py
#     spine_container.sh                           # open a shell inside it
#
# What it does: apptainer exec --nv --bind /sdf,... $SPINE_CONTAINER_PATH <cmd>
#   --nv     lets the container see the node's GPUs (harmless on CPU nodes)
#   --bind   makes /sdf (and friends) visible inside the container
# ----------------------------------------------------------------------------

# Already inside a container (e.g. a Jupyter terminal)? Just run the command.
if [[ -n $APPTAINER_CONTAINER || -n $SINGULARITY_CONTAINER ]]; then
    if [[ $# -eq 0 ]]; then exec bash; else exec "$@"; fi
fi

if [[ -z $SPINE_CONTAINER_PATH ]]; then
    echo "ERROR: SPINE_CONTAINER_PATH is not set. Source setup_env.sh first."
    exit 1
fi
if [[ ! -f $SPINE_CONTAINER_PATH ]]; then
    echo "ERROR: container not found: $SPINE_CONTAINER_PATH"
    echo "       See 00_s3df_basics/README.md, 'If the container is missing'."
    exit 1
fi

RUNNER=$(command -v apptainer || command -v singularity)
if [[ -z $RUNNER ]]; then
    echo "ERROR: neither apptainer nor singularity is available on $(hostname)."
    exit 1
fi

BINDS=/sdf
[[ -d /fs ]]       && BINDS+=,/fs
[[ -d /lscratch ]] && BINDS+=,/lscratch

# Ignore python packages you may have pip-installed in ~/.local on the host:
# they can clash with the container's own versions.
export PYTHONNOUSERSITE=1

if [[ $# -eq 0 ]]; then
    set -- bash
fi
exec "$RUNNER" exec --nv --bind "$BINDS" "$SPINE_CONTAINER_PATH" "$@"
