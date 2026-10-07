#!/bin/bash
# ----------------------------------------------------------------------------
# setup_env.sh -- sets every environment variable the tutorial relies on.
#
# SOURCE it (do not run it), so the variables stay in your shell:
#
#     source ~/spine_s3df_tutorial/00_s3df_basics/scripts/setup_env.sh
#
# You can put that line at the end of your ~/.bashrc so it happens every time
# you log in. It is safe to source it more than once.
#
# Anything you `export` BEFORE sourcing this file wins over the defaults below,
# e.g.   export WORKDIR=/sdf/data/neutrino/$USER/spine_tutorial
# ----------------------------------------------------------------------------

# ---- 1. Things YOU may need to change --------------------------------------

# Your Slurm account, written as <facility>:<repo>. Every batch job and
# interactive compute session is charged to it. Find yours with:
#     sacctmgr show assoc user=$USER format=Account%40
# Typical neutrino ML repos: neutrino:ml-dev, neutrino:dune-ml, neutrino:icarus-ml
export SPINE_ACCOUNT=${SPINE_ACCOUNT:-}

# Where you cloned spine-prod (see 00_s3df_basics/README.md, "One-time setup").
export SPINE_PROD_BASEDIR=${SPINE_PROD_BASEDIR:-$HOME/spine-prod}

# Only if you had to build your own container (README step 5.5), uncomment
# and use the FULL path of the .sif file you pulled:
# export SPINE_CONTAINER_PATH=/sdf/scratch/users/${USER:0:1}/$USER/spine_tutorial/images/spine_v1-4-0.sif

# ---- 2. Things you normally do NOT need to change --------------------------

# Folder this tutorial lives in (computed from this file's own location).
SPINE_TUTORIAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export SPINE_TUTORIAL

# Are we on S3DF? (/sdf only exists there)
if [[ -d /sdf/data/neutrino ]]; then
    export ON_S3DF=1
else
    export ON_S3DF=0
fi

# spine-prod's own setup: defines SPINE_CONFIG_PATH (where configs are looked
# up), SPINE_CONTAINER_PATH (the .sif image), etc. Only done once per shell.
if [[ -z $SPINE_PROD_CONFIGURED ]]; then
    if [[ -f $SPINE_PROD_BASEDIR/configure.sh ]]; then
        source "$SPINE_PROD_BASEDIR/configure.sh" > /dev/null
    else
        echo "WARNING: $SPINE_PROD_BASEDIR/configure.sh not found."
        echo "         Clone spine-prod first (00_s3df_basics/README.md, step 3)"
        echo "         or export SPINE_PROD_BASEDIR=/path/to/spine-prod before sourcing."
    fi
fi

# WORKDIR = where all big files (data, outputs, trained weights) go.
# NOT your home directory: $HOME is only 30 GB.
if [[ -z $WORKDIR ]]; then
    if [[ $ON_S3DF == 1 ]]; then
        SCRATCH=/sdf/scratch/users/${USER:0:1}/$USER
        if [[ -d $SCRATCH ]]; then
            WORKDIR=$SCRATCH/spine_tutorial
        else
            echo "WARNING: $SCRATCH does not exist; using \$HOME/spine_tutorial_work."
            echo "         Ask on #comp-sdf for your scratch area and set WORKDIR."
            WORKDIR=$HOME/spine_tutorial_work
        fi
    else
        WORKDIR=$SPINE_TUTORIAL/work      # laptop / off-site testing
    fi
fi
export WORKDIR
export TUTORIAL_DATA=$WORKDIR/data
mkdir -p "$WORKDIR" "$TUTORIAL_DATA"

# Downloaded model weights are cached here (and re-used by every job).
export SPINE_CACHE_DIR=${SPINE_CACHE_DIR:-$SPINE_PROD_BASEDIR/.cache/weights}

# numba (used by SPINE post-processing) crashes if it tries to use more
# threads than allowed. Cap it at the number of cores we have (max 64).
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export NUMBA_NUM_THREADS=$(( NCORES < 64 ? NCORES : 64 ))

# Where the small public example files can be downloaded from.
export SPINE_SAMPLE_URL=https://s3df.slac.stanford.edu/data/neutrino/spine/workshop

# Bigger official datasets (only exist on S3DF).
if [[ $ON_S3DF == 1 ]]; then
    export GENERIC_TRAIN=/sdf/data/neutrino/generic/mpvmpr_2020_01_v04/train.root
    export GENERIC_TEST=/sdf/data/neutrino/generic/mpvmpr_2020_01_v04/test.root
    export PDSP_TRAIN_LIST=/sdf/data/neutrino/pdune/sim/sp/mpvmpr_v1/train_file_list.txt
    export PDSP_TEST_LIST=/sdf/data/neutrino/pdune/sim/sp/mpvmpr_v1/test_file_list.txt
else
    export GENERIC_TRAIN=$TUTORIAL_DATA/generic_small.root
    export GENERIC_TEST=$TUTORIAL_DATA/generic_small.root
fi

# Make the helper scripts callable by name (spine_container.sh, compute_shell.sh).
case ":$PATH:" in
    *":$SPINE_TUTORIAL/00_s3df_basics/scripts:"*) ;;
    *) export PATH=$SPINE_TUTORIAL/00_s3df_basics/scripts:$PATH ;;
esac

# ---- 3. Summary -------------------------------------------------------------
if [[ -z $SPINE_TUTORIAL_QUIET ]]; then
    echo "SPINE tutorial environment"
    echo "  SPINE_TUTORIAL       = $SPINE_TUTORIAL"
    echo "  SPINE_PROD_BASEDIR   = $SPINE_PROD_BASEDIR"
    echo "  SPINE_CONTAINER_PATH = ${SPINE_CONTAINER_PATH:-<not set>}"
    echo "  WORKDIR              = $WORKDIR"
    echo "  SPINE_ACCOUNT        = ${SPINE_ACCOUNT:-<NOT SET -- edit setup_env.sh or export it>}"
fi
