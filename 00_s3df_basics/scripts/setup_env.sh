#!/bin/bash
# ----------------------------------------------------------------------------
# setup_env.sh -- sets every environment variable the tutorial relies on.
#
# SOURCE it (do not run it), so the variables stay in your shell:
#
#     source ~/spine-s3df-tutorial/00_s3df_basics/scripts/setup_env.sh
#
# You can put that line at the end of your ~/.bashrc so it happens every time
# you log in. It is safe to source it more than once.
#
# Anything you `export` BEFORE sourcing this file wins over the defaults below,
# e.g.   export WORKDIR=/sdf/data/neutrino/$USER/spine_tutorial
# ----------------------------------------------------------------------------

# On the S3DF login nodes (s3dflogin) only $HOME is mounted, so there is
# nothing to set up there: stop quietly. (Matters when this file is sourced
# from ~/.bashrc, which also runs on the login nodes.)
if [[ -d /sdf/home && ! -d /sdf/data/neutrino ]]; then
    return 0 2>/dev/null || exit 0
fi

# Print the summary only in interactive terminals. Output from ~/.bashrc in
# non-interactive sessions (scp, rsync, batch jobs) can break those tools.
if [[ $- != *i* ]]; then
    SPINE_TUTORIAL_QUIET=1
fi

# ---- 1. Things YOU may need to change --------------------------------------

# Your Slurm account, written as <facility>:<repo>. Every batch job and
# interactive compute session is charged to it. Find yours with:
#     sacctmgr show assoc user=$USER format=Account%40
# Typical neutrino ML repos: neutrino:ml-dev, neutrino:dune-ml, neutrino:icarus-ml
#
#   >>> PUT YOUR ACCOUNT BETWEEN THE QUOTES, e.g.  MY_ACCOUNT="neutrino:ml-dev"
MY_ACCOUNT=""
export SPINE_ACCOUNT=${SPINE_ACCOUNT:-$MY_ACCOUNT}

# Which spine-prod to use. By default this is the STANDARD S3DF installation,
#     /sdf/data/neutrino/software/spine-prod
# which is what all official productions are run with. Only uncomment the
# next line if you work on your OWN clone (development / testing changes):
# export SPINE_PROD_BASEDIR=$HOME/spine-prod
SPINE_PROD_OFFICIAL=/sdf/data/neutrino/software/spine-prod
if [[ -z $SPINE_PROD_BASEDIR ]]; then
    if [[ -d $SPINE_PROD_OFFICIAL ]]; then
        SPINE_PROD_BASEDIR=$SPINE_PROD_OFFICIAL
    else
        SPINE_PROD_BASEDIR=$HOME/spine-prod      # off S3DF: a personal clone
    fi
fi
export SPINE_PROD_BASEDIR

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
        echo "WARNING: $SPINE_PROD_BASEDIR/configure.sh not found." >&2
        echo "         On S3DF the standard spine-prod is $SPINE_PROD_OFFICIAL;" >&2
        echo "         elsewhere, clone spine-prod (00_s3df_basics/README.md, step 5.2)" >&2
        echo "         or export SPINE_PROD_BASEDIR=/path/to/spine-prod before sourcing." >&2
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
            echo "WARNING: $SCRATCH does not exist; using \$HOME/spine_tutorial_work." >&2
            echo "         Ask on #comp-sdf for your scratch area and set WORKDIR." >&2
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
# spine-prod's own cache is used when you may write to it; the shared standard
# installation is usually read-only for users, so then a personal cache is used.
if [[ -z $SPINE_CACHE_DIR ]]; then
    PROD_CACHE=$SPINE_PROD_BASEDIR/.cache/weights
    if [[ -w $PROD_CACHE || ( ! -e $PROD_CACHE && -w $SPINE_PROD_BASEDIR ) ]]; then
        SPINE_CACHE_DIR=$PROD_CACHE
    else
        SPINE_CACHE_DIR=$WORKDIR/.cache/weights
    fi
fi
export SPINE_CACHE_DIR

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
    if [[ $SPINE_PROD_BASEDIR == "$SPINE_PROD_OFFICIAL" ]]; then
        echo "  SPINE_PROD_BASEDIR   = $SPINE_PROD_BASEDIR  (standard S3DF install, used for official productions)"
    else
        echo "  SPINE_PROD_BASEDIR   = $SPINE_PROD_BASEDIR  (personal copy: NOT for official productions)"
    fi
    echo "  SPINE_CONTAINER_PATH = ${SPINE_CONTAINER_PATH:-<not set>}"
    echo "  WORKDIR              = $WORKDIR"
    if [[ -n $SPINE_ACCOUNT ]]; then
        echo "  SPINE_ACCOUNT        = $SPINE_ACCOUNT"
    else
        echo '  SPINE_ACCOUNT        = <NOT SET: put it in MY_ACCOUNT="..." in setup_env.sh (Part 0, step 5.4)>'
    fi
fi
