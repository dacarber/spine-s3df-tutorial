#!/bin/bash
# ----------------------------------------------------------------------------
# compute_shell.sh -- open an interactive shell ON A COMPUTE NODE.
#
# The login/interactive nodes (s3dflogin, neutrino) have no GPUs and are shared by
# everyone. Real work (running SPINE, training) must happen on a compute node,
# which you borrow from Slurm for a limited time.
#
# Usage:
#     compute_shell.sh                  # 1 GPU on 'ampere' for 1 hour
#     compute_shell.sh ampere 02:00:00  # 1 GPU on 'ampere' for 2 hours
#     compute_shell.sh turing           # 1 cheaper GPU (RTX 2080 Ti)
#     compute_shell.sh milano           # CPU only (fine for analysis)
#
# When the prompt changes to something like  [you@sdfampere012 ~]$  you are
# on the compute node. Type `exit` to give the node back.
#
# WARNING: if your laptop sleeps or the SSH connection drops, this session
# dies. Run it inside `tmux` (see CHEATSHEET.md) to survive disconnects.
# ----------------------------------------------------------------------------
PARTITION=${1:-ampere}
TIME=${2:-01:00:00}

if [[ -z $SPINE_ACCOUNT ]]; then
    echo "ERROR: SPINE_ACCOUNT is not set."
    echo "       Put your account in setup_env.sh, e.g. MY_ACCOUNT=\"neutrino:ml-dev\""
    echo "       (Part 0 README, step 5.4), then: source ~/spine_s3df_tutorial/00_s3df_basics/scripts/setup_env.sh"
    exit 1
fi

case $PARTITION in
    milano|roma|torino) GPU_ARGS=() ;;            # CPU-only partitions
    *)                  GPU_ARGS=(--gpus 1) ;;    # GPU partitions
esac

echo "Requesting a $PARTITION node for $TIME on account $SPINE_ACCOUNT ..."
echo "(this can take a few seconds to a few minutes if the cluster is busy)"
exec srun --account "$SPINE_ACCOUNT" --partition "$PARTITION" \
     --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 64G "${GPU_ARGS[@]}" \
     --time "$TIME" --pty /bin/bash
