# SPINE on S3DF — cheat sheet

Assumes `source ~/spine-s3df-tutorial/00_s3df_basics/scripts/setup_env.sh` has been run.

**spine-prod on S3DF:** the standard installation `/sdf/data/neutrino/software/spine-prod`
(= `$SPINE_PROD_BASEDIR`) is used for **all official productions**. Personal clones are for development only.
```bash
echo $SPINE_PROD_BASEDIR                                         # which spine-prod am I using?
git -C /sdf/data/neutrino/software/spine-prod log --oneline -1   # which version is it?
```

## Getting around S3DF
```bash
ssh <user>@s3dflogin.slac.stanford.edu     # login node (doorway only)
ssh neutrino                                # interactive node: edit, git, submit jobs (no GPU)
tmux new -s spine  /  Ctrl-B D  /  tmux attach -t spine    # survive disconnects
df -h ~        du -sh $WORKDIR              # disk usage (home is only 30 GB)
sacctmgr show assoc user=$USER format=Account%40           # which Slurm accounts can I use?
```
Jupyter: https://s3df.slac.stanford.edu/ondemand → Interactive Apps → Jupyter → Custom / Apptainer Image
(the form values are listed in [`00_s3df_basics/README.md`](00_s3df_basics/README.md#path-a--jupyterlab-in-your-browser-open-ondemand)).

## Compute nodes and the container
```bash
compute_shell.sh                 # 1 A100 GPU, 1 h        compute_shell.sh ampere 02:00:00
compute_shell.sh turing          # cheaper GPU            compute_shell.sh milano   (CPU only)
nvidia-smi                       # which GPU do I have?
spine_container.sh <command>     # run <command> inside the SPINE container
spine_container.sh               # shell inside the container
exit                             # leave container / give the node back
```

## Slurm
```bash
squeue -u $USER                                   # my jobs (PD = waiting, R = running)
scancel <jobid>                                   # cancel
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
seff <jobid>                                      # efficiency summary after it ends
```
| Partition | Hardware | Use |
|---|---|---|
| `ampere` | 4× A100 40 GB per node | inference and training (spine-prod default) |
| `hopper` | 4× H200 | big training |
| `turing` | 10× RTX 2080 Ti | cheap inference (`--profile s3df_turing --partition turing`) |
| `milano`, `roma` | CPUs only | analysis, litify, CSV export |

## Running SPINE (Part 1)
```bash
cd $WORKDIR
# one file, directly
spine_container.sh spine -c infer/generic/full_chain_240805.yaml -s IN.root \
    --output-dir $WORKDIR/out --output-suffix full --log-dir $WORKDIR/logs
# lite output
spine_container.sh spine -c $SPINE_TUTORIAL/configs/generic_full_chain_lite.yaml -s IN.root \
    --output-dir $WORKDIR/out --output-suffix lite --log-dir $WORKDIR/logs
# first N events only:   -n 10        CPU instead of GPU:   --world-size 0
# see a config:          spine_container.sh spine-config dump $SPINE_PROD_BASEDIR/config/infer/generic/full_chain_240805.yaml
```
```bash
# spine-prod, interactive (production recipe, in your shell)
$SPINE_PROD_BASEDIR/submit.py -I --config infer/generic/latest [--apply-mods lite] \
    --source /abs/path/IN.root --bind-paths /sdf --output-suffix spine
# spine-prod, batch (from the neutrino node) -- ALWAYS pass -A
$SPINE_PROD_BASEDIR/submit.py --config infer/protodune-sp/latest [--apply-mods lite|data|data lite] \
    --source-list files.txt [--files-per-task 2] -A $SPINE_ACCOUNT --time 01:00:00 [--dry-run]
$SPINE_PROD_BASEDIR/submit.py --list-mods infer/protodune-sp/full_chain_260906.yaml
$SPINE_PROD_BASEDIR/submit.py --config infer/common/litify.yaml --source-list runs/<run>/latest/outputs.txt -A $SPINE_ACCOUNT
```
| Detector | `latest` = | Account (usual) |
|---|---|---|
| generic | `infer/generic/full_chain_240805.yaml` | `neutrino:ml-dev` |
| ProtoDUNE-SP | `infer/protodune-sp/full_chain_260906.yaml` | `neutrino:dune-ml` |

**Where things go** (batch): `runs/<date>_spine_<det>_<cfg>/`, with `stdout.log`, `stderr.log`,
and `latest/{submit.sbatch, inputs.txt, outputs.txt, output/}`.

## Analysis (Part 2)
```python
# LArCV input (container only)
from spine.io.dataset import LArCVDataset
LArCVDataset.list_data(path)
ds = LArCVDataset(file_keys=path, schema={"data": {"provider": "sparse3d", "sparse_event": "sparse3d_pcluster"},
                                          "meta": {"provider": "meta", "sparse_event": "sparse3d_pcluster"}}, dtype="float32")
ev = ds[0]; xyz = ev["meta"].to_cm(ev["data"].coords, center=True)

# SPINE output (anywhere)
import os
from spine.config import load_config_file; from spine.driver import Driver
cfg = load_config_file(os.path.expandvars("$SPINE_TUTORIAL/configs/read_spine_output.yaml"))   # *_lite.yaml for lite files
cfg["io"]["reader"]["file_keys"] = "out.h5"
data = Driver(cfg).process(entry=0)
data["reco_particles"], data["truth_interactions"], data["interaction_matches_t2r"]
from spine.vis import Drawer; Drawer(data).get("particles", attr=["shape"], color_attr="shape").show()
from spine.constants import PID_LABELS, SHAPE_LABELS
```
```bash
spine_container.sh python $SPINE_TUTORIAL/02_analysis/print_event.py out.h5 --entry 0 [--lite]
spine_container.sh spine -c $SPINE_TUTORIAL/configs/ana_save_csv.yaml -s out.h5 --log-dir $WORKDIR/ana   # → CSVs
```

## Training (Part 3)
```bash
RUN=$WORKDIR/experiments/uresnet_ppn/toy
spine_container.sh spine -c $SPINE_TUTORIAL/configs/toy_train_uresnet_ppn.yaml \
    -s $GENERIC_TRAIN -n 3200 --val-source $GENERIC_TEST --val-num-entries 64 \
    --weight-prefix $RUN/weights/snapshot --log-dir $RUN
spine_container.sh spine -c $SPINE_TUTORIAL/configs/uresnet_ppn_validation.yaml -s $GENERIC_TEST -n 256 \
    --weight-path "$RUN/weights/snapshot-[0-9]*.ckpt" --log-dir $RUN/posthoc
# batch, with resumable run directory
$SPINE_PROD_BASEDIR/submit.py --config train/generic/uresnet_ppn/train_240805.yaml --stage train \
    --run-dir $WORKDIR/experiments/uresnet_ppn/run1 --source-list train.txt --val-source-list val.txt \
    -A $SPINE_ACCOUNT [--resume] [--tensorboard]
$SPINE_PROD_BASEDIR/submit.py --graceful-stop <jobid>
# swap one module / export a full-chain checkpoint
spine -c infer/generic/full_chain_240805.yaml ... --module-weight uresnet_ppn=snapshot-best.ckpt
spine -c model/<det>/full_chain/model_<date>.yaml --world-size 0 --module-weight A=a.ckpt ... --export-weights full.ckpt
```

## The mistakes everyone makes
| Symptom | Fix |
|---|---|
| job charged to the wrong account / rejected | always `-A $SPINE_ACCOUNT` (spine-prod defaults to `neutrino:ml-dev`) |
| `#SBATCH --partition=` empty | `--profile s3df_turing` also needs `--partition turing` |
| input "not found" with `submit.py -I` | add `--bind-paths /sdf` |
| `Permission denied` writing output | give `--output-dir`/`-o` |
| home directory full | `cd $WORKDIR` before `submit.py`; keep data out of `$HOME` |
| `GPUs requested … exceeds … visible devices` | get a GPU (`compute_shell.sh`) or `--world-size 0` |
| `sbatch: command not found` | you're in the Jupyter terminal (container). Submit from the neutrino node (`ssh neutrino`) |
| `KeyError: 'data_tensor'` reading `.h5` | lite file: use the `_lite` reader config / `--set build.lite=true` |
| `Cannot specify both save_step and save_epoch` | `--set train.save_epoch=null` |
| `-n` confusion | `spine -n` = number of events; `submit.py -n` = number of tasks |
| `configure.sh` errors | it needs bash: type `bash` first |
| `which spine` shows a host path | deactivate that env before `submit.py`, or batch jobs will use it |
