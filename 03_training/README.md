# Part 3 — Training SPINE and creating weights

**Goal:** understand how SPINE's networks are trained. You will run a **small training
yourself**: the first network of the chain, for 200 iterations. You'll watch the loss go down,
evaluate the saved checkpoints, and see how production weights are made and published.

| | Toy run (this tutorial) | Real production |
|---|---|---|
| network | UResNet + PPN only (stage 1 of the chain) | all stages, one after another |
| data | 3,200 generic events | ~100k–1M events per detector |
| length | 200 iterations, well under an hour on one A100 | days per stage, often 4 GPUs |
| result | checkpoints to learn from | a dated `.ckpt` used by `infer/<detector>/full_chain_<date>.yaml` |

**Jupyter users:** open [`03_toy_training.ipynb`](03_toy_training.ipynb) in a GPU session (`ampere`, 1 GPU).
Sections 4–6 below cover the batch and production side, which needs an `ssh` terminal.

---

## 1. Concepts in five minutes

- **Training** shows the network simulated events whose right answers (labels) are known,
  measures how wrong it is (the **loss**), and nudges its numbers (the **weights**) to be less wrong.
- One **iteration** processes one **batch** of events (e.g. 16). One **epoch** is one pass over the whole training set.
- Every so often the weights are saved as a **checkpoint**: `snapshot-<iteration>.ckpt`.
- **Validation** runs the current network on events it never trained on. If training loss keeps
  falling but validation loss rises, the network is memorising (**overfitting**). The checkpoint
  with the lowest validation loss is kept as `snapshot-best.ckpt`.
- The chosen checkpoint becomes **"the weights"** that inference configs point to.

## 2. SPINE is trained in stages

The full chain is several networks in a row. Each one is trained on its own, starting
from the true labels (or from the outputs of the already-trained stages before it).
Afterwards the networks are glued into one full-chain checkpoint:

```
 voxels ─▶ [deghost UResNet]* ─▶ [UResNet + PPN] ─▶ [Graph-SPICE] ─▶ [GrapPA shower] ─▶ [GrapPA interaction] ─▶ particles &
            remove ghost pts     semantic class      dense voxel      [GrapPA track ]    group particles,         interactions
            (*wire detectors     per voxel + key     clustering →     fragments →        primary/PID
             e.g. ProtoDUNE-SP)  points (start/end)  fragments        particles
```

The official recipes live in spine-prod:
```
config/train/<detector>/<stage>/train_<date>.yaml     ← one recipe per stage
pipelines/<detector>/full_chain_<date>.yaml           ← all stages, in order, as one workflow
```
At commit `cc682a6`, training recipes exist for **generic**, **nd-lar** and **protodune-sp**.

## 3. Anatomy of a training config

Look at the official generic UResNet+PPN recipe with everything expanded:
```bash
spine_container.sh spine-config dump \
    $SPINE_PROD_BASEDIR/config/train/generic/uresnet_ppn/train_240805.yaml | less
```
The important blocks:
```yaml
base:
  world_size: 1          # number of GPUs
  epochs: 50.0           # how long to train (or `iterations: N`)
io:
  loader:
    batch_size: 128      # events per iteration (split across GPUs)
    num_workers: 16      # CPU processes reading data
    sampler: {provider: random_sequence}    # shuffle the training events
    dataset:
      provider: larcv
      file_keys:         # the training files (given on the command line)
      schema: {data: ..., seg_label: ..., ppn_label: ...}   # same idea as Part 2a!
model:
  provider: uresnet_ppn
  weight_path:           # start from scratch (or from an existing checkpoint)
train:
  weight_prefix:         # where checkpoints go (given on the command line)
  save_epoch: 1.0        # checkpoint once per epoch (or `save_step: N` iterations)
  optimizer: {provider: Adam, lr: 0.001}
validation:
  file_keys:             # validation files (given on the command line)
  early_stopping: {monitor: loss, mode: min, patience: 5}
  best_checkpoint: true  # keep snapshot-best.ckpt
```
The tutorial's [`toy_train_uresnet_ppn.yaml`](../configs/toy_train_uresnet_ppn.yaml) **includes**
this recipe and changes only four numbers: 200 iterations, checkpoint every 50, batch 16, 4 workers.

## 4. Toy training, interactively

```bash
compute_shell.sh ampere 01:00:00          # 1 GPU for an hour
RUN=$WORKDIR/experiments/uresnet_ppn/toy  # everything from this training goes here

spine_container.sh spine -c $SPINE_TUTORIAL/configs/toy_train_uresnet_ppn.yaml \
    -s $GENERIC_TRAIN -n 3200 \
    --val-source $GENERIC_TEST --val-num-entries 64 \
    --weight-prefix $RUN/weights/snapshot \
    --log-dir $RUN
```
| Piece | Meaning |
|---|---|
| `-s $GENERIC_TRAIN -n 3200` | training file, first 3,200 events only |
| `--val-source … --val-num-entries 64` | validate on 64 events of the test file at every checkpoint |
| `--weight-prefix $RUN/weights/snapshot` | checkpoints become `$RUN/weights/snapshot-<N>.ckpt` |
| `--log-dir $RUN` | loss/accuracy logs (CSV) go here |

The same run **without** the tutorial config, using only command-line flags:
```bash
spine_container.sh spine -c train/generic/uresnet_ppn/train_240805.yaml \
    -s $GENERIC_TRAIN -n 3200 --val-source $GENERIC_TEST --val-num-entries 64 \
    --iterations 200 --batch-size 16 --num-workers 4 \
    --set train.save_epoch=null --set train.save_step=50 \
    --weight-prefix $RUN/weights/snapshot --log-dir $RUN
```
(`--set train.save_epoch=null` is needed because a recipe may use `save_epoch` *or* `save_step`, not both.)

**What you should see:** a line per iteration with the loss and accuracy, a validation summary at
iterations 50, 100, 150 and 200, and finally:
```
$RUN/
├── train_log-0000000.csv              ← one row per iteration
├── validation_log-0000050.csv  …-0000200.csv
└── weights/
    ├── snapshot-49.ckpt   snapshot-99.ckpt   snapshot-149.ckpt   snapshot-199.ckpt
    ├── snapshot-*.ckpt.sha256         ← checksums
    └── snapshot-best.ckpt             ← lowest validation loss so far
```
Checkpoint numbers count from 0, so the one saved after 50 iterations is `snapshot-49`.
Plot the curves with the notebook, or with `TrainDrawer` (see the notebook, section 3).

> ⚠️ `Cannot specify both save_step and save_epoch`: add `--set train.save_epoch=null` (or use the toy config).
> ⚠️ `Must provide a weight prefix`: you forgot `--weight-prefix`.
> ⚠️ `CUDA out of memory`: lower `--batch-size` (e.g. 8).

## 5. Evaluate checkpoints

Run every checkpoint over held-out events, without training:
```bash
spine_container.sh spine -c $SPINE_TUTORIAL/configs/uresnet_ppn_validation.yaml \
    -s $GENERIC_TEST -n 256 \
    --weight-path "$RUN/weights/snapshot-[0-9]*.ckpt" \
    --log-dir $RUN/posthoc
```
- **Quote the pattern**, so SPINE (not the shell) expands it. `[0-9]*` skips `snapshot-best`.
- SPINE writes one `inference_log-<N>.csv` per checkpoint.
- Alternative: `spine -c train/generic/uresnet_ppn/train_240805.yaml --inference …` converts any training recipe into an evaluation on the fly.

## 6. Training as a batch job (how real trainings run)

Training takes hours to days, so it always runs as a batch job. spine-prod manages a
**run directory** that keeps checkpoints, logs and job records together across restarts.
Official trainings, like official inference productions, use the standard S3DF spine-prod at
`/sdf/data/neutrino/software/spine-prod`, which is where `$SPINE_PROD_BASEDIR` points after `setup_env.sh`.
From the neutrino node (`ssh neutrino`):
```bash
cd $WORKDIR
$SPINE_PROD_BASEDIR/submit.py \
    --config train/generic/uresnet_ppn/train_240805.yaml \
    --stage train --run-dir $WORKDIR/experiments/uresnet_ppn/toy_batch \
    --source $GENERIC_TRAIN --num-entries 3200 \
    --val-source $GENERIC_TEST --val-num-entries 64 \
    --iterations 200 --batch-size 16 --num-workers 4 \
    --set train.save_epoch=null --set train.save_step=50 \
    -A $SPINE_ACCOUNT --time 01:00:00 --dry-run
```
Check `toy_batch/latest/submit.sbatch`, then run again without `--dry-run`.

| Task | Command |
|---|---|
| follow it | `squeue -u $USER`, `tail -f $WORKDIR/experiments/uresnet_ppn/toy_batch/stdout.log` |
| it hit the time limit: continue from the last checkpoint | same command + `--resume` |
| go back to a specific checkpoint | `--resume-from <run-dir>/weights/snapshot-99.ckpt` |
| ask it to stop cleanly (saves a checkpoint, exits successfully) | `$SPINE_PROD_BASEDIR/submit.py --graceful-stop <jobid>` |
| validate new checkpoints of a run later | `submit.py --config $SPINE_TUTORIAL/configs/uresnet_ppn_validation.yaml --stage validation --run-dir <run-dir> -A …` |
| live training curves | add `--tensorboard`, then run `tensorboard --logdir <run-dir>/tensorboard` inside the container |
| several GPUs | `--profile s3df_ampere_full` (4 × A100) or `--gpus 4` |

> ⚠️ A new training refuses to start in a non-empty `--run-dir`: use a new name, or `--resume`.
> ⚠️ `-n` in `submit.py` means `--ntasks`. Use the long `--num-entries` for events.

## 7. From checkpoints to "the weights"

**Swap one stage into the full chain.** Inference can load the published full-chain
checkpoint and replace one module with your own:
```bash
spine_container.sh spine -c infer/generic/full_chain_240805.yaml \
    -s $TUTORIAL_DATA/generic_small.root \
    --module-weight uresnet_ppn=$RUN/weights/snapshot-best.ckpt \
    --output-dir $WORKDIR/out --output-suffix my_uresnet
```
(After only 200 iterations your UResNet is much worse than the official one, so expect worse
reconstruction. That's the point of comparing!)

**Glue all trained stages into one checkpoint.** This is what spine-prod's pipelines do in their
`export_full_chain_weights` stage. It needs no GPU:
```bash
spine_container.sh spine -c model/protodune-sp/full_chain/model_260906.yaml --world-size 0 \
    --module-weight uresnet_deghost=<…>/uresnet_deghost/weights/snapshot-best.ckpt \
    --module-weight uresnet_ppn=<…>/uresnet_ppn/weights/snapshot-best.ckpt \
    --module-weight graph_spice=<…>/graph_spice/weights/snapshot-best.ckpt \
    --module-weight grappa_shower=<…>/grappa_shower/weights/snapshot-best.ckpt \
    --module-weight grappa_track=<…>/grappa_track/weights/snapshot-best.ckpt \
    --module-weight grappa_inter=<…>/grappa_inter/weights/snapshot-best.ckpt \
    --export-weights $WORKDIR/weights/full_chain_mine.ckpt
```

**Publish it.** Production inference configs point at a dated checkpoint. For example
`config/infer/protodune-sp/model/model_260906.yaml` contains:
```yaml
include: model/protodune-sp/full_chain/model_260906.yaml
override:
  model.weight_path: !download
    url: https://s3df.slac.stanford.edu/data/neutrino/spine/weights/protodune-sp/protodune-sp_snapshot_260906.ckpt
    hash: 754b45de…        # sha256 of the file, so everyone gets the identical weights
```
For your own tests a plain path also works: `model.weight_path: /sdf/…/full_chain_mine.ckpt`, or
on the command line, `--weight-path /sdf/…/full_chain_mine.ckpt`. A new dated
`full_chain_<date>.yaml` that includes your model file is how new weights become an
official configuration.

## 8. The real thing: the ProtoDUNE-SP staged pipeline

`pipelines/protodune-sp/full_chain_260906.yaml` trains all six networks on
`/sdf/data/neutrino/pdune/sim/sp/mpvmpr_v1/{train,test}_file_list.txt`. It has 21 stages:
event filtering, then for each network "train → cache its outputs for the next stage",
then export, evaluate and report. Each training stage uses a full 4-GPU ampere node for up
to 5 days. **Don't launch it just to try**, but do read it and dry-run it:
```bash
cd $WORKDIR
curl -o $WORKDIR/protodune-sp_snapshot_260906.ckpt \
    https://s3df.slac.stanford.edu/data/neutrino/spine/weights/protodune-sp/protodune-sp_snapshot_260906.ckpt
$SPINE_PROD_BASEDIR/submit.py \
    --pipeline $SPINE_PROD_BASEDIR/pipelines/protodune-sp/full_chain_260906.yaml \
    --workspace $WORKDIR/pdsp_pipeline \
    --warm-start $WORKDIR/protodune-sp_snapshot_260906.ckpt \
    -A $SPINE_ACCOUNT --dry-run
```
`--warm-start` initialises every stage from the published weights. This is **fine-tuning**,
the usual way to adapt the chain to a new simulation. Pass the pipeline file by its full path.

> ⚠️ At commit `cc682a6`, dry-running this pipeline **without** `--warm-start` stops at the first
> training stage with `warm_start_path and warm_start_modules must be provided together`.
> That's an upstream quirk, not your mistake.

## 9. Common mistakes

| Symptom | Cause → fix |
|---|---|
| `Cannot specify both save_step and save_epoch` | `--set train.save_epoch=null` when you pass `save_step` |
| `On-the-fly validation requires …` / validation `file_keys` empty | give `--val-source`/`--val-source-list`, or drop validation with `--set validation=null` |
| `Must provide a weight prefix` | `--weight-prefix <dir>/snapshot` (spine-prod sets it for you with `--run-dir`) |
| `CUDA out of memory` | smaller `--batch-size` |
| `GPUs requested exceeds visible devices` | not on a GPU node: `compute_shell.sh` |
| `run lifecycle options are currently supported in batch mode only` | `submit.py -I` can't be combined with `--stage/--run-dir/--resume`. Use plain `spine` interactively |
| validation curve looks averaged/odd in `TrainDrawer` | post-hoc `inference_log` files sit in the same folder as `validation_log`. Keep them in a separate `--log-dir` (e.g. `posthoc/`) |
| weight pattern matched nothing | quote it: `--weight-path "$RUN/weights/snapshot-[0-9]*.ckpt"` |

**Congratulations, you've completed the tutorial!** Keep [`../CHEATSHEET.md`](../CHEATSHEET.md) handy.
