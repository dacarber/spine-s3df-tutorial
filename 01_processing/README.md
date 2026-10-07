# Part 1 — Processing files through SPINE (full and lite outputs)

**Goal:** take LArCV input files (`.root`), run the SPINE *full chain* on them, and get
SPINE output files (`.h5`) in two flavours, **full** and **lite**. We do this three ways,
from the simplest to the most powerful:

| Way | Where it runs | Good for |
|---|---|---|
| A. `spine` directly | your compute shell / Jupyter | learning, quick tests on 1 file |
| B. `submit.py -I` (interactive) | your compute shell | testing a production setup exactly as batch will run it |
| C. `submit.py` (batch) | Slurm, in the background | real productions: many files, hours of GPU time |

**Jupyter users:** open [`01_run_spine.ipynb`](01_run_spine.ipynb) (GPU session on `ampere`).
It runs way A on both detectors and compares full vs lite. Come back here for ways B and C.

Time: ~45 minutes. Needs: Part 0 finished, `setup_env.sh` sourced.

---

## 1. The big picture

```
 LArCV file (.root)            SPINE full chain (neural networks + post-processing)          SPINE file (.h5)
 ─────────────────            ─────────────────────────────────────────────────────          ───────────────
 3D voxels + charge   ──────▶  segment voxels → cluster → build particles → group into  ──────▶  particles &
 (+ simulation truth)          interactions → compute energy, PID, direction, vertex…            interactions
```

**Full vs lite output.** Both contain the reconstructed (and, for simulation, true)
**particles** and **interactions** with all their properties. They differ in voxels:

| | **Full** | **Lite** |
|---|---|---|
| particle and interaction properties (PID, KE, start/end point, vertex, matches…) | ✅ | ✅ |
| every voxel (`points`, `depositions`) and labels | ✅ | ❌ |
| can draw event displays with hits | ✅ | ❌ (only summary markers) |
| can re-run post-processing that needs voxels | ✅ | ❌ |
| size (generic example) | 1.6 MB | 0.29 MB (~5× smaller; often 10–20× for real detectors) |
| use it for | development, debugging, event displays | large productions and physics analyses |

The switch is one setting, `io.writer.lite: true`, plus removing the heavy products from
the list of things to write. spine-prod packages that as the **`lite` modifier**.

## 2. Configs: what SPINE should do

A **config** is a YAML text file. spine-prod keeps the official ones in
`$SPINE_PROD_BASEDIR/config/infer/<detector>/`:

```
config/infer/generic/
├── full_chain_240805.yaml        ← a complete, dated configuration ("bundle")
├── base/  io/  model/  post/     ← the pieces it is built from
└── modifier/
    ├── lite/mod_lite_240718.yaml ← "write lite output"
    └── data/mod_data_240718.yaml ← "this is real data: no truth"
```

`full_chain_240805.yaml` contains only an `include:` list of its pieces. The **date**
(YYMMDD) identifies the trained weights, so a dated name always gives the same result.
spine-prod also understands **`infer/<detector>/latest`**, meaning "newest of each piece".

| Detector | `latest` currently equals | Weights |
|---|---|---|
| generic | `infer/generic/full_chain_240805.yaml` | `generic_snapshot_240805.ckpt` (115 MB) |
| ProtoDUNE-SP | `infer/protodune-sp/full_chain_260906.yaml` | `protodune-sp_snapshot_260906.ckpt` (154 MB) |

See a config with all its includes expanded (`q` to quit):
```bash
spine_container.sh spine-config dump $SPINE_PROD_BASEDIR/config/infer/generic/full_chain_240805.yaml | less
```
(`spine-config dump` needs the full path. `spine -c` and `submit.py --config` also accept
the short `infer/...` form, because they search `$SPINE_CONFIG_PATH`.)
List the modifiers that can be applied to a config:
```bash
$SPINE_PROD_BASEDIR/submit.py --list-mods infer/protodune-sp/full_chain_260906.yaml
```
This tutorial's own lite configs are in [`../configs/`](../configs/). Each one is just
"production bundle + lite modifier", for example
[`generic_full_chain_lite.yaml`](../configs/generic_full_chain_lite.yaml).

---

## 3. Way A — run `spine` directly on one file

Get a GPU and go to your work area:
```bash
compute_shell.sh                      # wait for the prompt to show a GPU node
cd $WORKDIR
```

**Generic, full output:**
```bash
spine_container.sh spine \
    -c infer/generic/full_chain_240805.yaml \
    -s $TUTORIAL_DATA/generic_small.root \
    --output-dir $WORKDIR/out --output-suffix full \
    --log-dir $WORKDIR/logs
```
What each part means:

| Piece | Meaning |
|---|---|
| `spine_container.sh` | run the next command inside the SPINE container |
| `spine` | the SPINE program |
| `-c …` | **c**onfig to use (looked up in spine-prod's `config/` folder) |
| `-s …` | **s**ource = input file(s) |
| `--output-dir`, `--output-suffix` | write `<input name>_<suffix>.h5` into that folder |
| `--log-dir` | where the per-batch log CSV goes |

**What you should see:**
- A banner showing the resolved configuration.
- `Will load 1 file(s)` and the number of entries.
- On the very first run only, a weight download.
- A progress table with timing, and finally `$WORKDIR/out/generic_small_full.h5`.

**Generic, lite output:** same command, different config and suffix:
```bash
spine_container.sh spine \
    -c $SPINE_TUTORIAL/configs/generic_full_chain_lite.yaml \
    -s $TUTORIAL_DATA/generic_small.root \
    --output-dir $WORKDIR/out --output-suffix lite \
    --log-dir $WORKDIR/logs
```

**ProtoDUNE-SP**, the first 10 events only (`-n 10`), full then lite:
```bash
spine_container.sh spine -c infer/protodune-sp/full_chain_260906.yaml \
    -s $TUTORIAL_DATA/protodune-sp_small.root -n 10 \
    --output-dir $WORKDIR/out --output-suffix full --log-dir $WORKDIR/logs

spine_container.sh spine -c $SPINE_TUTORIAL/configs/protodune-sp_full_chain_lite.yaml \
    -s $TUTORIAL_DATA/protodune-sp_small.root -n 10 \
    --output-dir $WORKDIR/out --output-suffix lite --log-dir $WORKDIR/logs
```

Compare the results:
```bash
ls -lh $WORKDIR/out
spine_container.sh python -c "
import h5py, sys
for f in sys.argv[1:]:
    print(f.split('/')[-1], sorted(h5py.File(f).keys()))
" $WORKDIR/out/*.h5
```
The full files list `points`, `depositions`, … and the lite ones don't.

> ⚠️ **Always give an output location.** Without `-o`/`--output-dir`, SPINE writes next to the
> input file. For shared data you can't write there, so you get `Permission denied`.
>
> 💡 **No GPU?** For the tiny generic file only, add `--world-size 0` to run on CPU (slow).
> Otherwise you get `The number of GPUs requested (1) exceeds the number of visible devices (0)`.

---

## 4. Way B — `submit.py -I`: the production recipe, in your shell

`submit.py` is spine-prod's front end. It assembles the config (resolving `latest` and
applying modifiers), creates a tidy **run directory**, and runs SPINE in the container.
`-I` (interactive) does all of that in your current shell instead of submitting a batch job.
This is a good last test before submitting hundreds of jobs.

```bash
cd $WORKDIR                                        # run directories are created HERE
$SPINE_PROD_BASEDIR/submit.py -I \
    --config infer/generic/latest \
    --source $TUTORIAL_DATA/generic_small.root \
    --bind-paths /sdf \
    --output-suffix spine

$SPINE_PROD_BASEDIR/submit.py -I \
    --config infer/generic/latest --apply-mods lite \
    --source $TUTORIAL_DATA/generic_small.root \
    --bind-paths /sdf \
    --output-suffix spine_lite
```
**What you should see:** a new folder `runs/<date>_<time>_interactive_generic_latest/`
containing the composed config (`generic_latest_240805_lite_composite.yaml`), `logs/`,
and `output/generic_small_spine_lite.h5`.

> ⚠️ **`--bind-paths /sdf` matters in `-I` mode.** Interactive runs only make the current
> folder and spine-prod visible inside the container. Without it, input files under `/sdf/...`
> are "not found". Batch jobs bind `/sdf` automatically.
>
> ⚠️ **`cd $WORKDIR` first.** `submit.py` creates `runs/` in whatever folder you are in. Run it
> from `$HOME` and your 30 GB fills up.

---

## 5. Way C — batch jobs with Slurm

Batch jobs run in the background on nodes Slurm picks, and they survive you logging out.
**Submit from the neutrino interactive node** (`ssh neutrino`) (not from a compute shell, and not from the Jupyter terminal).

**5.1 Dry run first.** `--dry-run` writes everything but doesn't submit:
```bash
cd $WORKDIR
$SPINE_PROD_BASEDIR/submit.py \
    --config infer/generic/latest \
    --source $TUTORIAL_DATA/generic_small.root \
    -A $SPINE_ACCOUNT --time 00:30:00 \
    --output-suffix spine \
    --dry-run
```
Open the generated job script and read the `#SBATCH` lines at the top:
```bash
less runs/*_spine_generic_latest/latest/submit.sbatch
```
```
#SBATCH --account=neutrino:ml-dev      ← who pays
#SBATCH --partition=ampere             ← which kind of node (A100 GPUs)
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8g
#SBATCH --time=00:30:00                ← killed if it runs longer than this
```
Further down, the script runs `singularity exec --bind /sdf/ --nv $SPINE_CONTAINER_PATH … spine -S … -c …`,
which is the same `spine` command you ran by hand.

**5.2 Submit for real:** run the same command without `--dry-run`. It prints a Slurm job ID.

**5.3 Watch it:**
```bash
squeue -u $USER                                   # PD = pending (waiting), R = running
tail -f runs/<your run>/stdout.log                # live log; Ctrl-C stops watching, not the job
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it finishes
seff <jobid>                                      # how efficiently you used what you asked for
scancel <jobid>                                   # cancel it
```

**5.4 Find the output.** Every run directory looks like this:
```
runs/20261007_104444_spine_generic_latest/
├── generic_latest_240805_composite.yaml   ← the exact config used
├── stdout.log, stderr.log                 ← links to the latest attempt's Slurm logs
├── latest → attempts/<timestamp>/
└── attempts/<timestamp>/
    ├── submit.sbatch                      ← the job script
    ├── inputs.txt                         ← input files
    ├── outputs.txt                        ← list of output files (great for Part 2!)
    └── output/generic_small_spine.h5
```

### Many files: file lists and job arrays

Real productions have hundreds of input files. Put their paths in a text file, one per line:
```bash
cat $PDSP_TEST_LIST | head -3              # an existing ProtoDUNE-SP list
ls -1 /path/to/my/files/*.root > my_files.txt   # make your own
```
Then split the work:
```bash
$SPINE_PROD_BASEDIR/submit.py \
    --config infer/protodune-sp/latest \
    --source-list $PDSP_TEST_LIST --num-files 8 \
    --files-per-task 2 \
    -A $SPINE_ACCOUNT --time 01:00:00 \
    --output-suffix spine --dry-run
```
This makes a **job array** of 4 tasks (`#SBATCH --array=1-4`), each processing 2 files.
Useful options:

| Option | Meaning |
|---|---|
| `--num-files 8` | only the first 8 files of the list (for testing) |
| `--files-per-task N` | N input files per array task |
| `--ntasks N` | with `--files-per-task`: at most N tasks run at once. Alone: split the files into N tasks |
| `--apply-mods lite` | write lite files directly |
| `--apply-mods data` | real data (no truth). Combine: `--apply-mods data lite` |
| `--profile s3df_turing --partition turing` | cheaper RTX 2080 Ti GPUs (see the warning below) |

All outputs of all tasks are listed in `runs/<run>/latest/outputs.txt`.

### Already have full files? Make lite ones later

```bash
$SPINE_PROD_BASEDIR/submit.py \
    --config infer/common/litify.yaml \
    --source-list runs/<your run>/latest/outputs.txt \
    -A $SPINE_ACCOUNT
```
This runs on CPU nodes (`milano`) because no neural network is needed. For a single
file, interactively: `spine_container.sh spine -c infer/common/litify.yaml -s full.h5 --output-dir $WORKDIR/out --output-suffix lite`.

---

## 6. Common mistakes

| Symptom | Cause → fix |
|---|---|
| `Invalid account or account/partition combination` | `-A` missing or wrong. **spine-prod defaults to `neutrino:ml-dev` for every detector** if you leave it out. Always pass `-A $SPINE_ACCOUNT`. |
| `#SBATCH --partition=` is empty, job rejected | the `s3df_turing` profile has no partition at this commit. Add `--partition turing`. |
| `No such file or directory` for an `/sdf/...` input in `-I` mode | add `--bind-paths /sdf` |
| `Permission denied` writing output | you didn't give `--output-dir`/`-o`, so SPINE tried to write beside a read-only input |
| `GPUs requested (1) exceeds … visible devices (0)` | you are not on a GPU node. Run `compute_shell.sh` first, or use `--world-size 0` for CPU |
| `sbatch: command not found` | you're in the Jupyter terminal (inside the container). Submit from the neutrino node (`ssh neutrino`). |
| home directory full | `runs/` created in `$HOME`. `cd $WORKDIR` before `submit.py`. |
| numba / OpenBLAS thread errors | `NUMBA_NUM_THREADS` too high. `setup_env.sh` caps it at 64. |
| `--set` value rejected by `submit.py` | no spaces or quotes allowed: `--set io.loader.batch_size=8` |
| `-n` means different things | `spine -n 10` = 10 events; `submit.py -n 10` = 10 tasks (`--ntasks`) |

Next: **[Part 2 — analyzing SPINE outputs and LArCV inputs](../02_analysis/README.md)**.
