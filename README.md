# SPINE on S3DF — a beginner's tutorial

This tutorial teaches you to run **SPINE**, the machine-learning reconstruction for
liquid-argon TPCs, on SLAC's **S3DF** computing cluster. It assumes you have never
used S3DF, SPINE or a Linux terminal. Every step comes in two flavours: one for
**JupyterLab** in a web browser and one for the **terminal**. Pick whichever you
prefer; you can switch at any time.

| Part | You will learn to… | Folder | Needs a GPU? |
|---|---|---|---|
| 0 | log in, find your way around, start Jupyter, get a compute node | [`00_s3df_basics/`](00_s3df_basics/README.md) | no |
| 1 | **process files through SPINE** and make *full* and *lite* output files | [`01_processing/`](01_processing/README.md) | yes (CPU works for the tiny generic sample) |
| 2 | **open LArCV input files** and **analyze SPINE output files** | [`02_analysis/`](02_analysis/README.md) | no |
| 3 | **train a SPINE model** and turn it into weights | [`03_training/`](03_training/README.md) | yes |

Do Part 0 first. After that, Parts 1 → 2 → 3 build on each other, but Part 2 also
works on its own, because it can download a ready-made output file.

Keep [`CHEATSHEET.md`](CHEATSHEET.md) open in another tab. It has every command on one page.

---

## What is in this folder

```
spine_s3df_tutorial/
├── README.md                  ← you are here
├── CHEATSHEET.md              ← one-page summary of every command
├── 00_s3df_basics/
│   ├── README.md              ← accounts, logging in, storage, Jupyter, compute nodes
│   ├── 00_check_environment.ipynb
│   └── scripts/
│       ├── setup_env.sh       ← sets all the variables the tutorial uses (source it!)
│       ├── compute_shell.sh   ← "give me a GPU node for an hour"
│       └── spine_container.sh ← "run this command inside the SPINE container"
├── 01_processing/   README.md + 01_run_spine.ipynb
├── 02_analysis/     README.md + 02a_open_larcv_files.ipynb + 02b_analyze_spine_output.ipynb
├── 03_training/     README.md + 03_toy_training.ipynb
└── configs/                   ← small SPINE config files used by the tutorial
```

## Versions this tutorial was written for

| Piece | Version | Where |
|---|---|---|
| **spine-prod** (production scripts and configs) | commit `cc682a6` (Oct 2026, "Migrate … to SPINE 1.4.0") | https://github.com/DeepLearnPhysics/spine-prod |
| **SPINE** (the reconstruction itself) | `1.4.0` | https://github.com/DeepLearnPhysics/spine |
| **Container** (all software pre-installed) | `spine:1.4.0` | `/sdf/data/neutrino/images/spine_v1-4-0.sif` on S3DF, or `ghcr.io/deeplearnphysics/spine:1.4.0` |
| Detectors used in examples | `generic` (toy detector) and `protodune-sp` | |

> **Why pin versions?** spine-prod and SPINE change quickly: configs, flags and
> Python functions get renamed. If you use a newer version and a command fails,
> check the spine-prod `README.md`, `QUICKREF.md` and `MIGRATION_*.md` files for what changed.
>
> **Older tutorials** (SPINE workshops before 2026, `lartpc_mlreco3d`, `run.sh`
> from spine-prod v0.3) use **old syntax**, for example `bin/run.py`, `parser:`
> keys and `spine.utils.globals`. Don't mix them with this one.

## The two ways of working

**Jupyter path (easiest to start):** open https://s3df.slac.stanford.edu/ondemand
in your browser, start a Jupyter session that runs *inside the SPINE container*,
then open the `.ipynb` notebooks. Each notebook explains itself cell by cell.
→ See [00_s3df_basics, "Path A"](00_s3df_basics/README.md#path-a--jupyterlab-in-your-browser-open-ondemand).

**Terminal path (needed for big jobs):** `ssh` to S3DF, borrow a compute node, run
commands. Every notebook starts with a **"Terminal equivalent"** box listing the
same commands, and each section's `README.md` is written for the terminal.
→ See [00_s3df_basics, "Path B"](00_s3df_basics/README.md#path-b--the-terminal).

## Glossary (come back here whenever a word is unfamiliar)

| Word | Meaning |
|---|---|
| **S3DF** | SLAC Shared Scientific Data Facility: the computing cluster we run on. |
| **Login node** (`s3dflogin`) | The machine you first `ssh` into. Only for hopping to other machines. Never run anything heavy here. |
| **Interactive node** (`ssh neutrino`) | A shared machine for editing files, `git`, and submitting jobs. **No GPUs.** |
| **Compute node** | A machine with GPUs or many CPUs that Slurm lends you for a limited time. |
| **Slurm** | The *scheduler*: you ask it for resources and it runs your work when they are free. |
| **Job** | One request to Slurm: either an *interactive* shell (`srun`) or a *batch* script (`sbatch`). |
| **Partition** | A group of identical compute nodes. `ampere` = A100 GPUs, `turing` = RTX 2080 Ti GPUs, `milano`/`roma` = CPUs only. |
| **Account** | Who pays for your job, written `facility:repo`, e.g. `neutrino:ml-dev`. You must be a member. |
| **Container** (`.sif` file) | A frozen, self-contained Linux environment with Python, PyTorch, ROOT, LArCV and SPINE pre-installed. Run with `apptainer`. |
| **Bind / mount** | Making a folder (e.g. `/sdf`) visible inside the container. |
| **LArCV** (`.root`) | The *input* format: 3D images (voxels) of the detector, plus simulation truth. Made upstream by the experiment's simulation. |
| **Voxel** | A 3D pixel. A LArTPC event is a cloud of voxels with an energy (charge) value each. |
| **HDF5** (`.h5`) | The *output* format SPINE writes: reconstructed and true particles and interactions. |
| **Full output** | `.h5` that also stores every voxel (`points`, `depositions`, labels). Big; lets you draw events and redo matching. |
| **Lite output** | `.h5` with only the particle and interaction summaries (energy, PID, direction, vertex…). ~5–20× smaller; no voxels. |
| **Full chain** | The complete SPINE reconstruction: voxels → semantic labels → clusters → particles → interactions. |
| **Config** (`.yaml`) | Text file telling SPINE what to read, which model to run, what to write. Configs can `include` other configs. |
| **Modifier** | A small config layered on top of another, e.g. `lite` (write lite output) or `data` (real data, no truth). |
| **Weights / checkpoint** (`.ckpt`) | The learned numbers of a trained network. Training produces checkpoints; the chosen one becomes "the weights". |
| **Iteration / epoch** | One iteration = one batch of events through training. One epoch = one pass over the whole training set. |
| **`$WORKDIR`** | Your big-file working area for this tutorial (set by `setup_env.sh`). |

## When something goes wrong

1. Read the **last 20 lines** of the error. The real message is usually at the bottom.
2. Check the "⚠️ Common mistake" boxes in that section.
3. Run the environment check: [`00_s3df_basics/00_check_environment.ipynb`](00_s3df_basics/00_check_environment.ipynb).
4. Ask: S3DF problems go to `#comp-sdf` on SLAC Slack or s3df-help@slac.stanford.edu. SPINE problems go to the DeepLearnPhysics channels or the GitHub issues.

## First-run checklist for maintainers

This tutorial was written and partly tested off-site. These items can only be
confirmed on S3DF itself. Tick them once, then update the text if anything differs:

- [ ] `/sdf/data/neutrino/images/spine_v1-4-0.sif` exists (otherwise follow "If the container is missing" in Part 0)
- [ ] the OnDemand "Custom Apptainer Image" session starts and the notebook kernel imports `spine`, `torch`, `larcv`
- [ ] `/sdf/scratch/users/<u>/<user>` is the right scratch path for new users
- [ ] Part 1 notebook runs generic and ProtoDUNE-SP inference on an `ampere` GPU
- [ ] Part 2a reads `generic_small.root` and `protodune-sp_small.root`
- [ ] Part 3 toy training writes `snapshot-49/99/149/199.ckpt` and `validation_log-*.csv`
