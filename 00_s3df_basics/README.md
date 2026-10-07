# Part 0 — S3DF basics

By the end of this part you will be able to:
- log in to S3DF;
- move around in a terminal;
- know where to keep files;
- finish the one-time setup for this tutorial;
- open JupyterLab or a terminal on a GPU node, inside the SPINE container.

Allow about 1 hour, most of it waiting for an account if you don't have one yet.

---

## 1. Get access (one time)

1. **SLAC account.** You need a SLAC computer account, which requires completing the SLAC
   cyber-security training (CS100). Your group contact or supervisor can start this for you.
2. **Neutrino facility.** Log in to https://coact.slac.stanford.edu → **Repos** →
   **Request Access to Facility** → pick **Neutrino**. A facility czar approves it.
3. **Slurm account ("repo").** Every compute job is charged to an *account*
   written `facility:repo`. Ask your group which one to use. Common ML ones are:
   - `neutrino:ml-dev`: general SPINE development (also spine-prod's default)
   - `neutrino:dune-ml`: DUNE detectors (2x2, ND-LAr, ProtoDUNE…)
   - `neutrino:icarus-ml`: ICARUS and SBND

   Once you can log in (next step), list the accounts you belong to with:
   ```bash
   sacctmgr show assoc user=$USER format=Account%40,Partition%20
   ```

> ⚠️ **Accounts expire** after ~60 days without logging in. Reactivate at
> https://identity.slac.stanford.edu/reactivate.

## 2. Log in

On your laptop, open a terminal: **Terminal** on macOS, **PowerShell** or WSL on Windows, any terminal on Linux. Then type
(replace `jdoe` with *your* SLAC username):

```bash
ssh jdoe@s3dflogin.slac.stanford.edu
```

Type your SLAC password (nothing appears while you type; that's normal), then
approve the Duo push. You are now on a **login node**. It is only a doorway:
hop straight to an **interactive node**:

```bash
ssh iana
```

Your prompt changes to something like `[jdoe@sdfiana012 ~]$`. **iana** is where you
edit files, use git, and submit jobs. It has **no GPUs**, so don't run SPINE here.

> 💡 **Shortcut.** On your laptop, add this to `~/.ssh/config`, then `ssh iana` takes you straight there:
> ```
> Host s3dflogin
>     HostName s3dflogin.slac.stanford.edu
>     User jdoe
> Host iana
>     HostName iana
>     User jdoe
>     ProxyJump s3dflogin
> ```
> Windows "Corrupted MAC on input" error? Add `-m hmac-sha2-512` to the ssh command.

## 3. Terminal survival kit

| Command | What it does | Example |
|---|---|---|
| `pwd` | **p**rint **w**orking **d**irectory: where am I? | `pwd` |
| `ls -lh` | list files, with human-readable sizes | `ls -lh $WORKDIR` |
| `cd DIR` | change directory (`cd` alone = go home, `cd ..` = go up) | `cd ~/spine-prod` |
| `mkdir -p DIR` | make a directory (and parents) | `mkdir -p $WORKDIR/out` |
| `cp A B`, `mv A B`, `rm A` | copy, move/rename, delete (**no undo!**) | `cp a.yaml b.yaml` |
| `less FILE` | read a file page by page (`q` quits, `/word` searches) | `less README.md` |
| `nano FILE` | simple text editor (`Ctrl-O` save, `Ctrl-X` quit) | `nano setup_env.sh` |
| `echo $VAR` | print a variable | `echo $WORKDIR` |
| `df -h DIR` | how full is this disk area? | `df -h ~` |
| `du -sh DIR` | how big is this folder? | `du -sh $WORKDIR` |
| `Tab` | auto-complete file names (press it constantly) | |
| `↑` / `Ctrl-R` | previous commands / search command history | |
| `Ctrl-C` | stop the running command | |

**`$SOMETHING`** is an *environment variable*: a named value the shell remembers,
like `$HOME` (your home folder) or `$USER` (your username). This tutorial sets its own
variables (`$WORKDIR`, `$SPINE_ACCOUNT`, …) in `setup_env.sh`.

**tmux** keeps your terminal alive when your laptop sleeps or WiFi drops:
- `tmux new -s spine` starts a session;
- `Ctrl-B` then `D` detaches from it;
- `tmux attach -t spine` reconnects later.

Note the iana node name (e.g. `sdfiana012`). To reattach you must `ssh` to that same node.

## 4. Where to keep files

| Place | Size | Backed up? | Use it for |
|---|---|---|---|
| `$HOME` (`/sdf/home/j/jdoe`) | **30 GB** | yes | code: this tutorial, spine-prod, your scripts |
| scratch (`/sdf/scratch/users/j/jdoe`) | 100 GB | **no**, old files may be purged | `$WORKDIR`: tutorial data, outputs, trained weights |
| `/sdf/data/neutrino/...` | large, shared | depends | long-term datasets and productions (ask your group where) |
| `/lscratch` | node-local | wiped after your job | temporary files during a job |

> ⚠️ **Common mistake: filling `$HOME`.** SPINE output and weight files are big.
> Always work inside `$WORKDIR`. If `df -h ~` shows you near 30 GB, move things out.

## 5. One-time setup

Do this **on iana**, in this order.

**5.1 Put this tutorial on S3DF.** For example, copy it from your laptop. Run this on the laptop, from the folder that contains `spine_s3df_tutorial`:
```bash
rsync -av spine_s3df_tutorial jdoe@s3dflogin.slac.stanford.edu:~/
```
If it lives in a git repository, `git clone` it into `~` instead.

**5.2 Get spine-prod at the version this tutorial uses:**
```bash
cd ~
git clone https://github.com/DeepLearnPhysics/spine-prod.git
cd spine-prod
git checkout cc682a6230f6bf416e8f0f336964a549208cc32f
```
(`git checkout <hash>` freezes the code at that exact version. Later, `git checkout main && git pull` gives you the newest one.)

**5.3 Check the host Python can run spine-prod's `submit.py`:**
```bash
python3 --version                      # needs 3.9 or newer
python3 -c "import jinja2, yaml" && echo OK
```
If that prints an error instead of `OK`: `python3 -m pip install --user jinja2 pyyaml`.

> ⚠️ **Do not `pip install spine` on iana.** If `which spine` prints a path on the host,
> `submit.py` will make your batch jobs run *that* copy instead of the container's.
> Keep any personal SPINE environment deactivated when submitting.

**5.4 Tell the tutorial your account.** Open the setup script and fill in `SPINE_ACCOUNT`:
```bash
nano ~/spine_s3df_tutorial/00_s3df_basics/scripts/setup_env.sh
#   export SPINE_ACCOUNT=${SPINE_ACCOUNT:-neutrino:ml-dev}     <- your account here
```
Then load it, and make it load automatically at every login:
```bash
source ~/spine_s3df_tutorial/00_s3df_basics/scripts/setup_env.sh
echo 'source ~/spine_s3df_tutorial/00_s3df_basics/scripts/setup_env.sh' >> ~/.bashrc
```
**What you should see:** a short summary listing `SPINE_TUTORIAL`, `SPINE_PROD_BASEDIR`, the container path, `WORKDIR` and your account.

> ⚠️ `configure.sh` and `setup_env.sh` need **bash**. S3DF's default shell is bash.
> If `echo $0` says `zsh` or `tcsh`, type `bash` first.

**5.5 Check the SPINE container exists:**
```bash
ls -lh $SPINE_CONTAINER_PATH          # /sdf/data/neutrino/images/spine_v1-4-0.sif
```
<details><summary><b>If the container is missing</b> (click to expand)</summary>

Build your own copy from the official image. This takes ~10–20 minutes and is best done on a CPU compute node:
```bash
compute_shell.sh milano 01:00:00
export APPTAINER_CACHEDIR=/lscratch/$USER/apptainer APPTAINER_TMPDIR=/lscratch/$USER/apptainer
mkdir -p $APPTAINER_CACHEDIR $WORKDIR/images
apptainer pull $WORKDIR/images/spine_v1-4-0.sif docker://ghcr.io/deeplearnphysics/spine:1.4.0
exit
```
Then, in `setup_env.sh`, uncomment the `export SPINE_CONTAINER_PATH=...` line in section 1.
It must hold the full path of the file you just pulled (`echo $WORKDIR/images/spine_v1-4-0.sif`
prints it). Open a fresh terminal and source `setup_env.sh` again.
(The cache variables stop Apptainer from filling your 30 GB home.)
</details>

**5.6 Download the trained weights once.** They are about 115 MB and 155 MB, and every job re-uses them:
```bash
spine_container.sh python3 $SPINE_PROD_BASEDIR/scripts/preload_downloads.py \
    infer/generic/full_chain_240805.yaml infer/protodune-sp/full_chain_260906.yaml
```

**5.7 Download the small example input files** into `$WORKDIR/data`:
```bash
cd $TUTORIAL_DATA
curl -O $SPINE_SAMPLE_URL/larcv/generic_small.root        # 0.8 MB, toy detector
curl -O $SPINE_SAMPLE_URL/larcv/protodune-sp_small.root   # 150 MB, ProtoDUNE-SP
curl -O $SPINE_SAMPLE_URL/reco/generic_small_spine.h5     # 1.6 MB, a ready-made SPINE output
ls -lh
```

---

## Path A — JupyterLab in your browser (Open OnDemand)

1. Go to **https://s3df.slac.stanford.edu/ondemand** and log in.
2. Top menu: **Interactive Apps → Jupyter**.
3. Fill in the form:

| Field | What to put |
|---|---|
| **Jupyter Image** | `Custom` |
| **Jupyter Image Version** | `Apptainer Image…` |
| **Commands to initiate Jupyter** | paste the block below |
| **Use JupyterLab instead of Jupyter Notebook?** | ✅ ticked |
| **Run on cluster type** | `Batch` |
| **Account** | your account, e.g. `neutrino:ml-dev` |
| **Partition** | `ampere` for Parts 1 and 3 (GPU) · `milano` for Part 2 (CPU) |
| **Number of hours** | `2` |
| **Number of CPU cores** | `8` |
| **Total Memory to allocate** | `65536` (that is 64 GB, in MB) |
| **Number of GPUs** | `1` on ampere · `0` on milano |

Paste this into **Commands to initiate Jupyter**:
```bash
export SPINE_TUTORIAL_QUIET=1
source $HOME/spine_s3df_tutorial/00_s3df_basics/scripts/setup_env.sh
export APPTAINER_IMAGE_PATH=$SPINE_CONTAINER_PATH
export PYTHONNOUSERSITE=1
function jupyter() { apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${APPTAINER_IMAGE_PATH} jupyter $@; }
```
(In plain words: load the tutorial settings, and when OnDemand starts Jupyter, start it
*inside the SPINE container* with the GPU and `/sdf` visible.)

4. Click **Launch**. The card says *Queued*, then *Running*. That can take a few minutes on a busy day.
5. Click **Connect to Jupyter**. In the file browser on the left, go to `spine_s3df_tutorial/` and open
   [`00_check_environment.ipynb`](00_check_environment.ipynb).
6. Run a cell with **Shift-Enter**. A `[*]` next to a cell means it is still running.

> ⚠️ **Session won't start / closes immediately.** In OnDemand, **My Interactive Sessions** → click the
> session ID → open `output.log`. Usually: wrong account name, no GPUs free on that partition
> (try `turing`), or the container path doesn't exist (step 5.5).
>
> ⚠️ **The Jupyter terminal is inside the container.** It's fine for `spine …` commands, but
> `sbatch`/`squeue` don't exist there. Submit batch jobs from an `ssh` terminal on iana.
>
> ⚠️ **The session ends when its hours run out**, and anything still running is killed. Save often.

## Path B — the terminal

From iana (inside `tmux`, ideally):

```bash
compute_shell.sh                 # ask Slurm for 1 GPU on 'ampere' for 1 hour
```
**What you should see:** `srun: job 12345678 queued and waiting for resources`, then a new
prompt on a GPU node, e.g. `[jdoe@sdfampere007 ~]$`. Check the GPU:
```bash
nvidia-smi                       # lists one A100 GPU
spine_container.sh spine --version
spine_container.sh spine --info  # which pieces of SPINE are available
spine_container.sh               # no arguments: opens a shell INSIDE the container
```
`exit` leaves the container shell; `exit` again gives the GPU node back.

Other sizes: `compute_shell.sh ampere 02:00:00` (2 hours), `compute_shell.sh turing`
(cheaper GPU), `compute_shell.sh milano` (CPU only, for analysis).

> ⚠️ **Lost connection = lost session.** An `srun` shell dies if your SSH connection drops.
> Use `tmux` on iana, or for anything longer than ~1 hour, submit a batch job (Part 1, step 4).
>
> ⚠️ **Don't leave idle compute shells open.** They block a GPU someone else could use, and they count against your group's allocation.

## Check your environment

- **Jupyter:** run every cell of [`00_check_environment.ipynb`](00_check_environment.ipynb). It ends with a ✅/❌ table.
- **Terminal:** same check, from a compute node:
  ```bash
  spine_container.sh jupyter nbconvert --to notebook --execute --stdout \
      $SPINE_TUTORIAL/00_s3df_basics/00_check_environment.ipynb > /dev/null && echo "all good"
  ```

Next: **[Part 1 — processing files through SPINE](../01_processing/README.md)**.
