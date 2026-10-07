# Part 2 — Analyzing SPINE outputs and opening LArCV inputs

Two kinds of files, two notebooks:

| Notebook | File type | What you learn | Needs |
|---|---|---|---|
| [`02a_open_larcv_files.ipynb`](02a_open_larcv_files.ipynb) | **LArCV** input (`.root`) | what's inside, load voxels and truth, 3D plots, ghost points | SPINE container (ROOT + LArCV). CPU is fine |
| [`02b_analyze_spine_output.ipynb`](02b_analyze_spine_output.ipynb) | **SPINE** output (`.h5`), full and lite | objects, event displays, tables, histograms, matching, efficiency/purity, CSV export | CPU only. Even works on a laptop with `pip install "spine[viz]"` |

**Jupyter users:** start a session on **`milano`** (CPU, 0 GPUs) and open the notebooks in order.
No GPU is needed: nothing here runs a neural network.

---

## Which tool for which job?

| I want to… | Use | Where shown |
|---|---|---|
| see which products a LArCV file contains | `LArCVDataset.list_data(path)` | 02a §1 |
| get voxels and true labels/particles from a LArCV file as numpy | `LArCVDataset(file_keys=…, schema={…})[i]` | 02a §2 |
| debug a LArCV file at the ROOT level | `ROOT.TChain("<product>_tree")` + `larcv.fill_3d_voxels` | 02a §5 |
| read a LArCV file without the container | `uproot` | 02a §6 |
| quickly list what's inside a SPINE `.h5` | `h5py.File(path).keys()` | 02b §1 |
| get particles and interactions as objects, with reco↔truth matches | `Driver` + [`read_spine_output.yaml`](../configs/read_spine_output.yaml) | 02b §2 |
| same, from a **lite** file | [`read_spine_output_lite.yaml`](../configs/read_spine_output_lite.yaml) (`build.lite: true`) | 02b §7 |
| draw an event | `spine.vis.Drawer(data).get("particles", …)` | 02b §4 |
| build a pandas table over many events | `obj.scalar_dict(attrs=[…])` | 02b §5 |
| measure selection efficiency and purity | `interaction_matches_t2r` / `_r2t` + [`selection.py`](selection.py) | 02b §6 |
| hand results to ROOT/Excel/anything | CSV via [`ana_save_csv.yaml`](../configs/ana_save_csv.yaml) | 02b §8 |

## Terminal recipes

All commands run inside the container (`spine_container.sh …`). CPU is fine, so you can
even run them on the neutrino node for small files, or on a `compute_shell.sh milano` node.

**What's inside a LArCV file?**
```bash
spine_container.sh python -c "
from spine.io.dataset import LArCVDataset
for kind, names in LArCVDataset.list_data('$TUTORIAL_DATA/protodune-sp_small.root').items():
    print(kind, names)"
```

**What's inside a SPINE file?**
```bash
spine_container.sh python -c "import h5py; f = h5py.File('$WORKDIR/out/generic_small_full.h5'); print(list(f), len(f['events']), 'events')"
```

**Print one event's particles and interactions** (a small script in this folder):
```bash
spine_container.sh python $SPINE_TUTORIAL/02_analysis/print_event.py $WORKDIR/out/generic_small_full.h5 --entry 2
spine_container.sh python $SPINE_TUTORIAL/02_analysis/print_event.py $WORKDIR/out/generic_small_lite.h5 --lite
```
What you should see (excerpt, for the example generic file):
```
RECO PARTICLES
   id inter shape   PID       primary   KE[MeV]  len[cm]  match
    ...
    9     3 Track   Proton    True         58.8      3.1  [0, 1]
   10     3 Track   Muon      True        175.4     64.6  [1]
...
RECO INTERACTIONS
    3  topology 1mu1p        vertex (-249.0, -32.7, -40.5) cm  fiducial=True  contained=True  match [3]
```

**Export to CSV:**
```bash
spine_container.sh spine -c $SPINE_TUTORIAL/configs/ana_save_csv.yaml \
    -s $WORKDIR/out/generic_small_full.h5 --log-dir $WORKDIR/ana
ls $WORKDIR/ana    # save_reco_particles.csv  save_reco_interactions.csv  save_truth_...csv
```
For lite files, add `--set build.lite=true`.

**Export a whole production to CSV as a batch job** (CPU nodes; from the neutrino node):
```bash
cd $WORKDIR
$SPINE_PROD_BASEDIR/submit.py \
    --config $SPINE_TUTORIAL/configs/ana_save_csv.yaml \
    --source-list runs/<your Part-1 run>/latest/outputs.txt \
    --profile s3df_milano -A $SPINE_ACCOUNT
```
`--profile s3df_milano` is important: for config files outside spine-prod, `submit.py`
can't tell the detector, so it would otherwise ask for a GPU node.

**Run a notebook from the terminal** and keep the results as a web page:
```bash
cd $SPINE_TUTORIAL/02_analysis
spine_container.sh jupyter nbconvert --to html --execute 02b_analyze_spine_output.ipynb
# -> 02b_analyze_spine_output.html (copy it to your laptop with scp to view)
```

## Your exercise: a selection

[`02b`](02b_analyze_spine_output.ipynb) section 6 asks you to write `select_interaction(inter)` in
[`selection.py`](selection.py), for example "contained νμ CC-like: one primary muon and at least one
primary proton, vertex in the fiducial volume". The notebook applies it to both truth and reco
and reports **efficiency** and **purity** using SPINE's reco↔truth matching.

## Common mistakes

| Symptom | Cause → fix |
|---|---|
| `KeyError: 'data_tensor'` when reading an `.h5` | it's a **lite** file read with the full config. Use `read_spine_output_lite.yaml` / `--lite` / `--set build.lite=true` |
| `ModuleNotFoundError: No module named 'ROOT'` (or `larcv`) | you're not inside the container. Use `spine_container.sh`, or the uproot section of 02a |
| `FileExistsError: … spine_log.csv already exists` | an old run's log is there. The tutorial configs set `overwrite_log: true`; for other configs add `--set base.overwrite_log=true` |
| plotly figure is blank | `import plotly.io as pio; pio.renderers.default = "iframe"` and re-run |
| `ke = -1`, `length = -1` | that quantity isn't defined for this object (e.g. `length` of a shower), or the post-processor that computes it didn't run |
| huge KE like `100000` | an exiting track whose energy estimate failed. Use `is_contained` to select reliable ones |
| numbers in `pid`/`shape` | translate with `from spine.constants import PID_LABELS, SHAPE_LABELS` |

Next: **[Part 3 — training and creating weights](../03_training/README.md)**.
