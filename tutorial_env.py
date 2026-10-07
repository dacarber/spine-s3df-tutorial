"""Small helpers shared by all the tutorial notebooks.

Every notebook starts with:

    import sys; sys.path.insert(0, "..")
    import tutorial_env as te
    te.setup()

`setup()` makes sure the same environment variables that
``00_s3df_basics/scripts/setup_env.sh`` defines are also defined inside
Jupyter, even if you forgot to source that script. Values that are already
set are never overwritten.
"""

import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

#: Folder that contains this file (the tutorial root)
TUTORIAL_DIR = Path(__file__).resolve().parent

#: True when running on S3DF (the /sdf file system only exists there)
ON_S3DF = Path("/sdf/data/neutrino").exists()

#: Public web area holding the small example files
SAMPLE_URL = "https://s3df.slac.stanford.edu/data/neutrino/spine/workshop"


def _default_workdir():
    if not ON_S3DF:
        return TUTORIAL_DIR / "work"
    user = os.environ.get("USER", "unknown")
    scratch = Path(f"/sdf/scratch/users/{user[0]}/{user}")
    if scratch.exists():
        return scratch / "spine_tutorial"
    return Path.home() / "spine_tutorial_work"


def setup(verbose=True):
    """Define the tutorial environment variables (if not already set)."""
    env = os.environ
    env.setdefault("SPINE_TUTORIAL", str(TUTORIAL_DIR))
    env.setdefault("SPINE_PROD_BASEDIR", str(Path.home() / "spine-prod"))
    env.setdefault("WORKDIR", str(_default_workdir()))
    env.setdefault("TUTORIAL_DATA", str(Path(env["WORKDIR"]) / "data"))
    env.setdefault("SPINE_SAMPLE_URL", SAMPLE_URL)

    # spine-prod's configs are found through SPINE_CONFIG_PATH
    prod_cfg = Path(env["SPINE_PROD_BASEDIR"]) / "config"
    if "SPINE_CONFIG_PATH" not in env and prod_cfg.exists():
        env["SPINE_CONFIG_PATH"] = str(prod_cfg)
    env.setdefault(
        "SPINE_CACHE_DIR", str(Path(env["SPINE_PROD_BASEDIR"]) / ".cache" / "weights")
    )
    env.setdefault("NUMBA_NUM_THREADS", str(min(os.cpu_count() or 4, 64)))

    if ON_S3DF:
        env.setdefault(
            "GENERIC_TRAIN", "/sdf/data/neutrino/generic/mpvmpr_2020_01_v04/train.root"
        )
        env.setdefault(
            "GENERIC_TEST", "/sdf/data/neutrino/generic/mpvmpr_2020_01_v04/test.root"
        )
        env.setdefault(
            "PDSP_TEST_LIST",
            "/sdf/data/neutrino/pdune/sim/sp/mpvmpr_v1/test_file_list.txt",
        )
    else:
        env.setdefault("GENERIC_TRAIN", str(Path(env["TUTORIAL_DATA"]) / "generic_small.root"))
        env.setdefault("GENERIC_TEST", str(Path(env["TUTORIAL_DATA"]) / "generic_small.root"))

    Path(env["WORKDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TUTORIAL_DATA"]).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"{'host':<18}: {socket.gethostname()}  (on S3DF: {ON_S3DF})")
        print(f"{'inside container':<18}: {in_container()}")
        for key in ("SPINE_TUTORIAL", "SPINE_PROD_BASEDIR", "WORKDIR", "TUTORIAL_DATA"):
            print(f"{key:<18}: {env[key]}")
    return env


def in_container():
    """True if this Python is running inside an Apptainer/Singularity image."""
    return bool(os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"))


def fetch(name, kind="larcv"):
    """Return the local path of an example file, downloading it if needed.

    Parameters
    ----------
    name : str
        File name, e.g. ``"generic_small.root"``
    kind : str
        ``"larcv"`` for input files, ``"reco"`` for SPINE output files
    """
    dest = Path(os.environ["TUTORIAL_DATA"]) / name
    if not dest.exists():
        url = f"{os.environ.get('SPINE_SAMPLE_URL', SAMPLE_URL)}/{kind}/{name}"
        print(f"Downloading {url}\n         -> {dest}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    print(f"{name}: {dest.stat().st_size / 1e6:.1f} MB at {dest}")
    return dest


def sh(command, check=True):
    """Run a shell command, printing it first and streaming its output.

    This is exactly what you would type in a terminal; the ``$ ...`` line
    shows the command with all variables already filled in.
    """
    expanded = os.path.expandvars(command)
    print(f"$ {expanded}\n", flush=True)
    proc = subprocess.Popen(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}. "
            "Read the last lines of the output above for the reason."
        )
    return proc.returncode


def has_gpu():
    """True if PyTorch can see a CUDA GPU."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def which(program):
    """Full path of a program on PATH, or None."""
    return shutil.which(program)
