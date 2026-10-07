#!/usr/bin/env python3
"""Print the particles and interactions of one event in a SPINE output file.

Terminal companion of 02b_analyze_spine_output.ipynb.

Usage (inside the container, CPU is fine):
    spine_container.sh python print_event.py FILE.h5            # entry 0
    spine_container.sh python print_event.py FILE.h5 --entry 3
    spine_container.sh python print_event.py FILE_lite.h5 --lite
"""

import argparse
import os
import sys

from spine.config import load_config_file
from spine.constants import PID_LABELS, SHAPE_LABELS
from spine.driver import Driver

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "..", "configs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="SPINE output file (.h5)")
    parser.add_argument("--entry", type=int, default=0, help="entry number (default 0)")
    parser.add_argument("--lite", action="store_true", help="the file is a LITE file")
    args = parser.parse_args()

    cfg_name = "read_spine_output_lite.yaml" if args.lite else "read_spine_output.yaml"
    cfg = load_config_file(os.path.join(CONFIG_DIR, cfg_name))
    cfg["io"]["reader"]["file_keys"] = args.file
    cfg["base"]["log_dir"] = os.environ.get("WORKDIR", ".") + "/logs"
    cfg["base"]["verbosity"] = "warning"      # hide SPINE's start-up banner

    driver = Driver(cfg)
    data = driver.process(entry=args.entry)

    run = data["run_info"]
    print(f"{args.file}  entry {args.entry}/{len(driver) - 1}  "
          f"(run {run.run}, subrun {run.subrun}, event {run.event})\n")

    for kind in ("reco", "truth"):
        key = f"{kind}_particles"
        if key not in data:
            continue
        print(f"{kind.upper()} PARTICLES")
        print(f"  {'id':>3} {'inter':>5} {'shape':<7} {'PID':<9} {'primary':<7} {'KE[MeV]':>9} {'len[cm]':>8}  match")
        for p in data[key]:
            print(f"  {p.id:>3} {p.interaction_id:>5} {SHAPE_LABELS[p.shape]:<7} {PID_LABELS[p.pid]:<9} "
                  f"{str(bool(p.is_primary)):<7} {p.ke:9.1f} {p.length:8.1f}  {p.match_ids.tolist()}")
        print()

    for kind in ("reco", "truth"):
        key = f"{kind}_interactions"
        if key not in data:
            continue
        print(f"{kind.upper()} INTERACTIONS")
        for inter in data[key]:
            vtx = ", ".join(f"{v:.1f}" for v in inter.vertex)
            print(f"  {inter.id:>3}  topology {inter.topology or '-':<12} "
                  f"vertex ({vtx}) cm  fiducial={bool(inter.is_fiducial)}  "
                  f"contained={bool(inter.is_contained)}  match {inter.match_ids.tolist()}")
        print()


if __name__ == "__main__":
    sys.exit(main())
