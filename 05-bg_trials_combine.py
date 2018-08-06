# coding: utf-8

"""
Combine output for each time window to a single file containing all trials.
"""

import os
import sys
import json
import gzip
import argparse
from glob import glob

try:
    from tqdm import tqdm
except ImportError:
    print("If you want fancy status bars: `pip install --user tqdm` ;)")
    tqdm = iter

from _paths import PATHS


parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("-n", "--dry", action="store_true", default=False)
parser.add_argument("--use_skylab_bins", action="store_true", default=False)
args = parser.parse_args()
use_skylab_bins = args.use_skylab_bins
dry_mode = args.dry

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    inpath = os.path.join(PATHS.data, "bg_trials_skylab_bins")
    outpath = os.path.join(PATHS.data, "bg_trials_combined_skylab_bins")
else:
    inpath = os.path.join(PATHS.data, "bg_trials")
    outpath = os.path.join(PATHS.data, "bg_trials_combined")

if os.path.isdir(outpath):
    res = raw_input("'{}' already exists. ".format(outpath) +
                    "\nAllow overwrites (y/n)? ")
    if not res.lower() in ("y", "yes"):
        print("Abort. Script has done nothing.")
        sys.exit()
    print("  Using output directory '{}'.".format(outpath))
else:
    os.makedirs(outpath)
    print("Created output directory '{}'.".format(outpath))

# Collect for all time windows
file_path = os.path.join(inpath, "job_*.json.gz")
files = sorted(glob(file_path))
print("Found {} trial files:".format(len(files)))
if len(files) > 0:
    print("  {}\n  ...\n  {}".format(files[0], files[-1]))
    # Build output dict
    trials = {
        "ns": [],
        "ts": [],
        "spectrum": None,
        "rnd_seed": [],
        "ntrials": 0,
        "ntrials_per_batch": [],
        }
    # Concatenate all files
    for _file in tqdm(files):
        with gzip.open(_file) as infile:
            trial_i = json.load(infile)
        trials["ns"] += trial_i["ns"]
        trials["ts"] += trial_i["ts"]
        trials["ntrials"] += trial_i["ntrials"]
        trials["rnd_seed"].append(trial_i["rnd_seed"])
        trials["ntrials_per_batch"].append(trial_i["ntrials"])
    trials["spectrum"] = trial_i["spectrum"]
    # Save it
    fpath = os.path.join(outpath, "bg_trials.json.gz")
    if dry_mode:
        print("  - Dry mode: Would save to\n    '{}'".format(fpath))
    else:
        with gzip.open(fpath, "w") as outf:
            json.dump(trials, fp=outf, indent=0, separators=(",", ":"))
            print("  - Saved to:\n    '{}'".format(fpath))
else:
    print("  - no trials found")
