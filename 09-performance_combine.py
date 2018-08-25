# coding: utf-8

"""
Combine the output for each grid point in the performance scans.
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
    inpath = os.path.join(PATHS.data, "performance_skylab_bins")
    outpath = os.path.join(PATHS.data, "performance_combined_skylab_bins")
else:
    inpath = os.path.join(PATHS.data, "performance")
    outpath = os.path.join(PATHS.data, "performance_combined")

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
file_path = os.path.join(inpath, "performance_mu=*.json.gz")
files = sorted(glob(file_path))
print("Found {} trial files:".format(len(files)))
if len(files) > 0:
    # Build output dict
    trials = {
        "mus": [],  # 1D array of scanned grid points in mean signal strength

        "ns": [],     # 2D array of fitted ns in ntrials per mu
        "ts": [],     # 2D array of fitted ts in ntrials per mu
        "gamma": [],  # 2D array of fitted gamma in ntrials per mu

        "ninj": [],       # 2D array of nr of injected evts in ntrials per mu
        "E0_inj": [],     # 1D array of injected flux E0 per mu
        "gamma_inj": [],  # 1D array of injected flux gamma per mu

        "rnd_seed": [],           # 1D array of used random seed per mu
        "ntrials_per_batch": [],  # 1D array of ntrials done per mu
        }
    # Concatenate all files
    for _file in tqdm(files):
        with gzip.open(_file) as infile:
            trial_i = json.load(infile)
        trials["mus"].append(trial_i["mu"])

        trials["ns"].append(trial_i["ns"])
        trials["ts"].append(trial_i["ts"])
        trials["gamma"].append(trial_i["gamma"])

        trials["ninj"].append(trial_i["ninj"])
        trials["E0_inj"].append(trial_i["E0_inj"])
        trials["gamma_inj"].append(trial_i["gamma_inj"])

        trials["rnd_seed"].append(trial_i["rnd_seed"])
        trials["ntrials_per_batch"].append(trial_i["ntrials"])
    # Save it
    fpath = os.path.join(outpath, "performance.json.gz")
    if dry_mode:
        print("  - Dry mode: Would save to\n    '{}'".format(fpath))
    else:
        with gzip.open(fpath, "w") as outf:
            json.dump(trials, fp=outf, indent=0, separators=(",", ":"))
            print("  - Saved to:\n    '{}'".format(fpath))
else:
    print("  - no trials found")
