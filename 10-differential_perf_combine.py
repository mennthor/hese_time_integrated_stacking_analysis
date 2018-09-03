# coding: utf-8

"""
Combine output all energy ranges to a single file containing all trials.
"""

import os
import re
import json
import gzip
import argparse
from glob import glob
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("If you want fancy status bars: `pip install --user tqdm` ;)")
    tqdm = iter

from _paths import PATHS, check_dir


parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store_true")
args = parser.parse_args()
dry_run = args.n


inpath = os.path.join(PATHS.data, "diff_perf_skylab_bins")

outpath = os.path.join(
    PATHS.data, "diff_perf_skylab_bins_combined")
check_dir(outpath)


file_path = os.path.join(inpath, "*.json.gz")
files = sorted(glob(file_path))

# Extract used mu points
pat = re.compile("lo\=(\d*\.\d*)\_")
lo_Es = np.unique(map(float, [pat.findall(fi)[0] for fi in files]))
pat = re.compile("hi\=(\d*\.\d*)")
hi_Es = np.unique(map(float, [pat.findall(fi)[0] for fi in files]))
if len(lo_Es) != len(hi_Es):
    if not np.array_equal(lo_Es[1:], hi_Es[:-1]):
        raise ValueError("Lower and upper bound mismatches in saced trials.")
print("Found {} energy bins:\n  {}".format(len(lo_Es), np.r_[lo_Es, hi_Es[-1]]))

# Build output dict: 3D ts -> rows are per E bin, cols per mu per E bin
trials = {
    "ts": [],       # 3D: (logE_bins, mus per bin, trials per mu per bin)
    "mus": [],      # 2D: used signal strengths per bin (usually all the same)
    "logE_lo": [],  # 1D: lower logE bin borders
    "logE_hi": [],  # 1D: upper logE bin borders
    "mu2flux": None,          # Assumed to be the same in all trials
    "mu2flux_per_src": None,  # Assumed to be the same in all trials
    "gamma_inj": None,        # Assumed to be the same in all trials
    "E0_inj": None,           # Assumed to be the same in all trials
    }

# Each table is sorted ascending in logEs and mus. TS vals get sorted too
for i, (lo_E, hi_E) in enumerate(zip(lo_Es, hi_Es)):
    # Create a new table for current bin
    trials["ts"].append([])

    trials["logE_lo"].append(lo_E)
    trials["logE_hi"].append(hi_E)

    files_per_E_bin = np.sort(glob(os.path.join(inpath,
        "mu=*_lo={:4.2f}_hi={:4.2f}.json.gz".format(lo_E, hi_E))))
    print(("- Found {} files for logE in " +
           "[{:4.2f}, {:4.2f}]").format(len(files_per_E_bin), lo_E, hi_E))

    if len(files_per_E_bin) > 0:
        print("  {}\n  ...\n  {}".format(
            files_per_E_bin[0], files_per_E_bin[-1]))
        # Extract used mus per bin, sort and store for the current logE bin
        pat = re.compile("mu\=(\d*\.\d*)\_")
        mus = map(float, [pat.findall(fi)[0] for fi in files_per_E_bin])
        idx = np.argsort(mus)
        mus = np.array(mus)[idx]
        trials["mus"].append(mus.tolist())

        # Concatenate all files, sorted accordingly to the mus
        for _file in tqdm(files_per_E_bin[idx]):
            with gzip.open(_file) as infile:
                trial_i = json.load(infile)
            # Append to 2nd dim
            trials["ts"][-1].append(sorted(trial_i["ts"]))
    else:
        print("  Skipping, no files...")

try:
    trials["mu2flux"] = trial_i["mu2flux"]
    trials["mu2flux_per_src"] = trial_i["mu2flux_per_src"]
    trials["gamma_inj"] = trial_i["gamma_inj"]
    trials["E0_inj"] = trial_i["E0_inj"]
except NameError:
    raise RuntimeError("No trial file loaded.")

# Save it
fpath = os.path.join(outpath, "diff_perf_combined.json.gz")
if not dry_run:
    with gzip.open(fpath, "w") as outf:
        json.dump(trials, fp=outf, indent=0, separators=(",", ":"))
        print("  - Saved to:\n    {}".format(fpath))
else:
    print("- Dry run, would store to:\n  '{}'".format(fpath))
