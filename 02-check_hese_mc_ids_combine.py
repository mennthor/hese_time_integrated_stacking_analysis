# coding:utf-8

"""
Combine output jsons for each sample to a single file containing only the IDs.
"""

from __future__ import division, print_function

import sys
import os
import json
from glob import glob
import gzip
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("If you want fancy status bars: `pip install --user tqdm` ;)")
    tqdm = iter

from _paths import PATHS
from myi3scripts import arr2str

inpath = os.path.join(PATHS.data, "check_hese_mc_ids")
outpath = os.path.join(PATHS.local, "check_hese_mc_ids")
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

files = np.array(glob(os.path.join(inpath, "*.json")))
file_names = np.array(map(lambda s: os.path.basename(s), files))
dataset_nums = set(map(lambda s: s.split("_")[0], file_names))

# Combine all to a single JSON file
print("Reading files from directory:\n  {}".format(inpath))
print("  Found JSON files for datasets: {}".format(arr2str(dataset_nums)))
run_ids_per_sam = {}
event_ids_per_sam = {}
for num in dataset_nums:
    print("Combining IDs from set '{}':".format(num))
    run_ids = []
    event_ids = []
    _files = files[map(lambda s: s.startswith(num), file_names)]
    for fi in tqdm(_files):
        with open(fi, "r") as inf:
            di = json.load(inf)

        run_ids += di["run_id"]
        event_ids += di["event_id"]

    assert len(run_ids) == len(event_ids)
    print("  Total events filtered: {}".format(len(run_ids)))
    run_ids_per_sam[num] = run_ids
    event_ids_per_sam[num] = event_ids


# Combine to single dict for the seperate datasets
set2num = {
    "IC79": ["6308"],
    "IC86_2011": ["9366", "9095", "9701"],
    "IC86_2012-2015": ["11029", "11069", "11070"],
}

comment = {
    "IC79": ("IDs for MC for IC79 only. " +
             "Sets: {}").format(arr2str(set2num["IC79"])),
    "IC86_2011": ("IDs for MC for IC86I, 2011 only. " +
                  "Sets: {}").format(arr2str(set2num["IC86_2011"])),
    "IC86_2012-2015": ("IDs for MC for PS IC86 2012-2015 and GFU 2015. " +
                       "Sets: {}").format(arr2str(set2num["IC86_2012-2015"]))
}

for name, nums in set2num.items():
    print("Combining IDs for sample {}.\n  {}".format(name, comment[name]))
    out_dict = {"run_id": [], "event_id": [], "comment": comment[name]}
    for num in nums:
        out_dict["run_id"] += run_ids_per_sam[num]
        out_dict["event_id"] += event_ids_per_sam[num]
    # Save to compressed JSON
    _outp = os.path.join(outpath, "{}.json.gz".format(name))
    with gzip.open(_outp, "wb") as outf:
        json.dump(out_dict, fp=outf, indent=2, separators=(",", ": "))
        print("  Saved to file:\n    '{}'".format(_outp))

print("Done")
