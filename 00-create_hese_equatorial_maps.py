# coding: utf8

"""
Make complete healpy logLLH maps from HESE scans and save in npy or json as
simple arrays.

Find original HESE rescan files at:
    /data/ana/IC79/starting-event/IC170922A_scans_all_historrical_events

This script processes all track maps, converts them to an equatorial map,
applies 1° smoothing in normal space and normalizes the map to have integral 1
over the whole sphere.
"""

from __future__ import print_function, division
import os
import json
import gzip
from glob import glob
import datetime
from concurrent.futures import ProcessPoolExecutor, wait
import numpy as np

from _paths import PATHS
from myi3scripts.hese import make_healpy_map_from_HESE_scan
from tdepps.utils import get_pixel_in_sigma_region


print("Start: {}\n".format(datetime.datetime.utcnow()))

inpath = os.path.join("/data", "ana", "IC79", "starting-event",
                      "IC170922A_scans_all_historrical_events",
                      "??????.i3.bz2_event0000")
folders = sorted(glob(inpath))
print("Found {} scan folders:".format(len(folders)))
for folder in folders:
    print("  {}".format(folder))

outpath = os.path.join(PATHS.data, "hese_scan_maps")
if not os.path.isdir(outpath):
    os.makedirs(outpath)
outfiles = map(lambda name: os.path.join(outpath, name),
               map(os.path.basename, folders))

icemodel = "SpiceMie"
scan_str = "step0[0-9]_{}_nside????.i3.bz2".format(icemodel)

# Work on multiple maps at once. If module not available, loop over all files
N_JOBS = len(folders)
with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
    processes = []
    for infolder, outfile in zip(folders, outfiles):
        processes.append(
            executor.submit(
                make_healpy_map_from_HESE_scan,
                infolder=infolder,
                scan_file_str=scan_str,
                outfile=outfile,
                coord="equ",
                outfmt="json",
                smooth_sigma=1.,
                )
            )
    results = wait(processes)

print("\nDone: {}".format(datetime.datetime.utcnow()))

# Now load all PDF maps again and remove artifacts from healpy smoothing
map_files = sorted(glob(os.path.join(outpath, "*.json.gz")))

# New outpath indicating the truncation
outpath = os.path.join(PATHS.local, "hese_scan_maps_truncated")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

for map_f in map_files:
    print("Removing artifacts from map:\n  {}".format(map_f))
    with gzip.open(map_f) as f:
        src = json.load(f)

    pdf_map = np.array(src["map"])

    # Set all entries to zero outside the 6 sigma region. This value is
    # empirical and resembles the size of smoothing artifacts after visual
    # inspection of the original maps
    _, _, in_region_pix = get_pixel_in_sigma_region(pdf_map, sigma=6.)
    truncated_map = np.zeros_like(pdf_map)
    truncated_map[in_region_pix] = pdf_map[in_region_pix]

    # Save map dict again
    src["map"] = truncated_map.tolist()
    fname = os.path.join(outpath, os.path.basename(map_f))
    print("- Done, saving truncated map to:\n  {}".format(fname))
    with gzip.open(fname, "w") as f:
        json.dump(src, fp=f, indent=2, separators=(",", ":"))
