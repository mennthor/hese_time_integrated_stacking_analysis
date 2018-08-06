# coding: utf-8

"""
Use scanned files to build a JSON file with the needed source information. Here
we discard the sample information because we assume simultanious emission from
all sources.
"""

import os
import json
from glob import glob
import gzip

from _paths import PATHS


src_path = os.path.join(PATHS.local, "hese_scan_maps_truncated")

outpath = os.path.join(PATHS.local, "source_list")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

# Load sources up to HESE 6yr, list from:
#   https://wiki.icecube.wisc.edu/index.php/Analysis_of_pre-public_alert_HESE/EHE_events#HESE
# Last Run ID is 127853 from late 86V (2015) run, next from 7yr is 128290
src_files = sorted(glob(os.path.join(src_path, "*.json.gz")))

sources = []
for src_file in src_files:
    with gzip.open(src_file) as _f:
        src_dict = json.load(_f)
        # Build a compact version with all relevant infos
        src_i = {}
        for key in ["run_id", "event_id", "mjd"]:
            src_i[key] = src_dict[key]
        # Store best fit from direct local trafo and map maximum
        src_i["ra"] = src_dict["bf_equ"]["ra"]
        src_i["dec"] = src_dict["bf_equ"]["dec"]
        src_i["ra_map"] = src_dict["bf_equ_pix"]["ra"]
        src_i["dec_map"] = src_dict["bf_equ_pix"]["dec"]
        # Also store the path to the original file which contains the skymap
        src_i["map_path"] = src_file
        sources.append(src_i)
        print("Loaded HESE source from run {}:\n  {}".format(
            src_i["run_id"], src_file))

print("Number of considered sources: {}".format(len(src_files)))

fname = os.path.join(outpath, "source_list.json")
with open(fname, "w") as fp:
    json.dump(sources, fp=fp, indent=2)
    print("Saved source list to\n  '{}'".format(fname))
