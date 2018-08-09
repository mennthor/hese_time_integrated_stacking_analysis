# coding: utf-8

"""
Makes single BG PDF object from the combined BG trials. PDFs are composed from
an empirical part with good statistics and a fitted exponential tail to get
continious p-values at the tails.
"""

import os
import json
import gzip
import argparse
import numpy as np

import tdepps.utils.stats as stats
from _paths import PATHS


parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("-n", "--dry", action="store_true", default=False)
parser.add_argument("--use_skylab_bins", action="store_true", default=False)
args = parser.parse_args()
use_skylab_bins = args.use_skylab_bins
dry_mode = args.dry

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    inpath = os.path.join(PATHS.data, "bg_trials_combined_skylab_bins")
    outpath = os.path.join(PATHS.local, "bg_pdf_skylab_bins")
    plotpath = os.path.join(PATHS.plots, "bg_pdf_skylab_bins")
else:
    inpath = os.path.join(PATHS.data, "bg_trials_combined")
    outpath = os.path.join(PATHS.local, "bg_pdf")
    plotpath = os.path.join(PATHS.plots, "bg_pdf")

if not os.path.isdir(outpath):
    os.makedirs(outpath)
if not os.path.isdir(plotpath):
    os.makedirs(plotpath)

fpath = os.path.join(inpath, "bg_trials.json.gz")
with gzip.open(fpath) as fp:
    trials = json.load(fp)
    print("- Loaded combined trials from:\n    {}".format(fpath))
ts = np.array(trials["ts"])
ts[ts <= 0] = 0.

# Create PDF object and scan the best threshold
print("- Scanning best threshold")
emp_dist = stats.ExpTailEmpiricalDist(ts, nzeros=0, thresh=np.amax(ts))
# Scan in a range with still good statistics, but leave the really good
# statistics part to the empirical PDF
lo, hi = emp_dist.ppf(q=100. * stats.sigma2prob([2., 4.]))
thresh_vals = np.arange(lo, hi, 0.1)
# Best fit: KS test p-value is larger than `pval_thresh` the first time
pval_thresh = 0.5
best_thresh, best_idx, pvals, scales = stats.scan_best_thresh(
    emp_dist=emp_dist, thresh_vals=thresh_vals, pval_thresh=pval_thresh)
print("  Best threshold is at TS = {:.2f}, {:.2f} sigma".format(
    best_thresh, stats.prob2sigma(stats.cdf_nzeros(x=ts, vals=best_thresh,
                                                   nzeros=0, sorted=False))[0]))

# Save whole PDF object to recoverable JSON file
pdf_name = os.path.join(outpath, "bg_pdf.json.gz")
if dry_mode:
    print("- Dry mode. Would save PDF as JSON to:\n    {}".format(pdf_name))
else:
    with gzip.open(pdf_name, "w") as fp:
        fp.write(emp_dist.to_json(dtype=np.float, indent=0,
                                  separators=(",", ":")))
        print("- Saved PDF as JSON to:\n    {}".format(pdf_name))
