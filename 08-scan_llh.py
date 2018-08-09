# coding: utf-8

"""
Unblind held back ontime data.
"""

from __future__ import print_function, division
import gc  # Manual garbage collection
import os
import gzip
import json
import argparse
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("If you want fancy status bars: `pip install --user tqdm` ;)")
    tqdm = iter

from skylab.ps_llh import PointSourceLLH, MultiPointSourceLLH
from skylab.llh_models import EnergyLLH

from tdepps.utils.stats import ExpTailEmpiricalDist, prob2sigma

from _paths import PATHS, check_dir
import _loader


def dict_print(d):
    shift = max(map(len, d.keys())) + 1
    for key, val in d.items():
        print("{0:{1:d}s}: ".format(key, shift), val)


parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--really_unblind", action="store_true")
parser.add_argument("--use_skylab_bins", action="store_true", default=False)
args = parser.parse_args()
really_unblind = args.really_unblind
use_skylab_bins = args.use_skylab_bins

if not really_unblind:
    print("## Not really unblinding, events get scrambled!   ##")
    scramble = True
else:
    print("## In real unblinding mode, it's getting serious! ##")
    scramble = False

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    outpath = os.path.join(PATHS.local, "llh_scan_skylab_bins")
    pdf_path = os.path.join(PATHS.local, "bg_pdf_skylab_bins", "bg_pdf.json.gz")
else:
    outpath = os.path.join(PATHS.local, "llh_scan")
    pdf_path = os.path.join(PATHS.local, "bg_pdf", "bg_pdf.json.gz")

check_dir(outpath, ask=False)

# Extract source info, use the same fixed spectral index -2 for all
sources = _loader.source_list_loader()
nsrcs = len(sources)
ra = [src["ra"] for src in sources]
dec = [src["dec"] for src in sources]
# Theo weights need to be normalized manually in skylab?
w = np.ones(nsrcs, dtype=float) / float(nsrcs)

# Create the multi LLH
rnd_seed = 1
multillh = MultiPointSourceLLH(seed=rnd_seed, ncpu=40)

livetimes = _loader.livetime_loader()
for name, livetime in sorted(livetimes.items()):
    print("\n# Setting up LLH for sample '{}'".format(name))
    # Get exp data and MC
    exp = _loader.exp_data_loader(name)[name]
    mc = _loader.mc_loader(name)[name]

    # Setup the energy LLH model with fixed index, only ns is fitted
    settings = _loader.settings_loader(name, skylab_bins=use_skylab_bins)[name]
    llh_model = EnergyLLH(**settings["llh_model_opts"])
    llh = PointSourceLLH(exp, mc, livetime, llh_model, scramble=scramble,
                         **settings["llh_opts"])
    multillh.add_sample(name, llh)

    del exp
    del mc
    gc.collect()

# Load the BG only distribution
with gzip.open(pdf_path) as fp:
    bg_ts_dist = ExpTailEmpiricalDist.from_json(fp)
    print("Loaded BG only TS distribution from:\n    {}".format(pdf_path))

# ##############################################################################
# Do a single fit to cache all values
bf_ts, bf_params = multillh.fit_source(
    src_ra=ra, src_dec=dec, src_w=w, scramble=scramble)
bf_ns, bf_gamma = bf_params["nsources"], bf_params["gamma"]

# Scan a grid in 'nsources' and 'gamma' fit paramters
ns_grid = np.arange(0., 50. + 0.05, 0.05)
gamma_grid = np.arange(1., 4. + 0.05, 0.05)

xx, yy = map(np.ravel, np.meshgrid(ns_grid, gamma_grid))
llh_values = np.zeros(len(ns_grid) * len(gamma_grid), dtype=float)
sigmas = np.zeros(len(ns_grid) * len(gamma_grid), dtype=float)

for i, (nsi, gi) in enumerate(zip(xx, yy)):
    print("scanning point {:d}/{:d}: ".format(i + 1, len(llh_values)) +
          "({:.1f}|{:.1f})".format(nsi, gi))
    llh_values[i], _ = multillh.llh(nsources=nsi, gamma=gi)
    sigmas[i] = prob2sigma(1. - bg_ts_dist.sf(llh_values[i]))
# ##############################################################################

# Format: ns, gamma, LLH value
fname = os.path.join(outpath, "llh_values.txt")
np.savetxt(fname=fname, X=np.vstack((xx, yy, llh_values, sigmas)).T,
           header="nsources gamma llh_value sigma", delimiter="\t")
