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
from skylab.spectral_models import PowerLaw

from tdepps.utils.stats import ExpTailEmpiricalDist, prob2sigma

from _paths import PATHS, check_dir
import _loader


def dict_print(d):
    shift = max(map(len, d.keys())) + 1
    for key, val in d.items():
        print("{0:{1:d}s}: ".format(key, shift), val)


parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--n_unblindings", type=int, default=1,
                    help="Do this many unblindings in a row, default is 1.")
parser.add_argument("--really_unblind", action="store_true")
parser.add_argument("--use_skylab_bins", action="store_true", default=False)
args = parser.parse_args()
n_unblindings = args.n_unblindings
really_unblind = args.really_unblind
use_skylab_bins = args.use_skylab_bins

if n_unblindings < 1:
    raise ValueError("'n_unblindings' must give an int > 0.")
else:
    print("Doing {:d} unblindings in a row.".format(n_unblindings))

if not really_unblind:
    print("## Not really unblinding, events get scrambled!   ##")
    scramble = True
else:
    print("## In real unblinding mode, it's getting serious! ##")
    scramble = False

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    outpath = os.path.join(PATHS.local, "unblinding_skylab_bins")
    pdf_path = os.path.join(PATHS.local, "bg_pdf_skylab_bins", "bg_pdf.json.gz")
else:
    outpath = os.path.join(PATHS.local, "unblinding")
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
result = {
    "ts": [],
    "ns": [],
    "gamma": [],
    "pval": [],
    "sigma": [],
    "n_unblindings": n_unblindings,
    "scrambled": scramble,
    }

# Get the unblinded result(s)
for i in tqdm(range(n_unblindings)):
    bf_ts, bf_params = multillh.fit_source(
        src_ra=ra, src_dec=dec, src_w=w, scramble=scramble)

    # Calculate the p-value and significance from the BG only TS distribution
    pval = bg_ts_dist.sf(bf_ts)
    sigma = prob2sigma(1. - pval)

    result["ts"].append(bf_ts)
    result["ns"].append(bf_params["nsources"])
    result["gamma"].append(bf_params["gamma"])
    result["pval"].append(pval[0])
    result["sigma"].append(sigma[0])

# ##############################################################################

# Print result
if n_unblindings == 1:
    for key, val in result.items():
        try:
            result[key] = val[0]
        except:
            pass
else:
    ts = np.array(result["ts"])
    nzero_trials = np.sum(ts == 0)
    print("Number of zero trials: {}".format(nzero_trials))
    print("Perc. of zero trials : {:.2%}".format(nzero_trials / n_unblindings))
    print("Number of trials > 1 : {}".format(np.sum(ts > 1)))

dict_print(result)

# Save as JSON
fname = os.path.join(outpath, "result.json")
with open(fname, "w") as fp:
    json.dump(result, fp=fp, indent=1)
    print("Saved to:\n  {}".format(fname))
