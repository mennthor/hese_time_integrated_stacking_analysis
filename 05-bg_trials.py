# coding: utf-8

"""
Single job of background only trials.
Loads data and settings and builds the models, likelihoods and injectors to do
the trials with.
"""

import gc  # Manual garbage collection
import os
import json
import gzip
import argparse
import numpy as np
from time import time

from skylab.ps_llh import PointSourceLLH, MultiPointSourceLLH
from skylab.llh_models import EnergyLLH
from skylab.spectral_models import PowerLaw

from _paths import PATHS
import _loader


def sec2str(sec):
    factors = [24. * 60. * 60., 60. * 60., 60., 1.]
    labels = ["d", "h", "m", "s"]
    splits = []
    for i, factor in enumerate(factors):
        splits.append(sec // factor)
        sec -= splits[-1] * factor

    out = []
    for i, (f, l) in enumerate(zip(splits, labels)):
        # First entry has more digits for overflow
        if i == 0:
            out.append("{:.0f}{}".format(f, l))
        else:
            out.append("{:02.0f}{}".format(f, l))

    return ":".join(out)


parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--rnd_seed", type=int)
parser.add_argument("--ntrials", type=int)
parser.add_argument("--job_id", type=str)
parser.add_argument("--use_skylab_bins", action="store_true", default=False)
args = parser.parse_args()
rnd_seed = args.rnd_seed
ntrials = args.ntrials
job_id = args.job_id
use_skylab_bins = args.use_skylab_bins

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    outpath = os.path.join(PATHS.data, "bg_trials_skylab_bins")
else:
    outpath = os.path.join(PATHS.data, "bg_trials")

if not os.path.isdir(outpath):
    os.makedirs(outpath)


# Extract source info, use the same fixed spectral index -2 for all
sources = _loader.source_list_loader()
nsrcs = len(sources)
fixed_gamma = 2.
# Concrete norm shouldn't matter here, weights get normalized
fixed_spectrum = PowerLaw(A=1, gamma=fixed_gamma, E0=1)

ra = [src["ra"] for src in sources]
dec = [src["dec"] for src in sources]
# Theo weights need to be normalized manually in skylab?
w = np.ones(nsrcs, dtype=float) / float(nsrcs)

# Create the multi LLH
# multillh = MultiPointSourceLLH(seed=rnd_seed, ncpu=40)  #  For local testing
multillh = MultiPointSourceLLH(seed=rnd_seed, ncpu=4)

livetimes = _loader.livetime_loader()
for name, livetime in sorted(livetimes.items()):
    print("\n# Setting up LLH for sample '{}'".format(name))
    # Get exp data and MC
    exp = _loader.exp_data_loader(name)[name]
    mc = _loader.mc_loader(name)[name]

    # Setup the energy LLH model with fixed index, only ns is fitted
    settings = _loader.settings_loader(name, skylab_bins=use_skylab_bins)[name]
    llh_model = EnergyLLH(spectrum=fixed_spectrum,
                          **settings["llh_model_opts"])
    llh = PointSourceLLH(exp, mc, livetime, llh_model, scramble=True,
                         **settings["llh_opts"])

    multillh.add_sample(name, llh)

    del exp
    del mc
    gc.collect()


# Do background only trials
print(":: Starting {} background trials ::".format(ntrials))
_t0 = time()
trials = multillh.do_trials(ntrials, src_ra=ra, src_dec=dec, src_w=w)
print(":: Done. {} ::".format(sec2str(time() - _t0)))

# Save trials
out = {
    "ts": trials["TS"].tolist(),
    "ns": trials["nsources"].tolist(),
    "spectrum": trials["spectrum"][0],
    "rnd_seed": rnd_seed,
    "ntrials": ntrials,
    }

_fname = os.path.join(outpath, "job_{}.json.gz".format(job_id))
with gzip.open(_fname, "w") as fp:
    json.dump(out, fp=fp, indent=2)
    print("Saved to:\n  {}".format(_fname))
