# coding: utf-8

"""
Single job of performance trials.
Loads data and settings and builds the models, likelihoods and injectors to do
the trials with.
"""

from __future__ import division, print_function

import gc  # Manual garbage collection
import os
import json
import gzip
import argparse
import numpy as np
import numpy.lib.recfunctions
from time import time

from skylab.ps_llh import PointSourceLLH, MultiPointSourceLLH
from skylab.llh_models import EnergyLLH

from tdepps.ps import HealpyPowerLawFluxInjector, MultiPowerLawFluxInjector
from tdepps.utils import make_src_records

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


def convert_signal_sample(sample, skylab_llh):
    """
    Converts a tdepps PowerLawFluxInjector signal sample to a skylab
    PointSourceLLH compatible form.
    For this, a new field 'sinDec' and 'time' needs to be created.
    'time' is not used in the steady scenario here, so a constant value is
    assigned.
    Additionally skylab uses an integer numbering for each sample, while tdepps
    uses the sample names thorughout.

    Parameters
    ----------
    sample : dict
        tdepps signal injector sample.
    skylab_llh :
        skylab MultiPointSourceLLH instance.

    Returns
    -------
    sl_sample : dict
        skylab compatible sample, with integer enum keys and added field
        ``'sinDec'``.
    """
    sam2enum = {v: k for k, v in skylab_llh._enums.items()}

    for name, sam in sample.items():
        # First convert to integer enums
        enum = sam2enum[name]
        sample[enum] = sample.pop(name)
        # Add sinDec and time info in
        # times can be constant, because they are not used here
        sample[enum] = numpy.lib.recfunctions.append_fields(
            sam, names="sinDec", usemask=False, data=np.sin(sam["dec"]))
    return sample


parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--mu", type=float,
                    help="Current signal strength scan grid point.")
parser.add_argument("--gamma", type=float, default=2., help="Spectral index " +
                    "to use for the injected unbroken power law.")
parser.add_argument("--E0", type=float, default=1., help="Normalization " +
                    "energy in GeV for the power law.")
parser.add_argument("--rnd_seed", type=int, help="Used random seed.")
parser.add_argument("--ntrials", type=int, help="Trials per mu grid point.")
parser.add_argument("--use_skylab_bins", action="store_true", default=False,
                    help="If given, uses the original skylab binning.")

args = parser.parse_args()
mu = args.mu
gamma_inj = args.gamma
E0_inj = args.E0
rnd_seed = args.rnd_seed
ntrials = args.ntrials
use_skylab_bins = args.use_skylab_bins

# Set signal strength scan grid
print("Injecting {:.2f} signal events.".format(mu))
print("Flux setting:")
print(" - gamma = {:.2f}".format(gamma_inj))
print(" - E0    = {:.1f} GeV".format(E0_inj))

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    outpath = os.path.join(PATHS.data, "performance_skylab_bins")
else:
    outpath = os.path.join(PATHS.data, "performance")

if not os.path.isdir(outpath):
    os.makedirs(outpath)

# Extract source info
sources = _loader.source_list_loader()
srcs_rec = np.lib.recfunctions.drop_fields(make_src_records(sources, 0, 0),
                                           drop_names=["dt0", "dt1"])
nsrcs = len(sources)
src_ra = srcs_rec["ra"]
src_dec = srcs_rec["dec"]
# Theo weights need to be normalized manually in skylab?
src_w = np.ones(nsrcs, dtype=float) / float(nsrcs)
srcs_rec["w_theo"] = src_w
# Load healpy reco map for each source
src_maps = _loader.source_map_loader(sources)
assert nsrcs == len(src_maps)

# Create the multi LLH
rndgen = np.random.RandomState(rnd_seed)
multillh = MultiPointSourceLLH(seed=rnd_seed + 1000, ncpu=1)
multiinj = MultiPowerLawFluxInjector(random_state=rndgen)

# Injector setting
inj_opts = {
    "mode": "band",
    "inj_sigma": 3.,
    "sindec_inj_width": 0.035,
    "dec_range": np.array([-np.pi / 2., np.pi / 2.])
    }

sig_injs = {}
livetimes = _loader.livetime_loader()
for name, livetime in sorted(livetimes.items()):
    print("\n# Setting up LLH for sample '{}'".format(name))
    # Get exp data and MC
    exp = _loader.exp_data_loader(name)[name]
    mc = _loader.mc_loader(name)[name]

    # Create the healpy injectors from tdepps, but they also work with skylab
    # with a small adapter
    sig_injs[name] = HealpyPowerLawFluxInjector(
        gamma=gamma_inj, E0=E0_inj, inj_opts=inj_opts, random_state=rndgen)
    sig_injs[name].fit(srcs_rec, src_maps=src_maps, MC=mc, livetime=livetime)

    # Strip unused fields  from data and mc to better connect to skylab
    keep = sig_injs[name].provided_data
    drop = filter(lambda s: s not in keep, exp.dtype.names)
    exp = np.lib.recfunctions.drop_fields(exp, drop)
    print("Stripped '{}' data from fields:\n  {}".format(name, drop))

    # Setup the energy LLH model with fixed index, only ns is fitted
    settings = _loader.settings_loader(name, skylab_bins=use_skylab_bins)[name]
    llh_model = EnergyLLH(**settings["llh_model_opts"])
    llh = PointSourceLLH(exp, mc, livetime, llh_model, scramble=True,
                         **settings["llh_opts"])

    multillh.add_sample(name, llh)

    del exp
    del mc
    gc.collect()

multiinj.fit(sig_injs)
del src_maps
gc.collect()

# Do performance trials by injecting on a grid
print(":: Starting {} performance trials ::".format(ntrials))
_t0 = time()

# Store true injected number of signal events per trial
ts = np.zeros(ntrials, dtype=float)
ns = np.zeros(ntrials, dtype=float)
gamma = np.zeros(ntrials, dtype=float)
nsig_injected = np.zeros(ntrials, dtype=int)

tenth = ntrials // 10
for i in range(ntrials):
    # Sample signal events and insert into the trial method
    nsig = rndgen.poisson(lam=mu, size=1)
    signal_sam = multiinj.sample(nsig)

    # Convert to a form skylab understands and does not reject
    signal_sam = convert_signal_sample(signal_sam, multillh)

    res = multillh.fit_source(src_ra=src_ra, src_dec=src_dec, src_w=src_w,
                              scramble=True, inject=signal_sam)
    bf_params = res[1]

    # Store results
    nsig_injected[i] = nsig
    ts[i] = res[0]
    ns[i] = bf_params["nsources"]
    gamma[i] = bf_params["gamma"]

    if (i + 1) % tenth == 0:
        print("{:.0%}".format((i + 1) / ntrials))

print(":: Done. {} ::".format(sec2str(time() - _t0)))

# Save trials
out = {
    "ts": ts.tolist(),
    "ns": ns.tolist(),
    "gamma": gamma.tolist(),
    "ninj": nsig_injected.tolist(),
    "mu": mu,
    "rnd_seed": rnd_seed,
    "ntrials": ntrials,
    "gamma_inj": gamma_inj,
    "E0_inj": E0_inj,
    }

_fname = os.path.join(outpath, "performance_mu={:.3f}.json.gz".format(mu))
with gzip.open(_fname, "w") as fp:
    json.dump(out, fp=fp, indent=1)
    print("Saved to:\n  {}".format(_fname))
