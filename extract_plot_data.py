# coding: utf-8

"""
Setup the analysis modules as used in the analysis chain to extract values for
plots from them.
"""

from __future__ import division, print_function

import os
import gc  # Manual garbage collection
import numpy as np
import json

from skylab.ps_llh import PointSourceLLH, MultiPointSourceLLH
from skylab.llh_models import EnergyLLH

from tdepps.ps import HealpyPowerLawFluxInjector, MultiPowerLawFluxInjector
from tdepps.utils import make_src_records

import _loader
from _paths import PATHS, check_dir


rnd_seed = 13462364
rndgen = np.random.RandomState(rnd_seed)

gamma_inj = 2.
E0_inj = 1.

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
    settings = _loader.settings_loader(name)[name]
    llh_model = EnergyLLH(**settings["llh_model_opts"])
    llh = PointSourceLLH(exp, mc, livetime, llh_model, scramble=True,
                         **settings["llh_opts"])

    # Add stuff needed for plots, unclear why this all stays empty...
    llh_model.sinDec_bins = np.array(llh_model.sinDec_bins)
    gammas = np.arange(1, 4.1, 0.1)
    llh_model._effA(llh.mc, llh.livetime, gamma=gammas)

    multillh.add_sample(name, llh)

    del exp
    del mc
    gc.collect()

multiinj.fit(sig_injs)
del src_maps
gc.collect()

# Fit a single source to setup all internals
res = multillh.fit_source(src_ra=src_ra, src_dec=src_dec, src_w=src_w,
                          scramble=True, inject=None)


# #############################################################################
# Prepare data for plots
# #############################################################################
llhs = multillh._samples
llh_mods = {k: llh_i.llh_model for k, llh_i in llhs.items()}

outpath = os.path.join(PATHS.repo, "out_local")
check_dir(outpath)

# #############################################################################
# Sample weights vs. gamma. sample_w has shape (len(gammas), nsamples)
sample_w = np.array([multillh.sample_weights(
    nsamples=4, src_dec=src_dec, gamma=gamma)[0] for gamma in gammas])

# Save as JSON as a list of weights per sample with length len(gammas)
out = {
    "gammas": gammas.tolist(),
    "sample_weights": [swi.tolist() for swi in sample_w.T],
    "enum": multillh._enums
    }
fname = os.path.join(outpath, "sample_split_weights.json")
with open(fname, "w") as fp:
    json.dump(out, fp=fp, indent=2)
    print("Saved sample weights for gamma stack to:\n  {}".format(fname))

# #############################################################################
# Background spline values
spl_vals = {}
sindec = np.linspace(-1, 1, 500)

for key in llhs.keys():
    llh_mod_i = llhs[key].llh_model
    evts = np.empty((len(sindec),), dtype=[("sinDec", float)])
    evts["sinDec"] = sindec
    spl_vals[key] = llh_mod_i.background(evts).tolist()

spl_vals["sindec"] = sindec.tolist()
spl_vals["enums"] = multillh._enums

fname = os.path.join(outpath, "background_spline_vals.json")
with open(fname, "w") as fp:
    json.dump(spl_vals, fp=fp, indent=2)
    print("Saved background spline vals to:\n  {}".format(fname))
# #############################################################################

# #############################################################################
# Energy SoB interpolated values for the whole gamma stack
# Scan intervals
logE = np.linspace(1, 10, 200)
sindec = np.linspace(-1, 1, 200)
# Bins are borders, evaluate in centers for proper pcolormesh
xmids = 0.5 * (sindec[:-1] + sindec[1:])
ymids = 0.5 * (logE[:-1] + logE[1:])
xx, yy = map(np.ravel, np.meshgrid(xmids, ymids))

for key in llhs.keys():
    name = multillh._enums[key]
    print("Getting energy SoB stack for LLH '{}' ({})".format(name, key))

    llh_i = llhs[key]
    llh_mod_i = llh_i.llh_model
    # This should be sindec and logE in this case
    evt_dtypes = llh_mod_i.hist_pars

    # Store points in event dict for weight method
    evts = np.empty((len(xx),), dtype=[(ni, float) for ni in evt_dtypes])
    evts["sinDec"] = xx
    evts["logE"] = yy

    # Get energy SoBs for all gammas in the stack
    w = [llh_mod_i.weight(ev=evts, gamma=gi)[0].reshape(len(xmids),
                                                        len(ymids)).tolist()
         for gi in gammas]

    out = {
        "gammas": gammas.tolist(),
        "sindec": sindec.tolist(),
        "logE": logE.tolist(),
        "order": "gamma, logE, sindec. pcolormesh(sindec, logE, w_stack[i])",
        "w_stack": w
        }

    fname = os.path.join(outpath, "energy_sob_stack_{}.json".format(name))
    with open(fname, "w") as fp:
        json.dump(out, fp=fp, indent=1)
        print("Saved SoB stack to:\n  {}".format(fname))
# #############################################################################
