# coding: utf-8

"""
Create jobfiles for `10-differential_perf.py`

##############################################################################
# Used seed range for performance trial jobs: [400000, 500000)
##############################################################################
"""

import os
import numpy as np
import argparse

from dagman import dagman
from _paths import PATHS


MIN_SEED, MAX_SEED = 400000, 500000

parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--use_skylab_bins", action="store_true", default=False,
                    help="If given, makes jobs that use the official " +
                    "skylab binning")
args = parser.parse_args()
use_skylab_bins = args.use_skylab_bins

print("Preparing job files for differential performance")
job_creator = dagman.DAGManJobCreator(mem=8)
job_name = "diff_perf"
script = os.path.join(PATHS.repo, "10-differential_perf.py")

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    job_dir = os.path.join(PATHS.jobs, "diff_perf_skylab_bins_CROSSCHECK")
else:
    job_dir = os.path.join(PATHS.jobs, "diff_perf_CROSSCHECK")

# Need to do enough trials for a proper TS estimation per mu grid point
ntrials_per_job = 5000

# Set mu scan range
# Explored by trial and error from examining chi2 fit for disc. pot.
start, stop, step = 1., 300., 10.
# For a pretty Neyman plane, you might want 1, 100, 1 for tighter mu bins
# and 1e4 trials per mu
mu = np.arange(start, stop + step, step)
n_mus = len(mu)

# Set log10 energy borders for the signal injector to inject differential in E
dlog_E = 0.5
# Cover with finer bins in region with most numu tracks [1e3-1e6]
log_E_nu_bins = np.unique(np.concatenate((
    np.arange(3., 6., 0.125),
    )))
# [1, 9] covers all used MC sets' true log10 energy ranges
# log_E_nu_bins = np.arange(2, 9 + dlog_E, dlog_E)
n_log_E_nu_bins = len(log_E_nu_bins) - 1

# Set physics parameters
gamma = 2.  # flux ~ (E / E0)^-gamma
E0 = 1.     # in GeV

njobs_tot = n_mus * n_log_E_nu_bins
print("Preparing {} performance trial jobs".format(njobs_tot))
print("  - {} logEnu bins per job".format(n_log_E_nu_bins))
print("  - {} trials per job".format(ntrials_per_job))

# mus: mu[0], ..., mu[0], mu[1], .., mu[1], ..., mu[-1], ..., mu[-1]
mus = np.concatenate([n_log_E_nu_bins * [mui] for mui in mu])

# Bin borders: [log_E_nu_bins, log_E_nu_bins, ..., log_E_nu_bins]
bins_lo = np.tile(log_E_nu_bins[:-1], reps=n_mus)
bins_hi = np.tile(log_E_nu_bins[1:], reps=n_mus)

# Make unique job identifiers:
job_args = {
    "mu": mus,
    "log10_E_nu_lo": bins_lo,
    "log10_E_nu_hi": bins_hi,
    "rnd_seed": np.arange(MIN_SEED,
                          MIN_SEED + njobs_tot).astype(int),
    "ntrials": njobs_tot * [ntrials_per_job],
    "gamma": njobs_tot * [gamma],
    "E0": njobs_tot * [E0],
    }

if use_skylab_bins:
    job_args.update(use_skylab_bins=njobs_tot * ["__FLAG__"])

if (np.any(job_args["rnd_seed"] < MIN_SEED) or
        np.any(job_args["rnd_seed"] >= MAX_SEED)):
    raise RuntimeError("Used a seed outside the allowed range!")

job_creator.create_job(script=script, job_args=job_args,
                       job_name=job_name, job_dir=job_dir, overwrite=True)
