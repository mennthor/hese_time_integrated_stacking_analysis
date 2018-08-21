# coding: utf-8

"""
Create jobfiles for `09-performance.py`.

##############################################################################
# Used seed range for performance trial jobs: [200000, 300000)
##############################################################################
"""

import os
import numpy as np
import argparse

from dagman import dagman
from _paths import PATHS


MIN_SEED, MAX_SEED = 200000, 300000

parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--testmode", action="store_true",
                    help="If given, make only a single testjob")
parser.add_argument("--use_skylab_bins", action="store_true", default=False,
                    help="If given, makes jobs that use the official " +
                    "skylab binning")
args = parser.parse_args()
testmode = args.testmode
use_skylab_bins = args.use_skylab_bins

job_creator = dagman.DAGManJobCreator(mem=8)
job_name = "hese_stacking"

script = os.path.join(PATHS.repo, "09-performance.py")

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    job_dir = os.path.join(PATHS.jobs, "performance_skylab_bins")
else:
    job_dir = os.path.join(PATHS.jobs, "performance")

# Need to do enough trials for a proper TS estimation per mu grid point
if testmode:
    ntrials_per_job = 100
else:
    ntrials_per_job = int(2e4)

# Set mu scan range (explored by trial and error)
start, stop, step = 1., 200., 1.
mu = np.arange(start, stop + step, step)

# Set physics parameters
gamma = 2.  # flux ~ (E / E0)^-gamma
E0 = 1.     # in GeV

njobs_tot = len(mu)
print("Preparing {} performance trial jobs".format(njobs_tot))
print("  - {} trials per job".format(ntrials_per_job))

# Make unique job identifiers:
job_args = {
    "mu": mu,
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
