# coding: utf-8

"""
Create jobfiles for `05-bg_trials.py`.

##############################################################################
# Used seed range for bg trial jobs: [0, 200000)
##############################################################################
"""

import os
import numpy as np
import argparse

from dagman import dagman
from _paths import PATHS


MIN_SEED, MAX_SEED = 0, 200000

parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--testmode", action="store_true",
                    help="If given, make only a single testjob")
parser.add_argument("--use_skylab_bins", action="store_true", default=False,
                    help="If given, makes jobs that use the official " +
                    "skylab binning")
args = parser.parse_args()
testmode = args.testmode
use_skylab_bins = args.use_skylab_bins

job_creator = dagman.DAGManJobCreator(mem=5)
job_name = "hese_stacking"

script = os.path.join(PATHS.repo, "05-bg_trials.py")

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    job_dir = os.path.join(PATHS.jobs, "bg_trials_skylab_bins")
else:
    job_dir = os.path.join(PATHS.jobs, "bg_trials")

if testmode:
    ntrials = 1000
    ntrials_per_job = 1000
else:
    ntrials = int(1e6)
    ntrials_per_job = int(1e4)

njobs_tot = ntrials / ntrials_per_job
if ntrials != njobs_tot * ntrials_per_job:
    raise ValueError("Job settings does not lead to exactly " +
                     "{} trials".format(ntrials))
print("Preparing {} total background trials".format(ntrials))
print("  - {} trials per job".format(ntrials_per_job))
print("Creating {} total jobfiles for all time windows".format(njobs_tot))

# Make unique job identifiers:
# job_ids: 000 ... 999
lead_zeros = int(np.ceil(np.log10(njobs_tot)))
job_ids = np.array(["{1:0{0:d}d}".format(lead_zeros, i) for i
                    in range(njobs_tot)])

job_args = {
    "rnd_seed": np.arange(MIN_SEED + 10000,
                          MIN_SEED + 10000 + njobs_tot).astype(int),
    "ntrials": njobs_tot * [ntrials_per_job],
    "job_id": job_ids,
    }

if use_skylab_bins:
    job_args.update(use_skylab_bins=njobs_tot * ["__FLAG__"])

if (np.any(job_args["rnd_seed"] < MIN_SEED) or
        np.any(job_args["rnd_seed"] >= MAX_SEED)):
    raise RuntimeError("Used a seed outside the allowed range!")

job_creator.create_job(script=script, job_args=job_args,
                       job_name=job_name, job_dir=job_dir, overwrite=True)
