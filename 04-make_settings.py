# coding: utf-8

"""
Generate setting files to build the analysis objects with, including:

1) Settings for the LLH model.
2) Settings for the LLH.

By storing and reading the settings we can change them easily for tests without
messing up stuff in each analysis script.
"""

import os
import json
import numpy as np
import argparse

from skylab.datasets import Datasets

from _paths import PATHS
import _loader


parser = argparse.ArgumentParser(description="hese_stacking")
parser.add_argument("--use_skylab_bins", action="store_true", default=False)
args = parser.parse_args()
use_skylab_bins = args.use_skylab_bins

if use_skylab_bins:
    print("Using official skylab binnings for the PDFs!")
    outpath = os.path.join(PATHS.local, "settings_skylab_bins")
else:
    outpath = os.path.join(PATHS.local, "settings")

if not os.path.isdir(outpath):
    os.makedirs(outpath)

# Binning used for injector and LLH models alike
# Finer resolution around the horizon region, where we usually switch the event
# selections from northern to southern samples
hor = np.sin(np.deg2rad(30))
sd_lo, sd_hi = -1., 1.
sindec_bins = np.unique(np.concatenate([
                        np.linspace(sd_lo, -hor, 3 + 1),  # south
                        np.linspace(-hor, +hor, 14 + 1),  # horizon
                        np.linspace(+hor, sd_hi, 3 + 1),  # north
                        ]))

pstracks = Datasets["PointSourceTracks"]
gfu = Datasets["GFU"]

key2set = {
    "IC79": pstracks,
    "IC86_2011": pstracks,
    "IC86_2012-2014": pstracks,
    "IC86_2015": gfu,
}

key2skylab = {
    "IC79": "IC79",
    "IC86_2011": "IC86, 2011",
    "IC86_2012-2014": "IC86, 2012-2014",
    "IC86_2015": "IC86, 2015",
}

# Make settings for each module per sample
sample_names = sorted(_loader.livetime_loader().keys())
for key in sample_names:
    print("Building settings file for sample '{}'".format(key))
    # Load data that settings depend on
    mc = _loader.mc_loader(key)[key]

    # :: LLH model ::
    # This is kind of arbitrary, but seems to produce robust PDFs
    logE_bins = np.linspace(
        np.floor(np.amin(mc["logE"])), np.ceil(np.amax(mc["logE"])), 30)

    # Get the official binnings from skylab instead
    if use_skylab_bins:
        sindec_bins = key2set[key].sinDec_bins(key2skylab[key])
        logE_bins = key2set[key].energy_bins(key2skylab[key])

    llh_model_opts = {
        # Watch the order of the bins in the list!
        "twodim_bins": [logE_bins.tolist(), sindec_bins.tolist()],
        "allow_empty": True,
        "kernel": 1.,              # Smoothing kernel width for energy PDF
        "seed": 2.,                # gamma index seed
        "bounds": [1., 4.],        # gamma index bounds
        "fill_method": "classic",
        }

    # :: LLH ::
    llh_opts = {
        "mode": "all",
        "nsource": 1.,
        }

    # :: Save settings per sample ::
    settings = {
        "llh_model_opts": llh_model_opts,
        "llh_opts": llh_opts,
        }

    _fname = os.path.join(outpath, key + ".json")
    with open(_fname, "w") as fp:
        json.dump(obj=settings, fp=fp, indent=2)
        print("  Saved to:\n    {}".format(_fname))
