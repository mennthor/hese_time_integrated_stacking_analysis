# coding: utf-8

"""
1) Remove HESE like events identified in `02-check_hese_mc_ids` from the
   simulation files.
2) Remove source HESE events from data sets.
"""

import os
import json
import gzip
import numpy as np

from skylab.datasets import Datasets

from _paths import PATHS
from _loader import source_list_loader
from myi3scripts import arr2str


def remove_hese_from_mc(mc, heseids):
    """
    Mask all values in ``mc`` that have the same run and event ID combination
    as in ``heseids``.

    Parameters
    ----------
    mc : record-array
        MC data, needs names ``'Run', 'Event'``.
    heseids : dict or record-array
        Needs names / keys ``'run_id', 'event_id``.

    Returns
    -------
    is_hese_like : array-like, shape (len(mc),)
        Mask: ``True`` for each event in ``mc`` that is HESE like.
    """
    # Make combined IDs to easily match against HESE IDs with `np.isin`
    factor_mc = 10**np.ceil(np.log10(np.amax(mc["Event"])))
    _evids = np.atleast_1d(heseids["event_id"])
    factor_hese = 10**np.ceil(np.log10(np.amax(_evids)))
    factor = max(factor_mc, factor_hese)

    combined_mcids = (factor * mc["Run"] + mc["Event"]).astype(int)
    assert np.all(combined_mcids > factor)  # Is int overflow a thing here?

    _runids = np.atleast_1d(heseids["run_id"])
    combined_heseids = (factor * _runids + _evids).astype(int)
    assert np.all(combined_heseids > factor)

    # Check which MC event is tagged as HESE like
    is_hese_like = np.isin(combined_mcids, combined_heseids)
    print("  Found {} / {} HESE like events in MC".format(np.sum(is_hese_like),
                                                          len(mc)))
    return is_hese_like


def remove_hese_from_exp_data(exp, src_dicts):
    """
    Mask source events in experimental data sample.

    Parameters
    ----------
    exp : record-array
        Experimental data, needs names ``'Run', 'Event'``.

    src_dicts : list of dicts
        One dict per source, must have keys ``'run_id', 'event_id'``.

    Returns
    -------
    is_hese_src : array-like, shape (len(exp),)
        Mask: ``True`` where a source HESE event is in the sample.
    """
    is_hese_src = np.zeros(len(exp), dtype=bool)
    # Check each source and combine masks
    print("  HESE events in experimental data:")
    for i, src in enumerate(src_dicts):
        mask_i = ((src["event_id"] == exp["Event"]) &
                  (src["run_id"] == exp["Run"]))
        is_hese_src = np.logical_or(is_hese_src, mask_i)
        print("  - Source {}: {}. Dec: {:.2f} deg. logE: {} log(GeV)".format(
            i, np.sum(mask_i), np.rad2deg(src["dec"]), exp[mask_i]["logE"]))
    return is_hese_src


exp_data_outpath = os.path.join(PATHS.data, "exp_no_hese")
mc_outpath = os.path.join(PATHS.data, "mc_no_hese")
out_paths = {"exp": exp_data_outpath, "mc": mc_outpath}
for _p in out_paths.values():
    if not os.path.isdir(_p):
        os.makedirs(_p)

# Load source list
sources = source_list_loader()

# Load needed data and MC from PS track and add in one year of GFU sample
ps_tracks = Datasets["PointSourceTracks"]
# We don't use "IC40", "IC59" because we don't have HESE removed MCs available
ps_sample_names = ["IC79", "IC86, 2011", "IC86, 2012-2014"]
gfu_tracks = Datasets["GFU"]
gfu_sample_names = ["IC86, 2015"]
all_sample_names = sorted(ps_sample_names + gfu_sample_names)

# Base MC is same for multiple samples, match names here
name2heseid_file = {
    "IC79": "IC79.json.gz",
    "IC86_2011": "IC86_2011.json.gz",
    "IC86_2012-2014": "IC86_2012-2015.json.gz",
    "IC86_2015": "IC86_2012-2015.json.gz"
}

# Save livetimes in separate JSON
livetimes = {}

for name in all_sample_names:
    print("Working with sample {}".format(name))

    if name in ps_sample_names:
        tracks = ps_tracks
    else:
        tracks = gfu_tracks

    exp, mc, livetime_days = tracks.season(name)

    exp_file, mc_file = tracks.files(name)
    print("  Loaded {} track sample from skylab:".format(
        "PS" if name in ps_sample_names else "GFU"))
    _info = arr2str(exp_file if isinstance(exp_file, list) else [exp_file],
                    sep="\n    ")
    print("    Data:\n      {}".format(_info))
    print("    MC  :\n      {}".format(mc_file))

    name = name.replace(", ", "_")

    # Save livetime
    livetimes[name] = livetime_days

    # Remove HESE source events from ontime data
    is_hese_src = remove_hese_from_exp_data(exp, sources)
    exp = exp[~is_hese_src]

    # Remove HESE like events from MC
    _fname = os.path.join(PATHS.local, "check_hese_mc_ids",
                          name2heseid_file[name])
    with gzip.open(_fname) as _file:
        heseids = json.load(_file)
        print("  Loaded HESE like MC IDs from :\n    {}".format(_fname))
    is_hese_like = remove_hese_from_mc(mc, heseids)
    mc = mc[~is_hese_like]

    # Save, also in npy format
    print("  Saving exp data w/o HESE sources and non-HESE like MCs at:")
    out_arrs = {"exp": exp, "mc": mc}
    for data_name in out_paths.keys():
        _fname = os.path.join(out_paths[data_name], name + ".npy")
        np.save(file=_fname, arr=out_arrs[data_name])
        print("    {:3s}: '{}'".format(data_name, _fname))

# Save livetime information
_fname = os.path.join(out_paths["exp"], "livetimes.json")
with open(_fname, "w") as fp:
    json.dump(livetimes, fp=fp, indent=2)
    print("Saved experimental livetimes in days at:\n  {}".format(_fname))
