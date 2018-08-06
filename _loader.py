# coding: utf-8

"""
Loader methods for used data formats. If a format changes, we only need to
change the loading part here once.
"""

import os as _os
import re as _re
import json as _json
import gzip as _gzip
import numpy as _np
from glob import glob as _glob

import tdepps.utils.stats as _stats

from _paths import PATHS as _PATHS


def result_loader():
    """
    Load the unblinding result file.

    Returns
    -------
    results : dict
        Result dictionary.
    """
    with open(_os.path.join(_PATHS.local, "unblinding", "result.json")) as fp:
        result = _json.load(fp)
    return result


def bg_pdf_loader(idx=None):
    """
    Loads background trial test statisitc distribution objects of type
    ``tdepps.utils.stats.emp_with_exp_tail_dist``.

    Parameters
    ----------
    idx : array-like or int or 'all' or ``None``, optional
        Which time window to load the background PDF for. If ``'all'``, all are
        loaded, if ``None`` a list of valid indices is returned.
        (default: ``None``)

    Returns
    -------
    pdfs : dict or list
        Dict with indices as key(s) and the distribution object(s) as value(s).
        If ``idx`` was ``None`` an array of valid indices is returned.
    """
    folder = _os.path.join(_PATHS.local, "bg_pdfs")
    files = sorted(_glob(_os.path.join(folder, "*")))
    file_names = map(_os.path.basename, files)

    if (idx is None) or (idx == "all"):
        regex = _re.compile(".*tw_([0-9]*)\.json\.gz")
        all_idx = []
        for fn in file_names:
            res = _re.match(regex, fn)
            all_idx.append(int(res.group(1)))
        all_idx = _np.sort(all_idx)
        if idx is None:
            return all_idx
    else:
        all_idx = _np.atleast_1d(idx)

    pdfs = {}
    for idx in all_idx:
        file_id = file_names.index("bg_pdf_tw_{:02d}.json.gz".format(idx))
        fname = files[file_id]
        print("Load bg PDF for time window {:d} from:\n  {}".format(idx,
                                                                    fname))
        with _gzip.open(fname) as json_file:
            pdfs[idx] = (_stats.ExpTailEmpiricalDist.from_json(json_file))

    return pdfs


def source_list_loader():
    """
    Load source lists.

    Returns
    -------
    sources : list of dicts
        List of dictionaries, one dict per source.
    """
    source_file = _os.path.join(_PATHS.local, "source_list", "source_list.json")
    with open(source_file) as _file:
        sources = _json.load(_file)

    print("Loaded source list from:\n  {}".format(source_file))
    return sources


def source_map_loader(src_list):
    """
    Load the reco LLH map for a given source from the source list loader.

    Parameters
    ----------
    src_list : list of dicts, shape (nsrcs)
        List of source dicts, as provided by ``source_list_loader``. Each dict
        must have key ``'map_path'``.

    Returns
    -------
    healpy_maps : array-like, shape (nsrcs, npix)
        Healpy map belonging to the given source for each source in the same
        order as in ``src_list``.
    """
    healpy_maps = []
    for src in src_list:
        fpath = src["map_path"]
        print("Loading map for source: {}".format(
            _os.path.basename(fpath)))
        with _gzip.open(fpath) as f:
            src = _json.load(f)

        healpy_maps.append(_np.array(src["map"]))

    return _np.atleast_2d(healpy_maps)


def settings_loader(names=None, skylab_bins=False):
    """
    Parameters
    ----------
    names : list of str or None or 'all', optional
        Name(s) of the datasets(s) to load. If ``None`` returns a list of all
        possible names. If ``'all'``, returns all available runlists.
        (default: ``None``)
    skylab_bins : bool, optional
        If ``True`` uses the settings with the official skylab binnings to build
        the PDFs. (default: ``False``)

    Returns
    -------
    offtime_data : dict or list
        Dict with name(s) as key(s) and the offtime data record array(s) as
        value(s). If ``names`` was ``None``, returns a list of possible input
        names. If ``names`` was ``'all'`` returns all available data array(s)
        the dict.
    """
    if skylab_bins:
        folder = _os.path.join(_PATHS.local, "settings_skylab_bins")
    else:
        folder = _os.path.join(_PATHS.local, "settings")
    return _common_loader(names, folder=folder, info="settings")


def livetime_loader():
    """
    Loads livetimes for experimental data.

    Returns
    -------
    livetimes : dict
        Livetimes in days per sample.
    """
    _fname = _os.path.join(_PATHS.data, "exp_no_hese", "livetimes.json")
    with open(_fname) as fp:
        livetimes = _json.load(fp)
    return livetimes


def exp_data_loader(names=None):
    """
    Parameters
    ----------
    names : list of str or None or 'all', optional
        Name(s) of the datasets(s) to load. If ``None`` returns a list of all
        possible names. If ``'all'``, returns all available runlists.
        (default: ``None``)

    Returns
    -------
    offtime_data : dict or list
        Dict with name(s) as key(s) and the offtime data record array(s) as
        value(s). If ``names`` was ``None``, returns a list of possible input
        names. If ``names`` was ``'all'`` returns all available data array(s)
        the dict.
    """
    folder = _os.path.join(_PATHS.data, "exp_no_hese")
    return _common_loader(names, folder=folder, info="experimental data")


def mc_loader(names=None):
    """
    Parameters
    ----------
    names : list of str or None or 'all', optional
        Name(s) of the datasets(s) to load. If ``None`` returns a list of all
        possible names. If ``'all'``, returns all available runlists.
        (default: ``None``)

    Returns
    -------
    mc : dict or list
        Dict with name(s) as key(s) and the MC record array(s) as value(s). If
        ``names`` was ``None``, returns a list of possible input names. If
        ``names`` was ``'all'`` returns all available MC array(s) in the dict.
    """
    folder = _os.path.join(_PATHS.data, "mc_no_hese")
    return _common_loader(names, folder=folder, info="MC")


def _common_loader(names, folder, info):
    """
    Outsourced some common loader code.

    Parameters
    ----------
    names : list of str, None or 'all'
        See explicit loaders.
    folder : string
        Full path to folder from where to load the data.
    info : str
        Info for print.

    Returns
    -------
    data : dict
        See explicit loader returns.
    """
    files = sorted(_glob(_os.path.join(folder, "*")))
    file_names = map(lambda s: _os.path.splitext(_os.path.basename(s))[0],
                     files)

    if names is None:
        return file_names
    else:
        if names == "all":
            names = file_names
        elif not isinstance(names, list):
            names = [names]

    data = {}
    for name in names:
        idx = file_names.index(name)
        fname = files[idx]
        print("Load {} for sample {} from:\n  {}".format(info, name, fname))
        ext = _os.path.splitext(fname)[1]
        if ext == ".npy":
            data[name] = _np.load(fname)
        elif ext == ".json":
            with open(fname) as json_file:
                data[name] = _json.load(json_file)
        else:
            raise ValueError("Couldn't load unknown datatype: '{}'".format(ext))

    return data
