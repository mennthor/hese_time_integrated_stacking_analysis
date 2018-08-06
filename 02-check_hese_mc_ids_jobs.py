# coding:utf-8

"""
Create jobfiles for `04-check_hese_mc_ids.py`.
"""

from __future__ import print_function, division
import warnings
import os
from glob import glob
import numpy as np

from _paths import PATHS
from dagman import dagman


def collect_structure(files):
    """
    Collect data set structure: folder and file paths.
    Assumes: path/[sets]/[folders: eg. 0000-0999]/[files.i3.bz2]
    and puts this strcuture in a dict.

    Parameters
    ----------
    files : dict
        Dictionary with  ``[dataset numbers]`` as keys. Value is a new dict with
        keys, values:

        - 'path', str:  Path to the dataset folders.
          ``path/[folders: eg. 0000-0999]/[files.i3.bz2]``.
        - 'gcd', str: Full path to a GCD file matching the simulation.
        - 'legacy', bool:  If ``True`` all files are assumed to be in a single
          folder structure. This may be useful if the files have been merged and
          moved to an analysis directory and are not available on /data/sim
          anymore. Assumed ``False`` if not given.

    Returns
    -------
    data : dict
        Structure of the dataset in the file system:
        - [sets] - dict
            + "path" - str
            + "folders" - dict
                + [folder] - dict
                    + "files" - list
                    + "nfiles" (in folder) - int
            + "nfiles" (in dataset) - int
            + "gcd" - str
    """
    data = {}
    for num in sorted(list(files.keys())):
        path = os.path.abspath(files[num]["path"])
        GCD = os.path.abspath(files[num]["gcd"])
        LEGACY = files[num].get("legacy", False)

        if not os.path.isfile(GCD):
            raise RuntimeError("GCD file '{}' doesn't exist.".format(GCD))
        if not LEGACY:  # Assume subfolders 00000-09999, ...
            folders = [os.path.join(path, s) for s in sorted(os.listdir(path))]
            folders = filter(os.path.isdir, folders)
        else:  # Assume everything is in this folder
            folders = [os.path.join(path, ".")]
        data[num] = {"path": path}
        data[num]["gcd"] = GCD
        data[num]["folders"] = {}

        print("\nDataset: ", num)
        print(" - Path:\n   ", path)
        print(" - GCD:\n   ", GCD)
        print(" - Folders:")
        nfiles = 0
        for folder in folders:
            folder_name = os.path.basename(folder)
            # Don't collect GCDs here but manually below
            sim_files = sorted([f for f in glob(folder + "/*.i3.bz2") if
                                not f.startswith("Geo")])
            data[num]["folders"][folder_name] = {"files": sim_files,
                                                 "nfiles": len(sim_files)}
            print("   + ", folder_name, ": ", len(sim_files), " files")
            nfiles += len(sim_files)
        data[num]["nfiles"] = nfiles
        print(" - Total files: ", nfiles)
        if nfiles == 0:
            raise RuntimeError("No files found for set '{}'. Make sure " +
                               "locations are valid or remove set from list.")

    return data


if __name__ == "__main__":
    # Job creation steering arguments
    job_creator = dagman.DAGManJobCreator(mem=1)
    job_name = "hese_transient_stacking"

    job_dir = os.path.join(PATHS.jobs, "check_hese_mc_ids")
    script = os.path.join(PATHS.repo, "02-check_hese_mc_ids.py")

    ###########################################################################
    # Collect dataset structure. Datset info on:
    #   https://grid.icecube.wisc.edu/simulation/dataset/<number>
    ###########################################################################
    print("\nCollecting simulation file paths:")
    # MC used for:
    # Point Source tracks IC86 2012-2014 datasets
    # GFU tracks IC86 2015-2017 datasets
    # Available as i3 on final level
    fpath = os.path.join("/data", "ana", "PointSource", "IC86_2012_PS", "files",
                         "sim", "2012", "neutrino-generator")
    gcd_path = os.path.join("/data", "sim", "IceCube", "2012", "filtered",
                            "level2", "neutrino-generator")
    files = {
        "11029": {
            "path": os.path.join(fpath, "11029"),
            "gcd": os.path.join(gcd_path, "11029", "00000-00999",
                                "GeoCalibDetectorStatus_2012.56063_V1.i3.gz"),
            "legacy": False,
            },
        "11069": {
            "path": os.path.join(fpath, "11069"),
            "gcd": os.path.join(gcd_path, "11069", "00000-00999",
                                "GeoCalibDetectorStatus_2012.56063_V1.i3.gz"),
            "legacy": False,
            },
        "11070": {
            "path": os.path.join(fpath, "11070"),
            "gcd": os.path.join(gcd_path, "11070", "00000-00999",
                                "GeoCalibDetectorStatus_2012.56063_V1.i3.gz"),
            "legacy": False,
            },
        }

    # MC used for PS tracks IC86 2011 datasets (only level 3 as i3)
    fpath = os.path.join("/data", "sim", "IceCube", "2011", "filtered",
                         "level2", "neutrino-generator")
    files.update({
        # This is stated on Stefan's wiki page
        "9095": {
            "path": os.path.join(fpath, "9095"),
            "gcd": os.path.join(fpath, "9095", "00000-00999",
                                "GeoCalibDetectorStatus_IC86.55697" +
                                "_corrected_V2.i3.gz"),
            "legacy": False,
        },
        # But these are the indices included in the final sample...
        "9701": {
            "path": os.path.join(fpath, "9701"),
            "gcd": os.path.join(fpath, "9701", "00000-00999/",
                                "GeoCalibDetectorStatus_IC86.55697" +
                                "_corrected_V2.i3.gz"),
            "legacy": False,
        },
        # These are definitely in there
        "9366": {
            "path": os.path.join(fpath, "9366"),
            "gcd": os.path.join(fpath, "9366", "00000-00999/",
                                "GeoCalibDetectorStatus_IC86.55697_" +
                                "corrected_V2.i3.gz"),
            "legacy": False,
        },
    })

    # MC used for PS tracks IC79 2010 datasets. Only ana level 3 mu as i3.
    # GCD is taken from similar set, which is still available on /data/sim
    files.update({
        "6308": {
            "path": os.path.join(
                "/data", "ana", "IC79", "level3-mu", "sim", "6308"),
            "gcd": os.path.join("/data", "sim", "IceCube", "2010", "filtered",
                                "level2", "neutrino-generator", "6359",
                                "00000-00999", "GeoCalibDetectorStatus_" +
                                "IC79.55380_corrected.i3.gz"),
            "legacy": True,
        }
    })

    # Combine to a single dict listing all files and folders
    data = collect_structure(files)

    ###########################################################################
    # Make job argument lists
    ###########################################################################
    # Job splitting args and job output paths
    nfiles_perjob = 100
    outpath = os.path.join(PATHS.data, "check_hese_mc_ids")
    if os.path.isdir(outpath):
        print("")
        warnings.warn("Output folder '{}' is already ".format(outpath) +
                      "existing. Check twice if nothing gets overwritten " +
                      "when starting jobs.", RuntimeWarning)
    else:
        os.makedirs(outpath)

    # Prepare job files by splitting arg list to jobfiles
    file_list = []
    gcd_list = []
    out_list = []
    print("")
    for num in data.keys():
        # Combine all files for the current sample
        folders = data[num]["folders"]
        files = np.concatenate([folder["files"] for folder in folders.values()])
        nfiles = len(files)
        assert nfiles == data[num]["nfiles"]

        # Split in chunks of ca. 10 files per job so jobs run fast
        nsplits = nfiles // nfiles_perjob
        chunked = np.array_split(files, nsplits)
        file_list.append(map(",".join, chunked))
        njobs = len(chunked)
        assert njobs == len(file_list[-1])
        print("Set: ", num)
        _min, _max = np.amin(map(len, chunked)), np.amax(map(len, chunked))
        if _min == _max:
            print("  Split {} files to {} jobs with {} ".format(
                nfiles, njobs, _min) + "files per job.")
        else:
            print("  Split {} files to {} jobs with {} - {} ".format(
                nfiles, njobs, _min, _max) + "files per job.")

        # Duplicate GCD info per job
        gcd_list.append(njobs * [data[num]["gcd"]])
        assert njobs == len(gcd_list[-1])
        assert np.all(np.array(gcd_list[-1]) == gcd_list[-1][0])

        # Outpath: ..[num]_<increment>.json
        lead_zeros = int(np.ceil(np.log10(nsplits)))
        outp = ["{2:}_{1:0{0:d}d}.json".format(lead_zeros, idx, num) for
                idx in np.arange(nsplits)]
        out_list.append([os.path.join(outpath, pi) for pi in outp])

    # Compress and write job files
    in_files = []
    map(in_files.extend, file_list)
    gcd_files = []
    map(gcd_files.extend, gcd_list)
    out_paths = []
    map(out_paths.extend, out_list)
    assert len(in_files) == len(out_paths) == len(gcd_files)
    print("\nTotal number of jobs: ", len(out_paths), "\n")

    job_args = {"infiles": in_files, "gcdfile": gcd_files, "outfile": out_paths}

    exe = [
        os.path.join("/cvmfs", "icecube.opensciencegrid.org", "py2-v2",
                     "RHEL_6_x86_64", "metaprojects", "combo", "stable",
                     "env-shell.sh"),
        os.path.join("/bin", "bash")
    ]
    job_creator.create_job(script=script, job_args=job_args, exe=exe,
                           job_name=job_name, job_dir=job_dir, overwrite=False)
