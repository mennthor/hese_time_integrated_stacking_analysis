# coding: utf-8

"""
Working directories for each branch.
By using the paths stored here consistently, we can set new working directories
for each branch, without interfering with previous results.

In a script use `from PATHS import PATHS` and eg. `local_path = PATHS.local`.
Get all available paths with `print(PATHS)`.
```
"""

import os as _os
import sys as _sys
from git import Repo as _Repo


def check_dir(dir, ask=False):
    """
    Checks if dirextory ``dir`` exists and ask for permission (if ``ask=True``)
    to continue if it does. If ``dir`` is not existing it is created.
    """
    if _os.path.isdir(dir):
        if ask:
            res = raw_input("'{}' already exists. ".format(dir) +
                            "\nAllow overwrites (y/n)? ")
            if not res.lower() in ("y", "yes"):
                print("Abort. Script has done nothing.")
                _sys.exit()
        print("Using exiting output directory '{}'.".format(dir))
    else:
        _os.makedirs(dir)
        print("Created output directory '{}'.".format(dir))


class _Paths(object):
    """
    Class to acces paths via it's attributes.
    Code adopted from scipy.optimize.OptimizeResult.
    """
    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
            try:
                return self._d[name]
            except KeyError:
                raise AttributeError(name)

    def __setattr__(self, name, val):
        if name != "_d":
            raise RuntimeError("PATHS is readonly.")
        else:
            # We could still overwrite _d, but who does that?
            super(_Paths, self).__setattr__(name, val)

    def __repr__(self):
        m = max(map(len, list(self._d.keys()))) + 1
        return '\n'.join([name.rjust(m) + ': ' + path
                          for name, path in self._d.items()])


# Insert the current branch name to automatically switch to a new work dir
_repo_path = _os.path.join("/home", "tmenne", "analysis",
                           "hese_time_independent_stacking_fit_index")
_repo_name = _os.path.basename(_repo_path)
_repo = _Repo(_repo_path)
_BRANCH_NAME = _repo.active_branch.name

_paths = {
    "repo": _repo_path,
    "local": _os.path.join(_repo_path, "out_" + _BRANCH_NAME),
    "data": _os.path.join("/data", "user", "tmenne", _repo_name,
                          "rawout_" + _BRANCH_NAME),
    "jobs": _os.path.join(_repo_path, "jobfiles_" + _BRANCH_NAME),
    "plots": _os.path.join(_repo_path, "plots_" + _BRANCH_NAME),
}

PATHS = _Paths(_paths)
