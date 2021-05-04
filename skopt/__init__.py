"""
Scikit-Optimize, or `skopt`, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. `skopt` is reusable
in many contexts and accessible.
"""
try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __SKOPT_SETUP__
except NameError:
    __SKOPT_SETUP__ = False


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.9.dev0"

if __SKOPT_SETUP__:
    import sys
    sys.stderr.write('Partial import of skopt during the build process.\n')
    # We are not importing the rest of scikit-optimize during the build
    # process, as it may not be compiled yet
else:
    import platform
    import struct
    from . import acquisition
    from . import benchmarks
    from . import callbacks
    from . import learning
    from . import optimizer

    from . import space
    from . import sampler
    from .optimizer import dummy_minimize
    from .optimizer import forest_minimize
    from .optimizer import gbrt_minimize
    from .optimizer import gp_minimize
    from .optimizer import ugp_minimize
    from .optimizer import Optimizer
    from .searchcv import BayesSearchCV
    from .space import Space
    from .utils import dump
    from .utils import expected_minimum
    from .utils import expected_minimum_random_sampling
    from .utils import load
    __all__ = (
        "acquisition",
        "benchmarks",
        "callbacks",
        "learning",
        "optimizer",
        "plots",
        "sampler",
        "space",
        "gp_minimize",
        "ugp_minimize",
        "dummy_minimize",
        "forest_minimize",
        "gbrt_minimize",
        "Optimizer",
        "dump",
        "load",
        "expected_minimum",
        "BayesSearchCV",
        "Space"
    )
    IS_PYPY = platform.python_implementation() == 'PyPy'
    _IS_32BIT = 8 * struct.calcsize("P") == 32
