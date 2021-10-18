import warnings

text = warnings.warn("pdspy v2.0.0 represents a major update to the pdspy code, and is not backwards compatible with the results of versions < 2.0.0. *Do not use v2.0.0 to work with results from earlier versions.* For more information, see pdspy.readthedocs.io.", stacklevel=2)

from . import constants
from . import dust
from . import gas
from . import imaging
from . import interferometry
from . import mcmc
from . import misc
from . import modeling
from . import plotting
from . import radmc3d
from . import spectroscopy
from . import stars
from . import statistics
from . import table
from . import utils
