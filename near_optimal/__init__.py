
#
# ~~~ Fetch local package version from setup.py
from pkg_resources import get_distribution, DistributionNotFound
dist = get_distribution(__name__)    # ~~~ tbh I am surprised that __name__ is already defined
__version__ = dist.version
