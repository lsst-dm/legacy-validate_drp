# LSST Data Management System
# Copyright 2016 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.


import numpy as np
import astropy.units as u

from lsst.verify import Measurement, Datum


def measurePA2(metric, pa1, pf1_thresh):
    """Measurement of PA2: millimag from median RMS (see PA1) of which
    PF1 of the samples can be found.

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        A PA2 `~lsst.verify.Metric` instance.
    pa1 : `lsst.verify.Measurement`
        A PA1 measurement instance.
    pf1_thresh : `astropy.units.Quantity`
        Quantity specifying the threshold at which to calculate PA2

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of PA2 and associated metadata.

    Notes
    -----
    The LSST Science Requirements Document (LPM-17) is commonly referred
    to as the SRD.  The SRD puts a limit that no more than PF1 % of difference
    will vary by more than PA2 millimag.  The design, minimum, and stretch
    goals are PF1 = (10, 20, 5) % at PA2 = (15, 15, 10) millimag following
    LPM-17 as of 2011-07-06, available at http://ls.st/LPM-17.
    """

    datums = {}
    datums['pf1_thresh'] = Datum(quantity=pf1_thresh, description="Threshold from the PF1 specification")

    # Use first random sample from original PA1 measurement
    magDiffs = pa1.extras['magDiff'].quantity[0, :]

    pf1Percentile = 100.*u.percent - pf1_thresh
    return Measurement(metric, np.percentile(np.abs(magDiffs.value), pf1Percentile.value) * magDiffs.unit,
                       extras=datums)
