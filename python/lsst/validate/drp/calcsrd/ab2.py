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

from lsst.verify import Measurement


"""
- Must require that the input be matched to a reference frame in r-band.
- Also require that the other frames are *not* r-band.
- Do the other frames all need to be the *same* band?
"""


def measureAB2(metric, ab1, abf1_spec):
    r"""Measurement of AB2: Maximum deviation of ABF1% of sources from the
        mean of AB1 measurements.

#    Measurement of ABF1: percentage of sources that may deviate by AB2 mas from the mean.

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        An AB2 `~lsst.verify.Metric` instance.
    matchedDataset : lsst.verify.Blob
        Contains the spacially matched dataset for the measurement
    rBandRefVisit : scalar int or str
        Visit ID for the visit to be used as the reference r-band visit.
    magRange : 2-element `list`, `tuple`, or `numpy.ndarray`, optional
        brighter, fainter limits of the magnitude range to include.
        Default: ``[17.5, 21.5]`` mag.
    verbose : `bool`, optional
        Output additional information on the analysis steps.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of AB2 and associated metadata.

    Notes
    -----
    This table below is provided ``validate_drp``\ 's :file:`metrics.yaml`.

    LPM-17 dated 2011-07-06

    Specification:
        The astrometric reference frame for an image obtained in a band other
        than r will be mapped to the corresponding r band image such that the
        rms of the distance distribution between the positions on the two
        frames will not exceed AB1 milliarcsec (for a large number of bright
        sources). No more than ABF1 % of the measurements will deviate by more
        than AB2 milliarcsec from the mean. The dependence of the distance
        between the positions on the two frames on the source color and
        observing conditions will be explicitly included in the astrometric
        model (Table 19).

    The requirements on the band-to-band astrometric transformation accuracy
    are driven by the detection of moving objects, de-blending of complex
    sources, and astrometric accuracy for sources detected in a single-band
    (e.g. high-redshift quasars detected only in the y band).

    ========================= ====== ======= =======
    Astrometric Repeatability          Specification
    ------------------------- ----------------------
                       Metric Design Minimum Stretch
    ========================= ====== ======= =======
            AB1 (milliarcsec)     10      20       5
            ABF1(%)               10      20       5
            AB2 (milliarcsec)     20      40      10
    ========================= ====== ======= =======

    Table 19: The requirements on the band-to-band astrometric transformation
    accuracy (per coordinate in arbitrary band, relative to the r band
    reference frame).
    """

#    import pdb
#    pdb.set_trace()

    if not np.isnan(ab1.quantity):
        # No more than ABF1 % of the measurements will deviate by more
        # than AB2 milliarcsec from the mean.
        # To compute AB2, use measured AB1 and spec for ABF1.
        meanSep = np.mean(ab1.extras['separations'].quantity.to(u.marcsec))
        ab2AtPercentile = np.percentile(
            np.abs(ab1.extras['separations'].quantity.to(u.marcsec) - meanSep),
            100. - abf1_spec.threshold.value) * u.marcsec
        quantity = ab2AtPercentile
        # # Note that the spec says deviations should be from the *mean*,
        # #   but we are using the median (ab1.quantity) instead.
        # quantity = ab2AtPercentile - ab1.quantity
    else:
        quantity = np.nan * ab1.quantity.unit
    return Measurement(metric, quantity, extras=ab1.extras)
