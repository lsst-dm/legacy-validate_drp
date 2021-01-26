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


def measureAFx(metric, amx, adx, adx_spec):
    r"""Measurement of AFx (x=1,2,3): The maximum fraction of astrometric
    distances which deviate by more than ADx milliarcsec (see AMx) (%).

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        AF1, AF2 or AF3 `~lsst.verify.Metric` instance.
    amx : `lsst.verify.Measurement`
        An AMx measurement, providing the median astrometric scatter in
        the annulus.
    adx : `lsst.verify.Measurement`
        An ADx measurement
    adx_spec : `lsst.verify.Spec`
        An instance of a `lsst.verify.Spec` containing the threshold against
        which to measure.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of AFx (x=1,2,3) and associated metadata.

    Notes
    -----
    This table below is provided ``validate_drp``\ 's :file:`metrics.yaml`.

    LPM-17 dated 2011-07-06

    Specification:
        The rms of the astrometric distance distribution for
        stellar pairs with separation of D arcmin (repeatability)
        will not exceed AMx milliarcsec (median distribution for a large number
        of sources). No more than AFx % of the sample will deviate by more than
        ADx milliarcsec from the median. AMx, AFx, and ADx are specified for
        D=5, 20 and 200 arcmin for x= 1, 2, and 3, in the same order (Table 18).

    The three selected characteristic distances reflect the size of an
    individual sensor, a raft, and the camera. The required median astrometric
    precision is driven by the desire to achieve a proper motion accuracy of
    0.2 mas/yr and parallax accuracy of 1.0 mas over the course of the survey.
    These two requirements correspond to relative astrometric precision for a
    single image of 10 mas (per coordinate).

    ========================= ====== ======= =======
    Astrometric Repeatability          Specification
    ------------------------- ----------------------
                       Metric Design Minimum Stretch
    ========================= ====== ======= =======
            AM1 (milliarcsec)     10      20       5
            AF1 (%)               10      20       5
            AD1 (milliarcsec)     20      40      10
            AM2 (milliarcsec)     10      20       5
            AF2 (%)               10      20       5
            AD2 (milliarcsec)     20      40      10
            AM3 (milliarcsec)     15      30      10
            AF3 (%)               10      20       5
            AD3 (milliarcsec)     30      50      20
    ========================= ====== ======= =======

    Table 18: The specifications for astrometric precision.
    The three blocks of values correspond to D=5, 20 and 200 arcmin,
    and to astrometric measurements performed in the r and i bands.
    """

    datums = {}
    datums['ADx'] = adx.datum
    if not np.isnan(amx.quantity):
        quantity = 100.*np.mean(amx.extras['rmsDistMas'].quantity > amx.quantity
                                + adx_spec.threshold)*u.Unit('percent')
    else:
        quantity = np.nan*u.Unit('percent')
    datums.update(amx.extras)
    return Measurement(metric, quantity, extras=datums)
