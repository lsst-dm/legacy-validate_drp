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
import lsst.geom as geom

from lsst.verify import Measurement, Datum

from ..util import sphDist

"""
- Must require that the input be matched to a reference frame in r-band.
- Also require that the other frames are *not* r-band.
- Do the other frames all need to be the *same* band?
"""


def measureAB1(metric, matchedDataset, rBandRefVisit=None, magRange=None, verbose=False):
    r"""Measurement of AB1: RMS difference between separations measured in
        the r-band and those measured in any other filter.

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        An AB1, `~lsst.verify.Metric` instance.
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
        Measurement of AB1 and associated metadata.

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

    matches = matchedDataset.safeMatches

    datums = {}

    if magRange is None:
        magRange = np.array([17.0, 21.5]) * u.mag
    else:
        assert len(magRange) == 2
        if not isinstance(magRange, u.Quantity):
            magRange = np.array(magRange) * u.mag
    datums['magRange'] = Datum(quantity=magRange, description='Stellar magnitude selection range.')

#    import pdb
#    pdb.set_trace()

    # Register measurement extras
    rmsDistances, separations = calcRmsDistances(
        matches,
        rBandRefVisit=rBandRefVisit,
        magRange=magRange,
        verbose=verbose)

    if len(rmsDistances) == 0:
        # Should be a proper log message
        print('No matched stars found to calculate AB1.')
        datums['rmsDistMas'] = Datum(quantity=None, label='RMS')
        quantity = np.nan * u.marcsec
        datums['separations'] = Datum(quantity=None, label='separations')
    else:
        datums['rmsDistMas'] = Datum(quantity=rmsDistances, label='RMS')
        datums['separations'] = Datum(quantity=separations, label='separations')
        quantity = np.median(rmsDistances)

    return Measurement(metric.name, quantity, extras=datums)


def calcRmsDistances(groupView, rBandRefVisit, magRange, verbose=False):
    """Calculate the RMS distance of a set of matched objects over visits.

    Parameters
    ----------
    groupView : lsst.afw.table.GroupView
        GroupView object of matched observations from MultiMatch.
    magRange : length-2 `astropy.units.Quantity`
        Magnitude range from which to select objects.
    verbose : bool, optional
        Output additional information on the analysis steps.

    Returns
    -------
    rmsDistances : `astropy.units.Quantity`
        RMS angular separations of a set of matched objects over visits.
    separations : `astropy.units.Quantity`
        Angular separations of the set a matched objects.
    """

    minMag, maxMag = magRange.to(u.mag).value

    def magInRange(cat):
        mag = cat.get('base_PsfFlux_mag')
        w, = np.where(np.isfinite(mag))
        medianMag = np.median(mag[w])
        return minMag <= medianMag and medianMag < maxMag

    groupViewInMagRange = groupView.where(magInRange)

    # Get lists of the unique objects and visits:
    uniqObj = groupViewInMagRange.ids
    uniqVisits = set()
    for id in uniqObj:
        uniqVisits.update(set(groupViewInMagRange[id].get('visit')))

    uniqVisits = list(uniqVisits)

    # Only do the calculation if the object exists in the reference catalog:
    if rBandRefVisit is None:
        # For now, set the "default" visit to be the first in the list:
        rBandRefVisit = uniqVisits[0]
    else:
        assert type(rBandRefVisit) == int
        if not isinstance(rBandRefVisit, int):
            rBandRefVisit = int(rBandRefVisit)

    # Remove the reference visit from the set of visits:
    uniqVisits.remove(rBandRefVisit)

    rmsDistances = list()

    # Loop over visits, calculating the RMS for each:
    for vis in uniqVisits:

        distancesVisit = list()

        for obj in uniqObj:
            visMatch = np.where(groupViewInMagRange[obj].get('visit') == vis)
            refMatch = np.where(groupViewInMagRange[obj].get('visit') == rBandRefVisit)

            raObj = groupViewInMagRange[obj].get('coord_ra')
            decObj = groupViewInMagRange[obj].get('coord_dec')

            # Require it to have a match in both the reference and visit image:
            if visMatch[0] and refMatch[0]:
                distances = sphDist(raObj[refMatch], decObj[refMatch],
                                    raObj[visMatch], decObj[visMatch])

                distancesVisit.append(distances)

    # import pdb
    # pdb.set_trace()

    # Return an array with units
    distancesVisit = np.array(distancesVisit) * u.radian

    finiteEntries = np.where(np.isfinite(distancesVisit))[0]
    # Need at least 2 distances to get a finite sample stdev
    if len(finiteEntries) > 1:
        # Calculate the RMS of these offsets:
        # ddof=1 to get sample standard deviation (e.g., 1/(n-1))
        pos_rms_rad = np.std(np.array(distancesVisit)[finiteEntries], ddof=1)
        pos_rms_mas = geom.radToMas(pos_rms_rad)  # milliarcsec
        rmsDistances.append(pos_rms_mas)

    rmsDistances = rmsDistances * u.marcsec

    return rmsDistances, distancesVisit.to(u.marcsec)


def radiansToMilliarcsec(rad):
    return np.rad2deg(rad)*3600*1000


def arcminToRadians(arcmin):
    return np.deg2rad(arcmin/60)
