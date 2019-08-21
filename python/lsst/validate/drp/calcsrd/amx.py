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

from ..util import (averageRaFromCat, averageDecFromCat,
                    sphDist)


def measureAMx(metric, matchedDataset, D, width=2., magRange=None, verbose=False):
    r"""Measurement of AMx (x=1,2,3): The maximum rms of the astrometric
    distance distribution for stellar pairs with separations of D arcmin
    (repeatability).

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        An AM1, AM2 or AM3 `~lsst.verify.Metric` instance.
    matchedDataset : lsst.verify.Blob
        Contains the spacially matched dataset for the measurement
    D : `astropy.units.Quantity`
        Radial distance of the annulus in arcmin
    width : `float` or `astropy.units.Quantity`, optional
        Width around fiducial distance to include. [arcmin]
    magRange : 2-element `list`, `tuple`, or `numpy.ndarray`, optional
        brighter, fainter limits of the magnitude range to include.
        Default: ``[17.5, 21.5]`` mag.
    verbose : `bool`, optional
        Output additional information on the analysis steps.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of AMx (x=1,2,3) and associated metadata.

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

    matches = matchedDataset.safeMatches

    datums = {}

    # Measurement Parameters
    datums['D'] = Datum(quantity=D, label="Distance", description="Radial distance of annulus (arcmin)")

    if not isinstance(width, u.Quantity):
        width = width * u.arcmin
    datums['Width'] = Datum(quantity=width, label='Width', description='Width of annulus')
    if magRange is None:
        magRange = np.array([17.0, 21.5]) * u.mag
    else:
        assert len(magRange) == 2
        if not isinstance(magRange, u.Quantity):
            magRange = np.array(magRange) * u.mag
    datums['magRange'] = Datum(quantity=magRange, description='Stellar magnitude selection range.')

    annulus = D + (width/2)*np.array([-1, +1])
    datums['annulus'] = Datum(quantity=annulus, label='annulus radii',
                              description='Inner and outer radii of selection annulus.')

    # Register measurement extras
    rmsDistances = calcRmsDistances(
        matches,
        annulus,
        magRange=magRange,
        verbose=verbose)

    if len(rmsDistances) == 0:
        # Should be a proper log message
        print('No stars found that are {0:.1f}--{1:.1f} apart.'.format(
              annulus[0], annulus[1]))
        datums['rmsDistMas'] = Datum(quantity=None, label='RMS')
        quantity = np.nan * u.marcsec
    else:
        datums['rmsDistMas'] = Datum(quantity=rmsDistances.to(u.marcsec), label='RMS')
        quantity = np.median(rmsDistances.to(u.marcsec))

    return Measurement(metric.name, quantity, extras=datums)


def calcRmsDistances(groupView, annulus, magRange, verbose=False):
    """Calculate the RMS distance of a set of matched objects over visits.

    Parameters
    ----------
    groupView : lsst.afw.table.GroupView
        GroupView object of matched observations from MultiMatch.
    annulus : length-2 `astropy.units.Quantity`
        Distance range (i.e., arcmin) in which to compare objects.
        E.g., `annulus=np.array([19, 21]) * u.arcmin` would consider all
        objects separated from each other between 19 and 21 arcminutes.
    magRange : length-2 `astropy.units.Quantity`
        Magnitude range from which to select objects.
    verbose : bool, optional
        Output additional information on the analysis steps.

    Returns
    -------
    rmsDistances : `astropy.units.Quantity`
        RMS angular separations of a set of matched objects over visits.
    """

    # First we make a list of the keys that we want the fields for
    importantKeys = [groupView.schema.find(name).key for
                     name in ['id', 'coord_ra', 'coord_dec',
                              'object', 'visit', 'base_PsfFlux_mag']]

    minMag, maxMag = magRange.to(u.mag).value

    def magInRange(cat):
        mag = cat.get('base_PsfFlux_mag')
        w, = np.where(np.isfinite(mag))
        medianMag = np.median(mag[w])
        return minMag <= medianMag and medianMag < maxMag

    groupViewInMagRange = groupView.where(magInRange)

    # List of lists of id, importantValue
    matchKeyOutput = [obj.get(key)
                      for key in importantKeys
                      for obj in groupViewInMagRange.groups]

    jump = len(groupViewInMagRange)

    ra = matchKeyOutput[1*jump:2*jump]
    dec = matchKeyOutput[2*jump:3*jump]
    visit = matchKeyOutput[4*jump:5*jump]

    # Calculate the mean position of each object from its constituent visits
    # `aggregate` calulates a quantity for each object in the groupView.
    meanRa = groupViewInMagRange.aggregate(averageRaFromCat)
    meanDec = groupViewInMagRange.aggregate(averageDecFromCat)

    annulusRadians = arcminToRadians(annulus.to(u.arcmin).value)

    rmsDistances = list()
    for obj1, (ra1, dec1, visit1) in enumerate(zip(meanRa, meanDec, visit)):
        dist = sphDist(ra1, dec1, meanRa[obj1+1:], meanDec[obj1+1:])
        objectsInAnnulus, = np.where((annulusRadians[0] <= dist) &
                                     (dist < annulusRadians[1]))
        for obj2 in objectsInAnnulus:
            distances = matchVisitComputeDistance(
                visit[obj1], ra[obj1], dec[obj1],
                visit[obj2], ra[obj2], dec[obj2])
            if not distances:
                if verbose:
                    print("No matching visits found for objs: %d and %d" %
                          (obj1, obj2))
                continue

            finiteEntries, = np.where(np.isfinite(distances))
            # Need at least 2 distances to get a finite sample stdev
            if len(finiteEntries) > 1:
                # ddof=1 to get sample standard deviation (e.g., 1/(n-1))
                rmsDist = np.std(np.array(distances)[finiteEntries], ddof=1)
                rmsDistances.append(rmsDist)

    # return quantity
    rmsDistances = np.array(rmsDistances) * u.radian
    return rmsDistances


def matchVisitComputeDistance(visit_obj1, ra_obj1, dec_obj1,
                              visit_obj2, ra_obj2, dec_obj2):
    """Calculate obj1-obj2 distance for each visit in which both objects are seen.

    For each visit shared between visit_obj1 and visit_obj2,
    calculate the spherical distance between the obj1 and obj2.

    visit_obj1 and visit_obj2 are assumed to be unsorted.

    Parameters
    ----------
    visit_obj1 : scalar, list, or numpy.array of int or str
        List of visits for object 1.
    ra_obj1 : scalar, list, or numpy.array of float
        List of RA in each visit for object 1.  [radians]
    dec_obj1 : scalar, list or numpy.array of float
        List of Dec in each visit for object 1. [radians]
    visit_obj2 : list or numpy.array of int or str
        List of visits for object 2.
    ra_obj2 : list or numpy.array of float
        List of RA in each visit for object 2.  [radians]
    dec_obj2 : list or numpy.array of float
        List of Dec in each visit for object 2.  [radians]

    Results
    -------
    list of float
        spherical distances (in radians) for matching visits.
    """
    distances = []
    visit_obj1_idx = np.argsort(visit_obj1)
    visit_obj2_idx = np.argsort(visit_obj2)
    j_raw = 0
    j = visit_obj2_idx[j_raw]
    for i in visit_obj1_idx:
        while (visit_obj2[j] < visit_obj1[i]) and (j_raw < len(visit_obj2_idx)-1):
            j_raw += 1
            j = visit_obj2_idx[j_raw]
        if visit_obj2[j] == visit_obj1[i]:
            if np.isfinite([ra_obj1[i], dec_obj1[i],
                            ra_obj2[j], dec_obj2[j]]).all():
                distances.append(sphDist(ra_obj1[i], dec_obj1[i],
                                         ra_obj2[j], dec_obj2[j]))
    return distances


def radiansToMilliarcsec(rad):
    return np.rad2deg(rad)*3600*1000


def arcminToRadians(arcmin):
    return np.deg2rad(arcmin/60)
