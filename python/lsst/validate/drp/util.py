# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
"""Miscellaneous functions to support lsst.validate.drp."""

import os

import numpy as np

import yaml

import lsst.daf.persistence as dafPersist
import lsst.pipe.base as pipeBase
import lsst.geom as geom


def ellipticity_from_cat(cat, slot_shape='slot_Shape'):
    """Calculate the ellipticity of the Shapes in a catalog from the 2nd moments.

    Parameters
    ----------
    cat : `lsst.afw.table.BaseCatalog`
       A catalog with 'slot_Shape' defined and '_xx', '_xy', '_yy'
       entries for the target of 'slot_Shape'.
       E.g., 'slot_shape' defined as 'base_SdssShape'
       And 'base_SdssShape_xx', 'base_SdssShape_xy', 'base_SdssShape_yy' defined.
    slot_shape : str, optional
       Specify what slot shape requested.  Intended use is to get the PSF shape
       estimates by specifying 'slot_shape=slot_PsfShape'
       instead of the default 'slot_shape=slot_Shape'.

    Returns
    -------
    e, e1, e2 : complex, float, float
        Complex ellipticity, real part, imaginary part
    """
    i_xx, i_xy, i_yy = cat.get(slot_shape+'_xx'), cat.get(slot_shape+'_xy'), cat.get(slot_shape+'_yy')
    return ellipticity(i_xx, i_xy, i_yy)


def ellipticity_from_shape(shape):
    """Calculate the ellipticty of shape from its moments.

    Parameters
    ----------
    shape : `lsst.afw.geom.ellipses.Quadrupole`
        The LSST generic shape object returned by psf.computeShape()
        or source.getShape() for a specific source.
        Imeplementation: just needs to have .getIxx, .getIxy, .getIyy methods
        that each return a float describing the respective second moments.

    Returns
    -------
    e, e1, e2 : complex, float, float
        Complex ellipticity, real part, imaginary part
    """
    i_xx, i_xy, i_yy = shape.getIxx(), shape.getIxy(), shape.getIyy()
    return ellipticity(i_xx, i_xy, i_yy)


def ellipticity(i_xx, i_xy, i_yy):
    """Calculate ellipticity from second moments.

    Parameters
    ----------
    i_xx : float or `numpy.array`
    i_xy : float or `numpy.array`
    i_yy : float or `numpy.array`

    Returns
    -------
    e, e1, e2 : (float, float, float) or (numpy.array, numpy.array, numpy.array)
        Complex ellipticity, real component, imaginary component
    """
    e = (i_xx - i_yy + 2j*i_xy) / (i_xx + i_yy)
    e1 = np.real(e)
    e2 = np.imag(e)
    return e, e1, e2


def averageRaDec(ra, dec):
    """Calculate average RA, Dec from input lists using spherical geometry.

    Parameters
    ----------
    ra : `list` [`float`]
        RA in [radians]
    dec : `list` [`float`]
        Dec in [radians]

    Returns
    -------
    float, float
       meanRa, meanDec -- Tuple of average RA, Dec [radians]
    """
    assert(len(ra) == len(dec))

    angleRa = [geom.Angle(r, geom.radians) for r in ra]
    angleDec = [geom.Angle(d, geom.radians) for d in dec]
    coords = [geom.SpherePoint(ar, ad, geom.radians) for (ar, ad) in zip(angleRa, angleDec)]

    meanRa, meanDec = geom.averageSpherePoint(coords)

    return meanRa.asRadians(), meanDec.asRadians()


def averageRaDecFromCat(cat):
    """Calculate the average right ascension and declination from a catalog.

    Convenience wrapper around averageRaDec

    Parameters
    ----------
    cat : collection
         Object with .get method for 'coord_ra', 'coord_dec' that returns radians.

    Returns
    -------
    ra_mean : `float`
        Mean RA in radians.
    dec_mean : `float`
        Mean Dec in radians.
    """
    return averageRaDec(cat.get('coord_ra'), cat.get('coord_dec'))


def positionRms(ra_mean, dec_mean, ra, dec):
    """Calculate the RMS between an array of coordinates and a reference (mean) position.

    Parameters
    ----------
    ra_mean : `float`
        Mean RA in radians.
    dec_mean : `float`
        Mean Dec in radians.
    ra : `numpy.array` [`float`]
        Array of RA in radians.
    dec : `numpy.array` [`float`]
        Array of Dec in radians.

    Returns
    -------
    pos_rms : `float`
        RMS scatter of positions in milliarcseconds.

    Notes
    -----
    The RMS of a single-element array will be returned as 0.
    The RMS of an empty array will be returned as NaN.
    """
    separations = sphDist(ra_mean, dec_mean, ra, dec)
    # Note we don't want `np.std` of separations, which would give us the
    #   std around the average of separations.
    # We've already taken out the average,
    #   so we want the sqrt of the mean of the squares.
    pos_rms_rad = np.sqrt(np.mean(separations**2))  # radians
    pos_rms_mas = geom.radToMas(pos_rms_rad)  # milliarcsec

    return pos_rms_mas


def positionRmsFromCat(cat):
    """Calculate the RMS for RA, Dec for a set of observations an object.

    Parameters
    ----------
    cat : collection
         Object with .get method for 'coord_ra', 'coord_dec' that returns radians.

    Returns
    -------
    pos_rms : `float`
        RMS scatter of positions in milliarcseconds.
    """
    ra_avg, dec_avg = averageRaDecFromCat(cat)
    ra, dec = cat.get('coord_ra'), cat.get('coord_dec')
    return positionRms(ra_avg, dec_avg, ra, dec)


def sphDist(ra_mean, dec_mean, ra, dec):
    """Calculate distance on the surface of a unit sphere.

    Parameters
    ----------
    ra_mean : `float`
        Mean RA in radians.
    dec_mean : `float`
        Mean Dec in radians.
    ra : `numpy.array` [`float`]
        Array of RA in radians.
    dec : `numpy.array` [`float`]
        Array of Dec in radians.

    Notes
    -----
    Uses the Haversine formula to preserve accuracy at small angles.

    Law of cosines approach doesn't work well for the typically very small
    differences that we're looking at here.
    """
    # Haversine
    dra = ra - ra_mean
    ddec = dec - dec_mean
    a = np.square(np.sin(ddec/2)) + \
        np.cos(dec_mean)*np.cos(dec)*np.square(np.sin(dra/2))
    dist = 2 * np.arcsin(np.sqrt(a))

    # This is what the law of cosines would look like
    #    dist = np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))

    # This will also work, but must run separately for each element
    # whereas the numpy version will run on either scalars or arrays:
    #   sp1 = geom.SpherePoint(ra1, dec1, geom.radians)
    #   sp2 = geom.SpherePoint(ra2, dec2, geom.radians)
    #   return sp1.separation(sp2).asRadians()

    return dist


def averageRaFromCat(cat):
    """Compute the average right ascension from a catalog of measurements.

    This function is used as an aggregate function to extract just RA
    from lsst.validate.drp.matchreduce.build_matched_dataset

    The actual computation involves both RA and Dec.

    The intent is to use this for a set of measurements of the same source
    but that's neither enforced nor required.

    Parameters
    ----------
    cat : collection
         Object with .get method for 'coord_ra', 'coord_dec' that returns radians.

    Returns
    -------
    ra_mean : `float`
        Mean RA in radians.
    """
    meanRa, meanDec = averageRaDecFromCat(cat)
    return meanRa


def averageDecFromCat(cat):
    """Compute the average declination from a catalog of measurements.

    This function is used as an aggregate function to extract just declination
    from lsst.validate.drp.matchreduce.build_matched_dataset

    The actual computation involves both RA and Dec.

    The intent is to use this for a set of measurements of the same source
    but that's neither enforced nor required.

    Parameters
    ----------
    cat : collection
         Object with .get method for 'coord_ra', 'coord_dec' that returns radians.

    Returns
    -------
    dec_mean : `float`
        Mean Dec in radians.
    """
    meanRa, meanDec = averageRaDecFromCat(cat)
    return meanDec


def medianEllipticityResidualsFromCat(cat):
    """Compute the median ellipticty residuals from a catalog of measurements.

    This function is used as an aggregate function to extract just declination
    from lsst.validate.drp.matchreduce.build_matched_dataset

    The intent is to use this for a set of measurements of the same source
    but that's neither enforced nor required.

    Parameters
    ----------
    cat : collection
         Object with .get method for 'e1', 'e2' that returns radians.

    Returns
    -------
    e1_median : `float`
        Median real ellipticity residual.
    e2_median : `float`
        Median imaginary ellipticity residual.
    """
    e1_median = np.median(cat.get('e1') - cat.get('psf_e1'))
    e2_median = np.median(cat.get('e2') - cat.get('psf_e2'))
    return e1_median, e2_median


def medianEllipticity1ResidualsFromCat(cat):
    """Compute the median real ellipticty residuals from a catalog of measurements.

    Parameters
    ----------
    cat : collection
         Object with .get method for 'e1', 'psf_e1' that returns radians.

    Returns
    -------
    e1_median : `float`
        Median imaginary ellipticity residual.
    """
    e1_median = np.median(cat.get('e1') - cat.get('psf_e1'))
    return e1_median


def medianEllipticity2ResidualsFromCat(cat):
    """Compute the median imaginary ellipticty residuals from a catalog of measurements.

    Parameters
    ----------
    cat : collection
         Object with .get method for 'e2', 'psf_e2' that returns radians.

    Returns
    -------
    e2_median : `float`
        Median imaginary ellipticity residual.
    """
    e2_median = np.median(cat.get('e2') - cat.get('psf_e2'))
    return e2_median


def getCcdKeyName(dataId):
    """Return the key in a dataId that's referring to the CCD or moral equivalent.

    Parameters
    ----------
    dataId : `dict`
        A dictionary that will be searched for a key that matches
        an entry in the hardcoded list of possible names for the CCD field.

    Returns
    -------
    name : `str`
        The name of the key.

    Notes
    -----
    Motivation: Different camera mappings use different keys to indicate
      the different amps/ccds in the same exposure.  This function looks
      through the reference dataId to locate a field that could be the one.
    """
    possibleCcdFieldNames = ['detector', 'ccd', 'ccdnum', 'camcol', 'sensor']

    for name in possibleCcdFieldNames:
        if name in dataId:
            return name
    else:
        return 'ccd'


def raftSensorToInt(visitId):
    """Construct an int that encodes raft, sensor coordinates.

    Parameters
    ----------
    visitId : `dict`
        A dictionary containing raft and sensor keys.

    Returns
    -------
    id : `int`
        The integer id of the raft/sensor.

    Examples
    --------
    >>> vId = {'filter': 'y', 'raft': '2,2', 'sensor': '1,2', 'visit': 307}
    >>> raftSensorToInt(vId)
    2212
    """
    def pair_to_int(tuple_string):
        x, y = tuple_string.split(',')
        return 10 * int(x) + 1 * int(y)

    raft_int = pair_to_int(visitId['raft'])
    sensor_int = pair_to_int(visitId['sensor'])
    return 100*raft_int + sensor_int


def repoNameToPrefix(repo):
    """Generate a base prefix for plots based on the repo name.

    Parameters
    ----------
    repo : `str`
        The repo path.

    Returns
    -------
    repo_base : `str`
        The base prefix for the repo.

    Examples
    --------
    >>> repoNameToPrefix('a/b/c')
    'a_b_c'
    >>> repoNameToPrefix('/bar/foo/')
    'bar_foo'
    >>> repoNameToPrefix('CFHT/output')
    'CFHT_output'
    >>> repoNameToPrefix('./CFHT/output')
    'CFHT_output'
    >>> repoNameToPrefix('.a/CFHT/output')
    'a_CFHT_output'
    >>> repoNameToPrefix('bar/foo.json')
    'bar_foo'
    """

    repo_base, ext = os.path.splitext(repo)
    return repo_base.lstrip('.').strip(os.sep).replace(os.sep, "_")


def discoverDataIds(repo, **kwargs):
    """Retrieve a list of all dataIds in a repo.

    Parameters
    ----------
    repo : `str`
        Path of a repository with 'src' entries.

    Returns
    -------
    dataIds : `list`
        dataIds in the butler that exist.

    Notes
    -----
    May consider making this an iterator if large N becomes important.
    However, will likely need to know things like, "all unique filters"
    of a data set anyway, so would need to go through chain at least once.
    """
    butler = dafPersist.Butler(repo)
    thisSubset = butler.subset(datasetType='src', **kwargs)
    # This totally works, but would be better to do this as a TaskRunner?
    dataIds = [dr.dataId for dr in thisSubset
               if dr.datasetExists(datasetType='src') and dr.datasetExists(datasetType='calexp')]
    # Make sure we have the filter information
    for dId in dataIds:
        response = butler.queryMetadata(datasetType='src', format=['filter'], dataId=dId)
        filterForThisDataId = response[0]
        dId['filter'] = filterForThisDataId

    return dataIds


def loadParameters(configFile):
    """Load configuration parameters from a yaml file.

    Parameters
    ----------
    configFile : `str`
        YAML file that stores visit, filter, ccd,
        good_mag_limit, medianAstromscatterRef, medianPhotoscatterRef, matchRef
        and other parameters

    Returns
    -------
    pipeStruct: `lsst.pipe.base.Struct`
        Struct with configuration parameters.
    """
    with open(configFile, mode='r') as stream:
        data = yaml.safe_load(stream)

    return pipeBase.Struct(**data)


def loadDataIdsAndParameters(configFile):
    """Load data IDs, magnitude range, and expected metrics from a yaml file.

    Parameters
    ----------
    configFile : `str`
        YAML file that stores visit, filter, ccd,
        and additional configuration parameters such as
        brightSnr, medianAstromscatterRef, medianPhotoscatterRef, matchRef

    Returns
    -------
    pipeStruct: `lsst.pipe.base.Struct`
        Struct with attributes of dataIds - dict and configuration parameters.
    """
    parameters = loadParameters(configFile).getDict()

    ccdKeyName = getCcdKeyName(parameters)
    try:
        dataIds = constructDataIds(parameters['filter'], parameters['visits'],
                                   parameters[ccdKeyName], ccdKeyName)
    except KeyError:
        # If the above parameters are not in the `parameters` dict,
        # presumably because they were not in the configFile
        # then we return no dataIds.
        dataIds = []

    return pipeBase.Struct(dataIds=dataIds, **parameters)


def constructDataIds(filters, visits, ccds, ccdKeyName='ccd'):
    """Returns a list of dataIds consisting of every combination of visit & ccd for each filter.

    Parameters
    ----------
    filters : `str` or `list` [`str`]
        If str, will be interpreted as one filter to be applied to all visits.
    visits : `list` [`int`]
    ccds : `list` [`int`]
    ccdKeyName : `str`, optional
        Name to distinguish different parts of a focal plane.
        Generally 'ccd', but might be 'ccdnum', or 'amp', or 'ccdamp'.
        Refer to your `obs_*/policy/*Mapper.paf`.

    Returns
    -------
    dataIds : `list`
        dataIDs suitable to be used with the LSST Butler.

    Examples
    --------
    >>> dataIds = constructDataIds('r', [100, 200], [10, 11, 12])
    >>> for dataId in dataIds: print(dataId)
    {'filter': 'r', 'visit': 100, 'ccd': 10}
    {'filter': 'r', 'visit': 100, 'ccd': 11}
    {'filter': 'r', 'visit': 100, 'ccd': 12}
    {'filter': 'r', 'visit': 200, 'ccd': 10}
    {'filter': 'r', 'visit': 200, 'ccd': 11}
    {'filter': 'r', 'visit': 200, 'ccd': 12}
    """
    if isinstance(filters, str):
        filters = [filters for _ in visits]

    assert len(filters) == len(visits)
    dataIds = [{'filter': f, 'visit': v, ccdKeyName: c}
               for (f, v) in zip(filters, visits)
               for c in ccds]

    return dataIds


def loadRunList(configFile):
    """Load run list from a YAML file.

    Parameters
    ----------
    configFile : `str`
        YAML file that stores visit, filter, ccd,

    Returns
    -------
    runList : `list`
        run list lines.

    Examples
    --------
    An example YAML file would include entries of (for some CFHT data)
        visits: [849375, 850587]
        filter: 'r'
        ccd: [12, 13, 14, 21, 22, 23]
    or (for some DECam data)
        visits: [176837, 176846]
        filter: 'z'
        ccdnum: [10, 11, 12, 13, 14, 15, 16, 17, 18]

    Note 'ccd' for CFHT and 'ccdnum' for DECam.  These entries will be used to build
    dataIds, so these fields should be as the camera mapping defines them.

    `visits` and `ccd` (or `ccdnum`) must be lists, even if there's only one element.
    """
    stream = open(configFile, mode='r')
    data = yaml.safe_load(stream)

    ccdKeyName = getCcdKeyName(data)
    runList = constructRunList(data['visits'], data[ccdKeyName], ccdKeyName=ccdKeyName)

    return runList


def constructRunList(visits, ccds, ccdKeyName='ccd'):
    """Construct a comprehensive runList for processCcd.py.

    Parameters
    ----------
    visits : `list` of `int`
        The desired visits.
    ccds : `list` of `int`
        The desired ccds.

    Returns
    -------
    `list`
        list of strings suitable to be used with the LSST Butler.

    Examples
    --------
    >>> runList = constructRunList([100, 200], [10, 11, 12])
    >>> print(runList)
    ['--id visit=100 ccd=10^11^12', '--id visit=200 ccd=10^11^12']
    >>> runList = constructRunList([100, 200], [10, 11, 12], ccdKeyName='ccdnum')
    >>> print(runList)
    ['--id visit=100 ccdnum=10^11^12', '--id visit=200 ccdnum=10^11^12']

    Notes
    -----
    The LSST parsing convention is to use '^' as list separators
    for arguments to `--id`.  While surprising, this convention
    allows for CCD names to include ','.  E.g., 'R1,2'.
    Currently ignores `filter` because `visit` should be unique w.r.t filter.
    """
    runList = ["--id visit=%d %s=%s" % (v, ccdKeyName, "^".join([str(c) for c in ccds]))
               for v in visits]

    return runList
