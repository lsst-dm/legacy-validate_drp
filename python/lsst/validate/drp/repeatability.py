# LSST Data Management System
# Copyright 2008-2019 AURA/LSST.
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

import astropy.units as u
import math
import numpy as np
import random
from scipy.stats import norm

import lsst.pipe.base as pipeBase
from lsst.verify import Measurement, Datum

thousandDivSqrtTwo = 1000/math.sqrt(2)


def measurePhotRepeat(metric, filterName, *args, **kwargs):
    """Measurement of a photometric repeatability metric across a set of
    observations.

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        A Metric to construct a Measurement for.
    filterName : `str`
        Name of filter used for all observations.
    *args
        Additional arguments to pass to `calcPhotRepeat`.
    **kwargs
        Additional keyword arguments to pass to `calcPhotRepeat`.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of the repeatability and its associated metadata.

    See also
    --------
    calcPhotRepeat: Computes statistics of magnitudes differences of sources across
        multiple visits. This is the main computation function behind
        repeatability measurement.
    """
    results = calcPhotRepeat(*args, **kwargs)
    datums = {}
    datums['filter_name'] = Datum(filterName, label='filter',
                                  description='Name of filter for this measurement')
    datums['rms'] = Datum(results['rms'], label='RMS',
                          description='Photometric repeatability RMS of stellar pairs for '
                          'each random sampling')
    datums['iqr'] = Datum(results['iqr'], label='IQR',
                          description='Photometric repeatability IQR of stellar pairs for '
                          'each random sample')
    datums['magDiff'] = Datum(results['magDiff'], label='Delta mag',
                              description='Photometric repeatability differences magnitudes for '
                              'stellar pairs for each random sample')
    datums['magMean'] = Datum(results['magMean'], label='mag',
                              description='Mean magnitude of pairs of stellar sources matched '
                              'across visits, for each random sample.')
    return Measurement(metric, results['repeatability'], extras=datums)


def calcPhotRepeat(matches, magKey, numRandomShuffles=50):
    """Calculate the photometric repeatability of measurements across a set
    of randomly selected pairs of visits.

    Parameters
    ----------
    matches : `lsst.afw.table.GroupView`
        `~lsst.afw.table.GroupView` of sources matched between visits,
        from MultiMatch, provided by
        `lsst.validate.drp.matchreduce.build_matched_dataset`.
    magKey : `lsst.afw.table` schema key
        Magnitude column key in the ``groupView``.
        E.g., ``magKey = allMatches.schema.find("slot_ModelFlux_mag").key``
        where ``allMatches`` is the result of
        `lsst.afw.table.MultiMatch.finish()`.
    numRandomShuffles : int
        Number of times to draw random pairs from the different observations.

    Returns
    -------
    statistics : `dict`
        Statistics to compute model_phot_rep. Fields are:

        - ``model_phot_rep``: scalar `~astropy.unit.Quantity` of mean ``iqr``.
          This is formally the model_phot_rep metric measurement.
        - ``rms``: `~astropy.unit.Quantity` array in mmag of photometric
          repeatability RMS across ``numRandomShuffles``.
          Shape: ``(nRandomSamples,)``.
        - ``iqr``: `~astropy.unit.Quantity` array in mmag of inter-quartile
          range of photometric repeatability distribution.
          Shape: ``(nRandomSamples,)``.
        - ``magDiff``: `~astropy.unit.Quantity` array of magnitude differences
          between pairs of sources. Shape: ``(nRandomSamples, nMatches)``.
        - ``magMean``: `~astropy.unit.Quantity` array of mean magnitudes of
          each pair of sources. Shape: ``(nRandomSamples, nMatches)``.

    Notes
    -----
    We calculate differences for ``numRandomShuffles`` different random
    realizations of the measurement pairs, to provide some estimate of the
    uncertainty on our RMS estimates due to the random shuffling.  This
    estimate could be stated and calculated from a more formally derived
    motivation but in practice 50 should be sufficient.

    The LSST Science Requirements Document (LPM-17), or SRD, characterizes the
    photometric repeatability by putting a requirement on the median RMS of
    measurements of non-variable bright stars.  This quantity is PA1, with a
    design, minimum, and stretch goals of (5, 8, 3) millimag following LPM-17
    as of 2011-07-06, available at http://ls.st/LPM-17. model_phot_rep is a
    similar quantity measured for extended sources (almost entirely galaxies),
    for which no requirement currently exists in the SRD.

    This present routine calculates this quantity in two different ways:

    1. RMS
    2. interquartile range (IQR)

    **The repeatability scalar measurement is the median of the IQR.**

    This function also returns additional quantities of interest:

    - the pair differences of observations of sources,
    - the mean magnitude of each source

    Examples
    --------
    Normally ``calcPhotRepeat`` is called by `measurePhotRepeat`, using
    data from `lsst.validate.drp.matchreduce.build_matched_dataset`. Here's an
    example of how to call ``calcPhotRepeat`` directly given the Butler output
    repository generated by examples/runHscQuickTest.sh:

    >>> import lsst.daf.persistence as dafPersist
    >>> from lsst.afw.table import SourceCatalog, SchemaMapper, Field
    >>> from lsst.afw.table import MultiMatch, SourceRecord, GroupView
    >>> from lsst.validate.drp.repeatability import calcPhotRepeat
    >>> from lsst.validate.drp.util import discoverDataIds
    >>> import numpy as np
    >>> repo = 'HscQuick/output'
    >>> butler = dafPersist.Butler(repo)
    >>> dataset = 'src'
    >>> schema = butler.get(dataset + '_schema', immediate=True).schema
    >>> visitDataIds = discoverDataIds(repo)
    >>> mmatch = None
    >>> for vId in visitDataIds:
    ...     cat = butler.get('src', vId)
    ...     calib = butler.get('calexp_photoCalib', vId)
    ...     cat = calib.calibrateCatalog(cat, ['modelfit_CModel'])
    ...     if mmatch is None:
    ...         mmatch = MultiMatch(cat.schema,
    ...                             dataIdFormat={'visit': np.int32, 'ccd': np.int32},
    ...                             RecordClass=SourceRecord)
    ...     mmatch.add(catalog=cat, dataId=vId)
    ...
    >>> matchCat = mmatch.finish()
    >>> allMatches = GroupView.build(matchCat)
    >>> magKey = allMatches.schema.find('slot_ModelFlux_mag').key
    >>> def matchFilter(cat):
    >>>     if len(cat) < 2:
    >>>         return False
    >>>     return np.isfinite(cat.get(magKey)).all()
    >>> repeat = calcPhotRepeat(allMatches.where(matchFilter), magKey)

    """
    mprSamples = [calcPhotRepeatSample(matches, magKey)
                  for _ in range(numRandomShuffles)]

    rms = np.array([mpr.rms for mpr in mprSamples]) * u.mmag
    iqr = np.array([mpr.iqr for mpr in mprSamples]) * u.mmag
    magDiff = np.array([mpr.magDiffs for mpr in mprSamples]) * u.mmag
    magMean = np.array([mpr.magMean for mpr in mprSamples]) * u.mag
    repeat = np.mean(iqr)
    return {'rms': rms, 'iqr': iqr, 'magDiff': magDiff, 'magMean': magMean, 'repeatability': repeat}


def calcPhotRepeatSample(matches, magKey):
    """Compute one realization of repeatability by randomly sampling pairs of
    visits.

    Parameters
    ----------
    matches : `lsst.afw.table.GroupView`
        `~lsst.afw.table.GroupView` of sources matched between visits,
        from MultiMatch, provided by
        `lsst.validate.drp.matchreduce.build_matched_dataset`.
    magKey : `lsst.afw.table` schema key
        Magnitude column key in the ``groupView``.
        E.g., ``magKey = allMatches.schema.find("base_PsfFlux_mag").key``
        where ``allMatches`` is the result of
        `lsst.afw.table.MultiMatch.finish()`.

    Returns
    -------
    metrics : `lsst.pipe.base.Struct`
        Metrics of pairs of sources matched between two visits. Fields are:

        - ``rms``: scalar RMS of differences of sources observed in this
          randomly sampled pair of visits.
        - ``iqr``: scalar inter-quartile range (IQR) of differences of sources
          observed in a randomly sampled pair of visits.
        - ``magDiffs`: array, shape ``(nMatches,)``, of magnitude differences
          (mmag) for observed sources across a randomly sampled pair of visits.
        - ``magMean``: array, shape ``(nMatches,)``, of mean magnitudes
          of sources observed across a randomly sampled pair of visits.

    See also
    --------
    calcPhotRepeat : A wrapper that repeatedly calls this function to build
        the repeatability measurement.
    """
    magDiffs = matches.aggregate(getRandomDiffRmsInMmags, field=magKey)
    magMean = matches.aggregate(np.mean, field=magKey)
    rms, iqr = computeWidths(magDiffs)
    return pipeBase.Struct(rms=rms, iqr=iqr, magDiffs=magDiffs, magMean=magMean,)


def getRandomDiffRmsInMmags(array):
    """Calculate the RMS difference in mmag between a random pairing of
    visits of a source.

    Parameters
    ----------
    array : `list` or `numpy.ndarray`
        Magnitudes from which to select the pair [mag].

    Returns
    -------
    rmsMmags : `float`
        RMS difference in mmag from a random pair of visits.

    Notes
    -----
    The LSST SRD recommends computing repeatability from a histogram of
    magnitude differences for the same source measured on two visits
    (using a median over the magDiffs to reject outliers).
    Because we have N>=2 measurements for each source, we select a random
    pair of visits for each source.  We divide each difference by sqrt(2)
    to obtain the RMS about the (unknown) mean magnitude,
    instead of obtaining just the RMS difference.

    See Also
    --------
    getRandomDiff : Get the difference between two randomly selected elements of an array.

    Examples
    --------
    >>> mag = [24.2, 25.5]
    >>> rms = getRandomDiffRmsInMmags(mag)
    >>> print(rms)
    212.132034
    """
    return thousandDivSqrtTwo * getRandomDiff(array)


def getRandomDiff(array):
    """Get the difference between two randomly selected elements of an array.

    Parameters
    ----------
    array : `list` or `numpy.ndarray`
        Input array.

    Returns
    -------
    float or int
        Difference between two random elements of the array.
    """
    a, b = random.sample(range(len(array)), 2)
    return array[a] - array[b]


def computeWidths(array):
    """Compute the RMS and the scaled inter-quartile range of an array.

    Parameters
    ----------
    array : `list` or `numpy.ndarray`
        Array.

    Returns
    -------
    rms : `float`
        RMS
    iqr : `float`
        Scaled inter-quartile range (IQR, see *Notes*).

    Notes
    -----
    We estimate the width of the histogram in two ways:

    - using a simple RMS,
    - using the interquartile range (IQR)

    The IQR is scaled by the IQR/RMS ratio for a Gaussian such that it
    if the array is Gaussian distributed, then the scaled IQR = RMS.
    """
    # For scalars, math.sqrt is several times faster than numpy.sqrt.
    rmsSigma = math.sqrt(np.mean(array**2))
    iqrSigma = np.subtract.reduce(np.percentile(array, [75, 25])) / (norm.ppf(0.75)*2)
    return rmsSigma, iqrSigma
