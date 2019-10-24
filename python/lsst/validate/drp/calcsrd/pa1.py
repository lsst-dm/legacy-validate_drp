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

import math

import numpy as np
import random
import scipy.stats
import astropy.units as u

import lsst.pipe.base as pipeBase
from lsst.verify import Measurement, Datum


def measurePA1(metric, matchedDataset, filterName, numRandomShuffles=50):
    """Measurement of the PA1 metric: photometric repeatability of
    measurements across a set of observations.

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        A PA1 `~lsst.verify.Metric` instance.
    matchedDataset : `lsst.verify.Blob`
        This contains the spacially matched catalog to do the measurement.
    filterName : str
        filter name used in this measurement (e.g., `'r'`)
    numRandomShuffles : int
        Number of times to draw random pairs from the different observations.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of PA1 and associated metadata.

    See also
    --------
    calcPa1: Computes statistics of magnitudes differences of stars across
        multiple visits. This is the main computation function behind
        the PA1 measurement.
    """

    matches = matchedDataset.safeMatches
    magKey = matchedDataset.magKey
    results = calcPa1(matches, magKey, numRandomShuffles=numRandomShuffles)
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
    return Measurement(metric, results['PA1'], extras=datums)


def calcPa1(matches, magKey, numRandomShuffles=50):
    """Calculate the photometric repeatability of measurements across a set
    of randomly selected pairs of visits.

    Parameters
    ----------
    matches : `lsst.afw.table.GroupView`
        `~lsst.afw.table.GroupView` of stars matched between visits,
        from MultiMatch, provided by
        `lsst.validate.drp.matchreduce.build_matched_dataset`.
    magKey : `lsst.afw.table` schema key
        Magnitude column key in the ``groupView``.
        E.g., ``magKey = allMatches.schema.find("base_PsfFlux_mag").key``
        where ``allMatches`` is the result of
        `lsst.afw.table.MultiMatch.finish()`.
    numRandomShuffles : int
        Number of times to draw random pairs from the different observations.

    Returns
    -------
    statistics : `dict`
        Statistics to compute PA1. Fields are:

        - ``PA1``: scalar `~astropy.unit.Quantity` of mean ``iqr``. This is
          formally the PA1 metric measurement.
        - ``rms``: `~astropy.unit.Quantity` array in mmag of photometric
          repeatability RMS across ``numRandomShuffles``.
          Shape: ``(nRandomSamples,)``.
        - ``iqr``: `~astropy.unit.Quantity` array in mmag of inter-quartile
          range of photometric repeatability distribution.
          Shape: ``(nRandomSamples,)``.
        - ``magDiff``: `~astropy.unit.Quantity` array of magnitude differences
          between pairs of stars. Shape: ``(nRandomSamples, nMatches)``.
        - ``magMean``: `~astropy.unit.Quantity` array of mean magnitudes of
          each pair of stars. Shape: ``(nRandomSamples, nMatches)``.

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
    as of 2011-07-06, available at http://ls.st/LPM-17.

    This present routine calculates this quantity in two different ways:

    1. RMS
    2. interquartile range (IQR)

    **The PA1 scalar measurement is the median of the IQR.**

    This function also returns additional quantities of interest:

    - the pair differences of observations of stars,
    - the mean magnitude of each star

    While the SRD specifies that we should just compute the RMS directly, the
    current filter doesn't screen out variable stars as carefully as the SRD
    specifies, so using a more robust estimator like the IQR allows us to
    reject some outliers.  However, the IRQ is also less sensitive some
    realistic sources of scatter such as bad zero points, that the metric
    should include.

    Examples
    --------
    Normally ``calcPa1`` is called by `measurePA1`, using data from
    `lsst.validate.drp.matchreduce.build_matched_dataset`. Here's an
    example of how to call ``calcPa1`` directly given a Butler output
    repository:

    >>> import lsst.daf.persistence as dafPersist
    >>> from lsst.afw.table import SourceCatalog, SchemaMapper, Field
    >>> from lsst.afw.table import MultiMatch, SourceRecord, GroupView
    >>> from lsst.validate.drp.calcsrd.pa1 import calcPa1
    >>> repo = "CFHT/output"
    >>> butler = dafPersist.Butler(repo)
    >>> dataset = 'src'
    >>> schema = butler.get(dataset + "_schema", immediate=True).schema
    >>> mmatch = MultiMatch(newSchema,
    >>>                     dataIdFormat={'visit': int, 'ccd': int},
    >>>                     radius=matchRadius,
    >>>                     RecordClass=SourceRecord)
    >>> for vId in visitDataIds:
    ...     cat = butler.get('src', vId)
    ...     mmatch.add(catalog=cat, dataId=vId)
    ...
    >>> matchCat = mmatch.finish()
    >>> allMatches = GroupView.build(matchCat)
    >>> allMatches
    >>> psfMagKey = allMatches.schema.find("base_PsfFlux_mag").key
    >>> pa1 = calcPa1(allMatches, psfMagKey)
    """
    pa1Samples = [calcPa1Sample(matches, magKey)
                  for n in range(numRandomShuffles)]

    rms = np.array([pa1.rms for pa1 in pa1Samples]) * u.mmag
    iqr = np.array([pa1.iqr for pa1 in pa1Samples]) * u.mmag
    magDiff = np.array([pa1.magDiffs for pa1 in pa1Samples]) * u.mmag
    magMean = np.array([pa1.magMean for pa1 in pa1Samples]) * u.mag
    pa1 = np.mean(iqr)
    return {'rms': rms, 'iqr': iqr, 'magDiff': magDiff, 'magMean': magMean,
            'PA1': pa1}


def calcPa1Sample(matches, magKey):
    """Compute one realization of PA1 by randomly sampling pairs of
    visits.

    Parameters
    ----------
    matches : `lsst.afw.table.GroupView`
        `~lsst.afw.table.GroupView` of stars matched between visits,
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
        Metrics of pairs of stars matched between two visits. Fields are:

        - ``rms``: scalar RMS of differences of stars observed in this
          randomly sampled pair of visits.
        - ``iqr``: scalar inter-quartile range (IQR) of differences of stars
          observed in a randomly sampled pair of visits.
        - ``magDiffs`: array, shape ``(nMatches,)``, of magnitude differences
          (mmag) for observed star across a randomly sampled pair of visits.
        - ``magMean``: array, shape ``(nMatches,)``, of mean magnitudes
          of stars observed across a randomly sampled pair of visits.

    See also
    --------
    calcPa1 : A wrapper that repeatedly calls this function to build
        the PA1 measurement.

    Examples
    --------
    Normally ``calcPa1`` is called by `measurePA1`, using data from
    `lsst.validate.drp.matchreduce.build_matched_dataset`. Here's an
    example of how to call ``calcPa1Sample`` directly given a Butler output
    repository:
    """
    magDiffs = matches.aggregate(getRandomDiffRmsInMmags, field=magKey)
    magMean = matches.aggregate(np.mean, field=magKey)
    rmsPA1, iqrPA1 = computeWidths(magDiffs)
    return pipeBase.Struct(rms=rmsPA1, iqr=iqrPA1,
                           magDiffs=magDiffs, magMean=magMean,)


def getRandomDiffRmsInMmags(array):
    """Calculate the RMS difference in mmag between a random pairing of
    visits of a star.

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
    magnitude differences for the same star measured on two visits
    (using a median over the magDiffs to reject outliers).
    Because we have N>=2 measurements for each star, we select a random
    pair of visits for each star.  We divide each difference by sqrt(2)
    to obtain RMS about the (unknown) mean magnitude,
    instead of obtaining just the RMS difference.

    See Also
    --------
    getRandomDiff : Get the difference

    Examples
    --------
    >>> mag = [24.2, 25.5]
    >>> rms = getRandomDiffRmsInMmags(mag)
    >>> print(rms)
    212.132034
    """
    # For scalars, math.sqrt is several times faster than numpy.sqrt.
    return (1000/math.sqrt(2)) * getRandomDiff(array)


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
    rmsSigma = math.sqrt(np.mean(array**2))
    iqrSigma = np.subtract.reduce(np.percentile(array, [75, 25])) / (scipy.stats.norm.ppf(0.75)*2)
    return rmsSigma, iqrSigma
