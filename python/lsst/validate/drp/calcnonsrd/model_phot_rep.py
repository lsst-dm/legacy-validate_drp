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

import numpy as np
import astropy.units as u

import lsst.pipe.base as pipeBase
from lsst.verify import Measurement, Datum

from lsst.validate.drp.calcsrd.pa1 import computeWidths, getRandomDiffRmsInMmags


def measureModelPhotRepeat(metric, matches, magKey, filterName, numRandomShuffles=50):
    """Measurement of the model_phot_rep metric: photometric repeatability of
    model measurements across a set of observations for stars and galaxies.

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        A model_phot_rep `~lsst.verify.Metric` instance.
    matchedDataset : `lsst.verify.Blob`
        This contains the spacially matched catalog to do the measurement.
    filterName : str
        filter name used in this measurement (e.g., `'r'`)
    numRandomShuffles : int
        Number of times to draw random pairs from the different observations.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of model_phot_rep and associated metadata.

    See also
    --------
    calcGalPhotRepeat: Computes statistics of magnitudes differences of
        galaxies across multiple visits. This is the main computation
        function behind the model_phot_rep measurement.
    """

    results = calcModelPhotRepeat(matches, magKey, numRandomShuffles=numRandomShuffles)
    datums = {}
    datums['filter_name'] = Datum(filterName, label='filter',
                                  description='Name of filter for this measurement')
    datums['rms'] = Datum(results['rms'], label='RMS',
                          description='Photometric repeatability RMS of galaxy pairs for '
                          'each random sampling')
    datums['iqr'] = Datum(results['iqr'], label='IQR',
                          description='Photometric repeatability IQR of galaxy pairs for '
                          'each random sample')
    datums['magDiff'] = Datum(results['magDiff'], label='Delta mag',
                              description='Photometric repeatability differences magnitudes for '
                              'galaxy pairs for each random sample')
    datums['magMean'] = Datum(results['magMean'], label='mag',
                              description='Mean magnitude of pairs of extended sources matched '
                              'across visits, for each random sample.')
    return Measurement(metric, results['model_phot_rep'], extras=datums)


def calcModelPhotRepeat(matches, magKey, numRandomShuffles=50):
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
          between pairs of galaxies. Shape: ``(nRandomSamples, nMatches)``.
        - ``magMean``: `~astropy.unit.Quantity` array of mean magnitudes of
          each pair of galaxies. Shape: ``(nRandomSamples, nMatches)``.

    Notes
    -----
    We calculate differences for ``numRandomShuffles`` different random
    realizations of the measurement pairs, to provide some estimate of the
    uncertainty on our RMS estimates due to the random shuffling.  This
    estimate could be stated and calculated from a more formally derived
    motivation but in practice 50 should be sufficient.

    The LSST Science Requirements Document (LPM-17), or SRD, characterizes the
    photometric repeatability by putting a requirement on the median RMS of
    measurements of non-variable bright stars.  model_phot_rep is a similar
    quantity measured for extended sources (almost exclusively galaxies),
    for which no requirement currently exists in the SRD.

    This present routine calculates this quantity in two different ways:

    1. RMS
    2. interquartile range (IQR)

    **The model_phot_rep scalar measurement is the median of the IQR.**

    This function also returns additional quantities of interest:

    - the pair differences of observations of galaxies,
    - the mean magnitude of each galaxy

    Examples
    --------
    Normally ``calcGalPhotRepeat`` is called by `measureGalPhotRepeat`, using
    data from `lsst.validate.drp.matchreduce.build_matched_dataset`. Here's an
    example of how to call ``calcGalPhotRepeat`` directly given a Butler output
    repository:

    >>> import lsst.daf.persistence as dafPersist
    >>> from lsst.afw.table import SourceCatalog, SchemaMapper, Field
    >>> from lsst.afw.table import MultiMatch, SourceRecord, GroupView
    >>> from lsst.validate.drp.calcnonsrd.model_phot_rep import calcModelPhotRepeat
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
    >>> model_phot_rep = calcModelPhotRepeat(allMatches, magKey)
    """
    mprSamples = [calcModelPhotRepeatSample(matches, magKey)
                  for _ in range(numRandomShuffles)]

    rms = np.array([mpr.rms for mpr in mprSamples]) * u.mmag
    iqr = np.array([mpr.iqr for mpr in mprSamples]) * u.mmag
    magDiff = np.array([mpr.magDiffs for mpr in mprSamples]) * u.mmag
    magMean = np.array([mpr.magMean for mpr in mprSamples]) * u.mag
    mpr = np.mean(iqr)
    return {'rms': rms, 'iqr': iqr, 'magDiff': magDiff, 'magMean': magMean, 'model_phot_rep': mpr}


def calcModelPhotRepeatSample(matches, magKey):
    """Compute one realization of model_phot_rep by randomly sampling pairs of
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
    calcModelPhotRepeat : A wrapper that repeatedly calls this function to build
        the model_phot_rep measurement.
    """
    magDiffs = matches.aggregate(getRandomDiffRmsInMmags, field=magKey)
    magMean = matches.aggregate(np.mean, field=magKey)
    rms, iqr = computeWidths(magDiffs)
    return pipeBase.Struct(rms=rms, iqr=iqr, magDiffs=magDiffs, magMean=magMean,)
