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

from lsst.validate.drp.repeatability import measurePhotRepeat
from lsst.validate.drp.matchreduce import reduceSources


def measure_model_phot_rep(metrics, filterName, matchedDataset, snr_bins=None):
    """Measurement of the model_phot_rep metric: photometric repeatability of
    measurements of across a set of observations.

    Parameters
    ----------
    metrics : `lsst.verify.metricset.MetricSet`
        A metric set containing all of the expected validate_drp.*PhotRep* metrics.
    filterName : `str`
        Name of filter used for all observations.
    matchedDataset : `lsst.verify.Blob`
        Matched dataset blob, as returned by `lsst.validate.drp.matchreduce.build_matched_dataset`.
    snr_bins : `iterable`
        An iterable of pairs of SNR bins, each specified as a pair of floats
        (lower, upper). Default [((5, 10), (10, 20)), ((20, 40), (40, 80))].
        The total number of bins must not exceed the number of metrics,
        which is currently four.

    Returns
    -------
    measurements : `list` [`lsst.verify.Measurement`]
        A list of metric measurements with associated metadata.

    Notes
    -----
    Each SNR bin can be specified independently; their edges don't need to align and could overlap.
    Bins are paired because reduce_sources already has a mechanism to apply two different SNR cuts,
    which could be generalized to an iterable if desired.

    The default SNR bins were chosen in DM-21380
    (https://jira.lsstcorp.org/browse/DM-21380) and are somewhat arbitrary.
    It is probably not useful to measure SNR<5, whereas one might reasonably
    prefer narrower bins or a higher maximum SNR than 80.
    """
    if snr_bins is None:
        snr_bins = [((5, 10), (10, 20)), ((20, 40), (40, 80))]
    name_flux_all = ["base_PsfFlux", 'slot_ModelFlux']
    measurements = []
    for name_flux in name_flux_all:
        key_model_mag = matchedDataset._matchedCatalog.schema.find(f"{name_flux}_mag").key
        if name_flux == 'slot_ModelFlux':
            name_sources = ['Gal', 'Star']
            prefix_metric = 'model'
        else:
            name_sources = ['Star']
            prefix_metric = 'psf'
        for idx_bins, ((snr_one_min, snr_one_max), (snr_two_min, snr_two_max)) in enumerate(snr_bins):
            bin_base = 1 + 2*idx_bins
            for source in name_sources:
                reduceSources(matchedDataset, matchedDataset._matchedCatalog, extended=source == 'Gal',
                              nameFluxKey=name_flux, goodSnr=snr_one_min, goodSnrMax=snr_one_max,
                              safeSnr=snr_two_min, safeSnrMax=snr_two_max)
                for bin_offset in [0, 1]:
                    model_phot_rep = measurePhotRepeat(
                        metrics[f'validate_drp.{prefix_metric}PhotRep{source}{bin_base + bin_offset}'],
                        filterName,
                        matchedDataset.goodMatches if bin_offset == 0 else matchedDataset.safeMatches,
                        key_model_mag)
                    measurements.append(model_phot_rep)
    return measurements
