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
"""Main driver functions for metric measurements, plotting, specification
grading, and persistence.
"""

__all__ = ['plot_metrics', 'print_metrics', 'print_pass_fail_summary',
           'run', 'runOneFilter']

import json
import os
import numpy as np
import astropy.units as u

from textwrap import TextWrapper
import astropy.visualization

from lsst.verify import Name
from lsst.verify import Job, MetricSet, SpecificationSet

from lsst.daf.persistence import Butler

from .util import repoNameToPrefix
from .matchreduce import build_matched_dataset, _reduceSources
from .photerrmodel import build_photometric_error_model
from .astromerrmodel import build_astrometric_error_model
from .calcsrd import (measurePA1, measurePA2, measurePF1, measureAMx,
                      measureAFx, measureADx, measureTEx)
from .calcnonsrd import measureModelPhotRepeat
from .plot import (plotAMx, plotPA1, plotTEx, plotPhotometryErrorModel,
                   plotAstrometryErrorModel)


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_json_output(filepath, metrics_package='verify_metrics'):
    """Read JSON from a file into a job object.

    Currently just does a trivial de-serialization with no checking
    to make sure that one results with a valid validate.base.job object.

    Parameters
    ----------
    filepath : `str`
        Source file name for JSON output.

    Returns
    -------
    job : A `validate.base.job` object.
    """
    with open(filepath, 'r') as infile:
        json_data = json.load(infile)

    job = Job.deserialize(**json_data)
    metrics = MetricSet.load_metrics_package(metrics_package)
    job.metrics.update(metrics)
    specs = SpecificationSet.load_metrics_package(metrics_package)
    job.specs.update(specs)
    return job


def get_filter_name_from_job(job):
    """Get the filtername from a validate.base.job object

    Assumes there is only one filter name and that it's the one in
    the first measurement

    Parameters
    ----------
    job : `validate.base.job` object

    Returns
    -------
    filter_name : `str`
    """

    return job.meta['filter_name']


def run(repo_or_json, metrics=None,
        outputPrefix=None, makePrint=True, makePlot=True,
        level='design', metrics_package='verify_metrics', **kwargs):
    """Main entrypoint from ``validateDrp.py``.

    Parameters
    ----------
    repo_or_json : `str`
        The repository.  This is generally the directory on disk
        that contains the repository and mapper.
        This can also be the filepath for a JSON file that contains
        the cached output from a previous run.
    makePrint : `bool`, optional
        Print calculated quantities (to stdout).
    makePlot : `bool`, optional
        Create plots for metrics.  Saved to current working directory.
    level : `str`
        Use <level> E.g., 'design', 'minimum', 'stretch'.
    """
    base_name, ext = os.path.splitext(repo_or_json)
    if ext == '.json':
        load_json = True
    else:
        load_json = False

    # I think I have to interrogate the kwargs to maintain compatibility
    # between Python 2 and Python 3
    # In Python 3 I would have let me mix in a keyword default after *args
    if outputPrefix is None:
        outputPrefix = repoNameToPrefix(base_name)

    if load_json:
        if not os.path.isfile(repo_or_json):
            print("Could not find JSON file %s" % (repo_or_json))
            return

        json_path = repo_or_json
        job = load_json_output(json_path, metrics_package)
        filterName = get_filter_name_from_job(job)
        jobs = {filterName: job}
    else:
        if not os.path.isdir(repo_or_json):
            print("Could not find repo %s" % (repo_or_json))
            return

        repo_path = repo_or_json
        jobs = runOneRepo(repo_path, metrics=metrics, outputPrefix=outputPrefix,
                          metrics_package=metrics_package, **kwargs)

    for filterName, job in jobs.items():
        if makePrint:
            print_metrics(job)
        if makePlot:
            if outputPrefix is None or outputPrefix == '':
                thisOutputPrefix = "%s" % filterName
            else:
                thisOutputPrefix = "%s_%s" % (outputPrefix, filterName)
            plot_metrics(job, filterName, outputPrefix=thisOutputPrefix)

    print_pass_fail_summary(jobs, default_level=level)


def runOneRepo(repo, dataIds=None, metrics=None, outputPrefix='', verbose=False,
               instrument=None, dataset_repo_url=None,
               metrics_package='verify_metrics', **kwargs):
    r"""Calculate statistics for all filters in a repo.

    Runs multiple filters, if necessary, through repeated calls to `runOneFilter`.
    Assesses results against SRD specs at specified `level`.

    Parameters
    ---------
    repo : `str`
        The repository.  This is generally the directory on disk
        that contains the repository and mapper.
    dataIds : `list` of `dict`
        List of butler data IDs of Image catalogs to compare to reference.
        The calexp cpixel image is needed for the photometric calibration.
        Tract IDs must be included if useJointCal is True.
    metrics : `dict` or `collections.OrderedDict`
        Dictionary of `lsst.validate.base.Metric` instances. Typically this is
        data from ``validate_drp``\ 's ``metrics.yaml`` and loaded with
        `lsst.validate.base.load_metrics`.
    outputPrefix : `str`, optional
        Specify the beginning filename for output files.
        The name of each filter will be appended to outputPrefix.
    level : `str`, optional
        The level of the specification to check: "design", "minimum", "stretch".
    verbose : `bool`
        Provide detailed output.
    instrument : `str`
        Name of the instrument.  If None will be extracted from the Butler mapper.
    dataset_repo_url : `str`
        Location of the dataset used.  If None will be set to the path of the repo.
    metrics_package : `string`
        Name of the metrics package to be used in the Jobs created.

    Notes
    -----
    Names of plot files or JSON file are generated based on repository name,
    unless overriden by specifying `ouputPrefix`.
    E.g., Analyzing a repository ``CFHT/output``
    will result in filenames that start with ``CFHT_output_``.
    The filter name is added to this prefix.  If the filter name has spaces,
    there will be annoyance and sadness as those spaces will appear in the filenames.
    """

    def extract_instrument_from_repo(repo):
        """Extract the last part of the mapper name from a Butler repo.
        'lsst.obs.lsstSim.lsstSimMapper.LsstSimMapper' -> 'LSSTSIM'
        'lsst.obs.cfht.megacamMapper.MegacamMapper' -> 'CFHT'
        'lsst.obs.decam.decamMapper.DecamMapper' -> 'DECAM'
        'lsst.obs.hsc.hscMapper.HscMapper' -> 'HSC'
        """
        mapper_class = Butler.getMapperClass(repo)
        instrument = mapper_class.getCameraName()
        return instrument.upper()

    if instrument is None:
        instrument = extract_instrument_from_repo(repo)
    if dataset_repo_url is None:
        dataset_repo_url = repo

    allFilters = set([d['filter'] for d in dataIds])

    jobs = {}
    for filterName in allFilters:
        # Do this here so that each outputPrefix will have a different name for each filter.
        if outputPrefix is None or outputPrefix == '':
            thisOutputPrefix = "%s" % filterName
        else:
            thisOutputPrefix = "%s_%s" % (outputPrefix, filterName)
        theseVisitDataIds = [v for v in dataIds if v['filter'] == filterName]
        job = runOneFilter(repo, theseVisitDataIds, metrics,
                           outputPrefix=thisOutputPrefix,
                           verbose=verbose, filterName=filterName,
                           instrument=instrument,
                           dataset_repo_url=dataset_repo_url,
                           metrics_package=metrics_package, **kwargs)
        jobs[filterName] = job

    return jobs


def runOneFilter(repo, visitDataIds, metrics, brightSnr=100,
                 makeJson=True, filterName=None, outputPrefix='',
                 useJointCal=False, skipTEx=False, verbose=False,
                 metrics_package='verify_metrics',
                 instrument='Unknown', dataset_repo_url='./',
                 skipGalaxies=False, **kwargs):
    r"""Main executable for the case where there is just one filter.

    Plot files and JSON files are generated in the local directory
    prefixed with the repository name (where '_' replace path separators),
    unless overriden by specifying `outputPrefix`.
    E.g., Analyzing a repository ``CFHT/output``
    will result in filenames that start with ``CFHT_output_``.

    Parameters
    ----------
    repo : string or Butler
        A Butler or a repository URL that can be used to construct one.
    dataIds : list of dict
        List of `butler` data IDs of Image catalogs to compare to reference.
        The `calexp` pixel image is needed for the photometric calibration
        unless useJointCal is True, in which the `photoCalib` and `wcs`
        datasets are used instead.  Note that these have data IDs that include
        the tract number.
    metrics : `dict` or `collections.OrderedDict`
        Dictionary of `lsst.validate.base.Metric` instances. Typically this is
        data from ``validate_drp``\ 's ``metrics.yaml`` and loaded with
        `lsst.validate.base.load_metrics`.
    brightSnr : float, optional
        Minimum SNR for a star to be considered bright
    makeJson : bool, optional
        Create JSON output file for metrics.  Saved to current working directory.
    outputPrefix : str, optional
        Specify the beginning filename for output files.
    filterName : str, optional
        Name of the filter (bandpass).
    useJointCal : bool, optional
        Use jointcal/meas_mosaic outputs to calibrate positions and fluxes.
    skipTEx : bool, optional
        Skip TEx calculations (useful for older catalogs that don't have
        PsfShape measurements).
    verbose : bool, optional
        Output additional information on the analysis steps.
    skipGalaxies : bool, optional
        Whether to skip processing galaxies even if model measurements are available; default False.
    """
    job = Job.load_metrics_package(meta={'instrument': instrument,
                                         'filter_name': filterName,
                                         'dataset_repo_url': dataset_repo_url},
                                   subset='validate_drp',
                                   package_name_or_path=metrics_package)

    matchedDataset = build_matched_dataset(repo, visitDataIds,
                                           useJointCal=useJointCal,
                                           skipTEx=skipTEx, skipGalaxies=skipGalaxies)

    photomModel = build_photometric_error_model(matchedDataset)
    astromModel = build_astrometric_error_model(matchedDataset)

    linkedBlobs = [matchedDataset, photomModel, astromModel]

    metrics = job.metrics
    specs = job.specs

    def add_measurement(measurement):
        for blob in linkedBlobs:
            measurement.link_blob(blob)
        job.measurements.insert(measurement)

    for x, D in zip((1, 2, 3), (5., 20., 200.)):
        amxName = 'AM{0:d}'.format(x)
        afxName = 'AF{0:d}'.format(x)
        adxName = 'AD{0:d}'.format(x)

        amx = measureAMx(metrics['validate_drp.'+amxName], matchedDataset, D*u.arcmin)
        add_measurement(amx)

        afx_spec_set = specs.subset(required_meta={'instrument': 'HSC'}, spec_tags=[afxName, ])
        adx_spec_set = specs.subset(required_meta={'instrument': 'HSC'}, spec_tags=[adxName, ])
        for afx_spec_key, adx_spec_key in zip(afx_spec_set, adx_spec_set):
            afx_spec = afx_spec_set[afx_spec_key]
            adx_spec = adx_spec_set[adx_spec_key]
            adx = measureADx(metrics[adx_spec.metric_name], amx, afx_spec)
            add_measurement(adx)
            afx = measureAFx(metrics[afx_spec.metric_name], amx, adx, adx_spec)
            add_measurement(afx)

    pa1 = measurePA1(metrics['validate_drp.PA1'], matchedDataset, filterName)
    add_measurement(pa1)

    snr_base = [5, 20]
    name_flux_all = ["base_PsfFlux"]
    if not skipGalaxies:
        name_flux_all.append('slot_ModelFlux')
    for name_flux in name_flux_all:
        key_model_mag = matchedDataset._matchedCatalog.schema.find(f"{name_flux}_mag").key
        bin_base = 1
        if name_flux == 'slot_ModelFlux':
            name_sources = ['Gal', 'Star']
            prefix_metric = 'model'
        else:
            name_sources = ['Star']
            prefix_metric = 'psf'
        for snr in snr_base:
            for source in name_sources:
                _reduceSources(matchedDataset, matchedDataset._matchedCatalog, extended=source == 'Gal',
                               nameFluxKey=name_flux, goodSnr=snr, goodSnrMax=snr*2,
                               safeSnr=snr*2, safeSnrMax=snr*4)
                for bin_offset in range(2):
                    model_phot_rep = measureModelPhotRepeat(
                        metrics[f'validate_drp.{prefix_metric}PhotRep{source}{bin_base + bin_offset}'],
                        matchedDataset.goodMatches if bin_offset == 0 else matchedDataset.safeMatches,
                        key_model_mag, filterName)
                    add_measurement(model_phot_rep)
            bin_base += 2

    pf1_spec_set = specs.subset(required_meta={'instrument': instrument, 'filter_name': filterName},
                                spec_tags=['PF1', ])
    pa2_spec_set = specs.subset(required_meta={'instrument': instrument, 'filter_name': filterName},
                                spec_tags=['PA2', ])
    # I worry these might not always be in the right order.  Sorting...
    pf1_spec_keys = list(pf1_spec_set.keys())
    pa2_spec_keys = list(pa2_spec_set.keys())
    pf1_spec_keys.sort()
    pa2_spec_keys.sort()
    for pf1_spec_key, pa2_spec_key in zip(pf1_spec_keys, pa2_spec_keys):
        pf1_spec = pf1_spec_set[pf1_spec_key]
        pa2_spec = pa2_spec_set[pa2_spec_key]

        pa2 = measurePA2(metrics[pa2_spec.metric_name], pa1, pf1_spec.threshold)
        add_measurement(pa2)

        pf1 = measurePF1(metrics[pf1_spec.metric_name], pa1, pa2_spec)
        add_measurement(pf1)

    if not skipTEx:
        for x, D, bin_range_operator in zip((1, 2), (1.0, 5.0), ("<=", ">=")):
            texName = 'TE{0:d}'.format(x)
            tex = measureTEx(metrics['validate_drp.'+texName], matchedDataset, D*u.arcmin, bin_range_operator)
            add_measurement(tex)

    if makeJson:
        job.write(outputPrefix+'.json')

    return job


def get_metric(level, metric_label, in_specs):
    for spec in in_specs:
        if level in str(spec) and metric_label in str(spec):
            break
    return Name(package=spec.package, metric=spec.metric)


def plot_metrics(job, filterName, outputPrefix=''):
    """Plot AM1, AM2, AM3, PA1 plus related informational plots.

    Parameters
    ----------
    job : `lsst.validate.base.Job`
        The job to load data from.
    filterName : `str`
        string identifying the filter.
    """
    astropy.visualization.quantity_support()

    specs = job.specs
    measurements = job.measurements
    spec_name = 'design'
    for x in (1, 2, 3):
        amxName = 'AM{0:d}'.format(x)
        afxName = 'AF{0:d}'.format(x)
        # ADx is included on the AFx plots

        amx = measurements[get_metric(spec_name, amxName, specs)]
        afx = measurements[get_metric(spec_name, afxName, specs)]

        if amx.quantity is not None:
            try:
                plotAMx(job, amx, afx, filterName, amxSpecName=spec_name,
                        outputPrefix=outputPrefix)
            except RuntimeError as e:
                print(e)
                print('\tSkipped plot{}'.format(amxName))

    try:
        pa1 = measurements[get_metric(spec_name, 'PA1', specs)]
        plotPA1(pa1, outputPrefix=outputPrefix)
    except RuntimeError as e:
        print(e)
        print('\tSkipped plotPA1')

    try:
        matchedDataset = pa1.blobs['MatchedMultiVisitDataset']
        photomModel = pa1.blobs['PhotometricErrorModel']
        filterName = pa1.extras['filter_name']
        plotPhotometryErrorModel(matchedDataset, photomModel,
                                 filterName=filterName,
                                 outputPrefix=outputPrefix)
    except KeyError as e:
        print(e)
        print('\tSkipped plotPhotometryErrorModel')

    try:
        am1 = measurements[get_metric(spec_name, 'AM1', specs)]
        matchedDataset = am1.blobs['MatchedMultiVisitDataset']
        astromModel = am1.blobs['AnalyticAstrometryModel']
        plotAstrometryErrorModel(matchedDataset, astromModel,
                                 outputPrefix=outputPrefix)
    except KeyError as e:
        print(e)
        print('\tSkipped plotAstrometryErrorModel')

    for x in (1, 2):
        texName = 'TE{0:d}'.format(x)

        try:
            measurement = measurements[get_metric(spec_name, texName, specs)]
            plotTEx(job, measurement, filterName,
                    texSpecName='design',
                    outputPrefix=outputPrefix)
        except (RuntimeError, KeyError) as e:
            print(e)
            print('\tSkipped plot{}'.format(texName))


def get_specs_metrics(job):
    # Get specs for this filter
    subset = job.specs.subset(required_meta={'instrument': job.meta['instrument'],
                                             'filter_name': job.meta['filter_name']},
                              spec_tags=['chromatic'])
    # Get specs that don't depend on filter
    subset.update(job.specs.subset(required_meta={'instrument': job.meta['instrument']},
                                   spec_tags=['achromatic']))
    metrics = {}
    specs = {}
    for spec in subset:
        metric_name = spec.metric.split('_')[0]  # Take first part for linked metrics
        if metric_name in metrics:
            metrics[metric_name].append(Name(package=spec.package, metric=spec.metric))
            specs[metric_name].append(spec)
        else:
            metrics[metric_name] = [Name(package=spec.package, metric=spec.metric), ]
            specs[metric_name] = [spec, ]
    return specs, metrics


def print_metrics(job, levels=('minimum', 'design', 'stretch')):
    specs, metrics = get_specs_metrics(job)

    print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC)
    print(Bcolors.BOLD + Bcolors.HEADER +
          '{band} band metric measurements'.format(band=job.meta['filter_name']) +
          Bcolors.ENDC)
    print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC)

    wrapper = TextWrapper(width=65)
    for metric_name, metric_set in metrics.items():
        metric = job.metrics[metric_set[0]]  # Pick the first one for the description
        print(Bcolors.HEADER + '{name} - {reference}'.format(
            name=metric.name, reference=metric.reference))
        print(wrapper.fill(Bcolors.ENDC + '{description}'.format(
            description=metric.description).strip()))

        for spec_key, metric_key in zip(specs[metric_name], metrics[metric_name]):
            level = None
            if 'release' in job.specs[spec_key].tags:
                # Skip release specs
                continue
            for l in levels:
                if l in str(spec_key):
                    level = l
            try:
                m = job.measurements[metric_key]
            except KeyError:
                print('\tSkipped {metric_key:12s} with spec {spec}: no such measurement'.format(
                    metric_key=metric_name, spec=level))
                continue

            if np.isnan(m.quantity):
                print('\tSkipped {metric_key:12s} no measurement'.format(
                    metric_key=".".join([metric_name, level])))
                continue

            spec = job.specs[spec_key]
            passed = spec.check(m.quantity)
            if passed:
                prefix = Bcolors.OKBLUE + '\tPassed '
            else:
                prefix = Bcolors.FAIL + '\tFailed '
            infoStr = '{specName:12s} {meas:.4g} {op} {spec:.4g}'.format(
                specName=level,
                meas=m.quantity,
                op=spec.operator_str,
                spec=spec.threshold)
            print(prefix + infoStr + Bcolors.ENDC)


def print_pass_fail_summary(jobs, levels=('minimum', 'design', 'stretch'), default_level='design'):
    currentTestCount = 0
    currentFailCount = 0

    for filterName, job in jobs.items():
        specs, metrics = get_specs_metrics(job)
        print('')
        print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC)
        print(Bcolors.BOLD + Bcolors.HEADER + '{0} band summary'.format(filterName) + Bcolors.ENDC)
        print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC)

        for specName in levels:
            measurementCount = 0
            failCount = 0
            for key, m in job.measurements.items():
                if np.isnan(m.quantity):
                    continue
                measurementCount += 1
                metric = key.metric.split("_")[0]  # For compound metrics
                spec_set = specs.get(metric, None)
                if spec_set is None:
                    continue
                spec = None
                for spec_key in spec_set:
                    if specName in spec_key.spec:
                        spec = job.specs[spec_key]
                if spec is None:
                    for spec_key in spec_set:
                        if specName in spec_key.metric:  # For dependent metrics
                            spec = job.specs[spec_key]
                if spec is not None and not spec.check(m.quantity):
                    failCount += 1

            if specName == default_level:
                currentTestCount += measurementCount
                currentFailCount += failCount

            if failCount == 0:
                print('Passed {level:12s} {count:d} measurements'.format(
                    level=specName, count=measurementCount))
            else:
                msg = 'Failed {level:12s} {failCount} of {count:d} failed'.format(
                    level=specName, failCount=failCount, count=measurementCount)
                print(Bcolors.FAIL + msg + Bcolors.ENDC)

        print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC + '\n')

    # print summary against current spec level
    print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC)
    print(Bcolors.BOLD + Bcolors.HEADER + '{0} level summary'.format(default_level) + Bcolors.ENDC)
    print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC)
    if currentFailCount > 0:
        msg = 'FAILED ({failCount:d}/{count:d} measurements)'.format(
            failCount=currentFailCount, count=currentTestCount)
        print(Bcolors.FAIL + msg + Bcolors.ENDC)
    else:
        print('PASSED ({count:d}/{count:d} measurements)'.format(
            count=currentTestCount))

    print(Bcolors.BOLD + Bcolors.HEADER + "=" * 65 + Bcolors.ENDC)
