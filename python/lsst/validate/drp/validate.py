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

from __future__ import print_function, absolute_import

import os
from textwrap import TextWrapper

import yaml

from lsst.utils import getPackageDir
from lsst.validate.base import Metric, Job

from .util import repoNameToPrefix


__all__ = ['run', 'runOneFilter']


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def run(repo, dataIds, outputPrefix=None, level="design", verbose=False, **kwargs):
    """Main executable.

    Runs multiple filters, if necessary, through repeated calls to `runOneFilter`.
    Assesses results against SRD specs at specified `level`.

    Inputs
    ------
    repo : string
        The repository.  This is generally the directory on disk
        that contains the repository and mapper.
    dataIds : list of dict
        List of `butler` data IDs of Image catalogs to compare to reference.
        The `calexp` cpixel image is needed for the photometric calibration.
    outputPrefix : str, optional
        Specify the beginning filename for output files.
        The name of each filter will be appended to outputPrefix.
    level : str
        The level of the specification to check: "design", "minimum", "stretch"
    verbose : bool
        Provide detailed output.

    Outputs
    -------
    Names of plot files or JSON file are generated based on repository name,
    unless overriden by specifying `ouputPrefix`.
    E.g., Analyzing a repository "CFHT/output"
        will result in filenames that start with "CFHT_output_".
    The filter name is added to this prefix.  If the filter name has spaces,
        there will be annoyance and sadness as those spaces will appear in the filenames.
    """

    allFilters = set([d['filter'] for d in dataIds])

    if outputPrefix is None:
        outputPrefix = repoNameToPrefix(repo)

    jobs = {}
    for filt in allFilters:
        # Do this here so that each outputPrefix will have a different name for each filter.
        thisOutputPrefix = "%s_%s_" % (outputPrefix.rstrip('_'), filt)
        theseVisitDataIds = [v for v in dataIds if v['filter'] == filt]
        job = runOneFilter(repo, theseVisitDataIds,
                           outputPrefix=thisOutputPrefix,
                           verbose=verbose, filterName=filt,
                           **kwargs)
        jobs[filt] = job

    for filt, job in jobs.items():
        print('')
        print(bcolors.BOLD + bcolors.HEADER + "=" * 65 + bcolors.ENDC)
        print(bcolors.BOLD + bcolors.HEADER + '{0} band summary'.format(filt) + bcolors.ENDC)
        print(bcolors.BOLD + bcolors.HEADER + "=" * 65 + bcolors.ENDC)

        for specName in job.availableSpecLevels:
            passed = True

            measurementCount = 0
            failCount = 0
            for m in job._measurements:
                if m.value is None:
                    continue
                measurementCount += 1
                if not m.checkSpec(specName):
                    passed = False
                    failCount += 1

            if passed:
                print('Passed {level:12s} {count:d} measurements'.format(
                    level=specName, count=measurementCount))
            else:
                msg = 'Failed {level:12s} {failCount} of {count:d} failed'.format(
                    level=specName, failCount=failCount, count=measurementCount)
                print(bcolors.FAIL + msg + bcolors.ENDC)


def runOneFilter(repo, visitDataIds, brightSnr=100,
                 medianAstromscatterRef=25, medianPhotoscatterRef=25, matchRef=500,
                 makePrint=True, makePlot=True, makeJson=True,
                 filterName=None, outputPrefix=None,
                 verbose=False,
                 **kwargs):
    """Main executable for the case where there is just one filter.

    Plot files and JSON files are generated in the local directory
        prefixed with the repository name (where '_' replace path separators),
    unless overriden by specifying `outputPrefix`.
    E.g., Analyzing a repository "CFHT/output"
        will result in filenames that start with "CFHT_output_".

    Parameters
    ----------
    repo : string
        The repository.  This is generally the directory on disk
        that contains the repository and mapper.
    dataIds : list of dict
        List of `butler` data IDs of Image catalogs to compare to reference.
        The `calexp` cpixel image is needed for the photometric calibration.
    brightSnr : float, optional
        Minimum SNR for a star to be considered bright
    medianAstromscatterRef : float, optional
        Expected astrometric RMS [mas] across visits.
    medianPhotoscatterRef : float, optional
        Expected photometric RMS [mmag] across visits.
    matchRef : int, optional
        Expectation of the number of stars that should be matched across visits.
    makePrint : bool, optional
        Print calculated quantities (to stdout).
    makePlot : bool, optional
        Create plots for metrics.  Saved to current working directory.
    makeJson : bool, optional
        Create JSON output file for metrics.  Saved to current working directory.
    outputPrefix : str, optional
        Specify the beginning filename for output files.
    filterName : str, optional
        Name of the filter (bandpass).
    verbose : bool, optional
        Output additional information on the analysis steps.
    """

    if outputPrefix is None:
        outputPrefix = repoNameToPrefix(repo)

    job = Job()

    job.write_json(outputPrefix.rstrip('_') + '.json')

    if makePrint:
        orderedMetrics = ['PA1', 'PF1', 'PA2',
                          'AM1', 'AM2', 'AM3',
                          'AF1', 'AF2', 'AF3',
                          'AD1', 'AD2', 'AD3']
        print(bcolors.BOLD + bcolors.HEADER + "=" * 65 + bcolors.ENDC)
        print(bcolors.BOLD + bcolors.HEADER +
              '{band} band metric measurements'.format(band=filterName) +
              bcolors.ENDC)
        print(bcolors.BOLD + bcolors.HEADER + "=" * 65 + bcolors.ENDC)

        wrapper = TextWrapper(width=65)

        for metricName in orderedMetrics:
            metric = Metric.fromYaml(metricName, yamlDoc=yamlDoc)
            print(bcolors.HEADER + '{name} - {reference}'.format(
                name=metric.name, reference=metric.reference))
            print(wrapper.fill(bcolors.ENDC + '{description}'.format(
                description=metric.description).strip()))

            for specName in metric.getSpecNames(bandpass=filterName):
                try:
                    m = job.getMeasurement(metricName, specName=specName,
                                           bandpass=filterName)
                except RuntimeError:
                    print('\tSkipped {specName:12s} no spec'.format(
                        specName=specName))
                    continue

                if m.value is None:
                    print('\tSkipped {specName:12s} no measurement'.format(
                        specName=specName))
                    continue

                spec = metric.getSpec(specName, bandpass=filterName)
                passed = m.checkSpec(specName)
                if passed:
                    prefix = bcolors.OKBLUE + '\tPassed '
                else:
                    prefix = bcolors.FAIL + '\tFailed '
                infoStr = '{specName:12s} {meas:.4f} {op} {spec:.4f} {units}'.format(
                    specName=specName,
                    meas=m.value,
                    op=metric._operatorStr,  # FIXME make public attribute
                    spec=spec.value,
                    units=spec.units)
                print(prefix + infoStr + bcolors.ENDC)

    return job
