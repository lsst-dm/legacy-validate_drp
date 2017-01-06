#!/usr/bin/env python

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

from __future__ import print_function

import argparse
import os.path
import sys

from lsst.utils import getPackageDir
from lsst.validate.base import load_metrics
from lsst.validate.drp import validate, util


if __name__ == "__main__":
    description = """
    Calculate and plot validation Key Project Metrics from the LSST SRD.
    http://ls.st/LPM-17

    Produces results to:
    STDOUT
        Summary of key metrics
    REPONAME*.png
        Plots of key metrics.  Generated in current working directory.
    REPONAME*.json
        JSON serialization of each KPM.

    where REPONAME is based on the repository name but with path separators
    replaced with underscores.  E.g., "Cfht/output" -> "Cfht_output_"
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('repo', type=str,
                        help='path to a repository containing the output of processCcd')
    parser.add_argument('--configFile', '-c', type=str, default=None,
                        help='YAML configuration file validation parameters and dataIds.')
    parser.add_argument('--metricsFile',
                        default=os.path.join(getPackageDir('validate_drp'),
                                             'etc', 'metrics.yaml'),
                        help='Path of YAML file with LPM-17 metric definitions.')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
                        help='Display additional information about the analysis.')
    parser.add_argument('--noplot', dest='makePlot',
                        default=True, action='store_false',
                        help='Skip making plots of performance.')
    parser.add_argument('--level', type=str, default='design',
                        help='Level of SRD requirement to meet: "minimum", "design", "stretch"')

    args = parser.parse_args()

    if not os.path.isdir(args.repo):
        print("Could not find repo %r" % (args.repo,))
        sys.exit(1)

    kwargs = {}
    if args.configFile:
        pbStruct = util.loadDataIdsAndParameters(args.configFile)
        kwargs = pbStruct.getDict()

    kwargs['verbose'] = args.verbose
    kwargs['makePlot'] = args.makePlot

    if not args.configFile or not pbStruct.dataIds:
        kwargs['dataIds'] = util.discoverDataIds(args.repo)
        if args.verbose:
            print("VISITDATAIDS: ", kwargs['dataIds'])

    kwargs['verbose'] = args.verbose
    kwargs['level'] = args.level

    if not os.path.exists(args.metricsFile):
        print('Could not find metric definitions: {0}'.format(args.metricsFile))
        sys.exit(1)
    metrics = load_metrics(args.metricsFile)
    kwargs['metrics'] = metrics

    validate.run(args.repo, **kwargs)
