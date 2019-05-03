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


# Ensure that this script will run on a mis-configured node
# that may default to a backend that requires X
# e.g., 'Qt5Agg', even though there is is no display available
# Putting this here in the command-line script is fine because no one
# should import this script.
import matplotlib
matplotlib.use('Agg')  # noqa E402

import argparse

from lsst.validate.drp import validate, util


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('repo', type=str,
                        help='path to a repository containing the output of processCcd')
    parser.add_argument('--outputPrefix', '-o', type=str, default=None,
                        help="""
                        Define basic name prefix for output files.  Can include paths.
                        E.g., --outputPrefix="mydir/awesome_reduction" will produce
                        "mydir/awesome_reduction_r.json" for the r-band JSON file.
                        """)
    parser.add_argument('--configFile', '-c', type=str, default=None,
                        help='YAML configuration file validation parameters and dataIds.')
    parser.add_argument('--metricsPackage',
                        default='verify_metrics',
                        help='Name of the repository with YAML definitions of LPM-17 metrics.')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
                        help='Display additional information about the analysis.')
    parser.add_argument('--notitle', dest='showTitle',
                        default=True, action='store_false',
                        help='Skip titles on plots for cleaner figures.')
    parser.add_argument('--noplot', dest='makePlot',
                        default=True, action='store_false',
                        help='Skip making plots of performance.')
    parser.add_argument('--plotext', dest='plotExt', default='png',
                        help='Plot file extension name.  E.g., "png", "pdf", "svg".')
    parser.add_argument('--level', type=str, default='design',
                        help='Level of SRD requirement to meet: "minimum", "design", "stretch"')

    args = parser.parse_args()

    # Should clean up the duplication here between this and validate.run
    if args.repo[-5:] == '.json':
        load_json = True
    else:
        load_json = False

    kwargs = {}

    if not load_json:
        if args.configFile:
            pbStruct = util.loadDataIdsAndParameters(args.configFile)
            kwargs = pbStruct.getDict()

        if not args.configFile or not pbStruct.dataIds:
            kwargs['dataIds'] = util.discoverDataIds(args.repo)
            if args.verbose:
                print("VISITDATAIDS: ", kwargs['dataIds'])

        kwargs['metrics_package'] = args.metricsPackage

    kwargs['verbose'] = args.verbose
    kwargs['makePlot'] = args.makePlot
    kwargs['level'] = args.level
    kwargs['outputPrefix'] = args.outputPrefix
    kwargs['showTitle'] = args.showTitle
    kwargs['plotExt'] = args.plotExt

    validate.run(args.repo, **kwargs)
