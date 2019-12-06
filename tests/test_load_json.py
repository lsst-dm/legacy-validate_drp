#
# LSST Data Management System
# Copyright 2012-2016 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#


import os
import unittest

import lsst.utils

from lsst.utils.tests import ExecutablesTestCase
from lsst.validate.drp.validate import (
    get_filter_name_from_job, load_json_output, plot_metrics, print_metrics)


class ParseJsonJob(ExecutablesTestCase):
    """Testing loading of JSON cache files."""

    def setUp(self):
        testDataDir = os.path.dirname(__file__)
        self.jsonFile = os.path.join(testDataDir, 'CfhtQuick_output_r.json')
        self.jsonFile_filter = 'r'
        self.longMessage = True
        self.executable_dir = os.path.join(lsst.utils.getPackageDir("VALIDATE_DRP"),
                                           "bin")

    def testLoadFromValidate(self):
        self.assertExecutable("validateDrp.py",
                              root_dir=self.executable_dir,
                              args=[self.jsonFile, "--noplot"],
                              msg="CFHT Quick Test failed")

    def testLoadJsonJob(self):
        """Can we load a Job from a JSON file?"""

        # without throwing an error
        job = load_json_output(self.jsonFile)
        # Spot-check a few attributes
        self.assertEqual(len(job.measurements), 42)

    def testParseJobFilterName(self):
        """Do we correctly read the filterName from a Job object?"""
        job = load_json_output(self.jsonFile)
        filterName = get_filter_name_from_job(job)
        self.assertEqual(filterName, self.jsonFile_filter)

    def testPrintMetricsFromJsonJob(self):
        """Does printing the metrics run without error using itself?

        This is in essence half the big test.
        If we load a file do we get printed metrics?
        Next TODO is to actually check for the values printed.
        """
        job = load_json_output(self.jsonFile)
        filterName = get_filter_name_from_job(job)
        print_metrics(job, filterName)

    def testPlotMetricsFromJsonJob(self):
        """Does plotting the metrics run and produce the correct filenames?

        This could be loosely seen as a test of the outputPrefix handling
        and thus DM-11410, it's a rather incomplete one because the chain
        of wrapping functions is different.
        """
        job = load_json_output(self.jsonFile)
        filterName = get_filter_name_from_job(job)

        noOutputPrefixFiles = ['check_astrometry.png',
                               'check_photometry.png',
                               'PA1.png',
                               'validate_drp.AM1_D_5_arcmin_17.0_21.5_mag.png',
                               'validate_drp.TE1_D_1_arcmin.png',
                               'validate_drp.TE2_D_5_arcmin.png']
        # test with no output prefix.
        plot_metrics(job, filterName)
        for filename in noOutputPrefixFiles:
            assert os.path.exists(filename), "File not created: %s"%filename
            os.remove(filename)

        # test that outputPrefix is prepended correctly.
        outputPrefix = 'foobarbaz'
        outputPrefixFiles = ['%s_%s' % (outputPrefix, f) for f in noOutputPrefixFiles]
        plot_metrics(job, filterName, outputPrefix=outputPrefix)
        for filename in outputPrefixFiles:
            assert os.path.exists(filename), "File not created: %s"%filename
            os.remove(filename)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
