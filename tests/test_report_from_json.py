# LSST Data Management System
# Copyright 2017 AURA/LSST.
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


import os
import tempfile
import unittest

import lsst.utils
from lsst.validate.drp import report_performance


class ReportPerformanceFromJob(unittest.TestCase):
    """Testing release performance summary."""

    def setUp(self):
        test_data_dir = os.path.dirname(__file__)
        self.release_specs_package = 'verify_metrics'
        self.srd_levels = ['design', 'minimum']
        self.release_levels = ['FY17', 'FY18']
        self.report_files = [os.path.join(test_data_dir, f) for f in
                             ['CfhtQuick_output_r_report_{}_{}.rst'.format(s, r)
                              for s, r in zip(self.srd_levels, self.release_levels)]]
        self.json_file = os.path.join(test_data_dir, 'CfhtQuick_output_r.json')
        self.json_file_filter = 'r'

    def test_generate_report_from_json(self):
        """Can we read in JSON results and make a report.

        Reference datasets are from a v13.0 validation_data_hsc run.

        Check for the default (srd_level, release_level) = ('design', 'FY17') and ('minimum', 'FY18')
        """
        # Manually use temporary directories here,
        #  because I can't figure out how to get py.test tmpdir fixture
        #  to work in the unittest.TestCase context.
        tmp_dir = tempfile.mkdtemp()

        srd_release_report = zip(self.srd_levels, self.release_levels, self.report_files)
        for srd_level, release_level, ref_file in srd_release_report:
            out_file_name = os.path.join(
                tmp_dir,
                "report_performance_test_{}_{}.rst".format(srd_level, release_level))
            report_performance.run(
                [self.json_file],
                out_file_name,
                srd_level=srd_level,
                release_specs_package='verify_metrics',
                release_level=release_level)

            assert(os.path.exists(out_file_name))
            with open(out_file_name) as fh:
                of_lines = fh.readlines()
            with open(ref_file) as fh:
                rf_lines = fh.readlines()
            self.maxDiff = None
            self.assertEqual(''.join(of_lines), ''.join(rf_lines),
                             msg=f"Files are {out_file_name} and {ref_file}")
            # Cleanup our temp file
            os.remove(out_file_name)

        # Cleanup our temp directory
        os.removedirs(tmp_dir)


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
