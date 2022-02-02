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


import numpy as np
import os
import unittest

import lsst.utils.tests
from lsst.validate.drp.photerrmodel import photErrModel, fitPhotErrModel


class PhotErrCase(lsst.utils.tests.TestCase):
    """Testing photometric error model fitting performance and failure modes."""
    def setUp(self):
        self.sigmaSys, self.gamma, self.m5 = 0.01, 0.039, 24.35  # mag, '', mag

        # Set seed for repeatibility
        np.random.seed(96701237)
        m = np.random.randn(1000)*2 + 25
        self.mag = m[m < 25]
        self.mag_err = photErrModel(
            self.mag, self.sigmaSys, self.gamma, self.m5)

        # Resample mag
        # Set seed for repeatibility.
        #  We explicityly reset it here to a known value
        #  in case `photErrModel` called np.random
        np.random.seed(23987)
        self.noisy_mag = self.mag + np.random.randn(len(self.mag_err))*self.mag_err

    def test_perfect_fit_phot_error_model(self):
        """Does a simple fit to a small, perfect, data set work?"""
        fit_results = fitPhotErrModel(self.mag, self.mag_err)

        self.assertFloatsAlmostEqual(
            fit_results['sigmaSys'].value, self.sigmaSys, atol=1e-7)
        self.assertFloatsAlmostEqual(
            fit_results['gamma'].value, self.gamma, atol=1e-7)
        self.assertFloatsAlmostEqual(
            fit_results['m5'].value, self.m5, atol=1e-7)

    def test_noisy_fit_phot_error_model(self):
        """Does a simple fit to a small, perfect, data set work?"""
        fit_results = fitPhotErrModel(self.noisy_mag, self.mag_err)

        # Different absolute tolerances because we expect some variation in the
        # fit results to the hnoisy data and that variation is different
        # for different parameters
        self.assertFloatsAlmostEqual(
            fit_results['sigmaSys'].value, self.sigmaSys, atol=2e-2)
        self.assertFloatsAlmostEqual(
            fit_results['gamma'].value, self.gamma, atol=2e-2)
        self.assertFloatsAlmostEqual(
            fit_results['m5'].value, self.m5, atol=0.2)

    def test_failed_fit_phot_error_model(self):
        """Does a failed fit recover and return NaN?"""
        testDir = os.path.dirname(__file__)
        failing_datafile = os.path.join(testDir, 'mag_magerr_bright.dat')
        failing_mag, failing_mag_err = np.loadtxt(failing_datafile)

        fit_results = fitPhotErrModel(failing_mag, failing_mag_err)

        self.assertTrue(np.isnan(fit_results['sigmaSys'].value))
        self.assertTrue(np.isnan(fit_results['gamma'].value))
        self.assertTrue(np.isnan(fit_results['m5'].value))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
