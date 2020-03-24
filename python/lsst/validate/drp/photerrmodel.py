# LSST Data Management System
# Copyright 2016 AURA/LSST.
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
"""Analytic single-visit photometric error model.
"""

__all__ = ['photErrModel', 'fitPhotErrModel', 'build_photometric_error_model']

import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit

from lsst.verify import Blob, Datum


def photErrModel(mag, sigmaSys, gamma, m5, **kwargs):
    r"""Model of photometric error for a single visit.

    The model is described in the LSST Overview paper:
    http://arxiv.org/abs/0805.2366v4.

    Photometric error of a single visit (Eq. 4):

    .. math:

       \sigma_1^2 = \sigma_\mathrm{sys}^2 + \sigma_\mathrm{rand}^2

    Where random uncertainty is (Eq. 5):

    .. math::

       \sigma_\mathrm{rand}^2 = (0.04 - \gamma) * x + \gamma * x^2~[\mathrm{mag}^2]

    and $x = 10^(0.4*(m-m_5))$.

    Parameters
    ----------
    mag : `astropy.unit.Quantity`
        Source magnitude.
    sigmaSq : `astropy.unit.Quantity`
        Systematic photometric error (magnitude).
    gamma : `astropy.unit.Quantity`
        Proxy for sky brightness and readout noise (dimensionless).
    m5 : `astropy.unit.Quantity`
        5-sigma depth (magnitude).

    Returns
    -------
    sigma : `astropy.units.Quantity`
        Photometric error for a single visit.
    """
    x = 10**(0.4*(mag - m5))
    sigmaRandSq = (0.04 - gamma) * x + gamma * x**2
    sigmaSq = sigmaSys**2 + sigmaRandSq
    return np.sqrt(sigmaSq)


def fitPhotErrModel(mag, mag_err):
    """Fit photometric error model from the LSST Overview paper:

    http://arxiv.org/abs/0805.2366v4

    The fit is performed with `scipy.optimize.curvefit`.

    Parameters
    ----------
    mag : `astropy.units.Quantity`
        Magnitude.
    mag_err : `astropy.units.Quantity`
        Magnitude uncertainty or variation.

    Returns
    -------
    params : `dict`
        Fitted model. Fields are:

        - `sigmaSys`: systematic error (magnitude, `astropy.units.Quantity`).
        - `gamma`: Proxy for sky brightness and readout noise (dimensionless,
          `astropy.units.Quantity`).
        - `m5`: 5-sigma limiting depth (magnitude, `astropy.units.Quantity`).

    See also
    --------
    photErrModel
    """
    # if not isinstance(mag, u.Quantity):
    #     mag = mag * u.mag
    # if not isinstance(mag_err, u.Quantity):
    #     mag_err = mag_err * u.mag

    # p0 = [0.01 * u.mag,  # sigmaSys
    #       0.039 * u.Unit(''),  # gamma
    #       24.35 * u.mag]  # m5

    # fit_params, fit_param_covariance = curve_fit(
    #     photErrModel, mag, mag_err, p0=p0)

    # params = {
    #     'sigmaSys': fit_params[0],
    #     'gamma': fit_params[1],
    #     'm5': fit_params[2],
    # }
    if isinstance(mag, u.Quantity):
        mag = mag.to(u.mag).value
    if isinstance(mag_err, u.Quantity):
        mag_err = mag_err.to(u.mag).value

    p0 = [0.01,  # sigmaSys (mag)
          0.039,  # gamma ('')
          24.35]  # m5 (mag)

    try:
        fit_params, fit_param_covariance = curve_fit(
            photErrModel, mag, mag_err, p0=p0)
        sigmaSys, gamma, m5 = fit_params
    except (RuntimeError, ValueError) as e:
        print("fitPhotErrorModel fitting failed with")
        print(e)
        print("sigmaSys, gamma, m5 are being set to NaN.")
        sigmaSys, gamma, m5 = np.nan, np.nan, np.nan

    params = {
        'sigmaSys': sigmaSys * u.mag,
        'gamma': gamma * u.Unit(''),
        'm5': m5 * u.mag,
    }
    return params


def build_photometric_error_model(matchedMultiVisitDataset, brightSnrMin=100, medianRef=100,
                                  matchRef=500):
    r"""Returns a serializable analytic photometry error model for a single visit.

    This model is originally presented in http://arxiv.org/abs/0805.2366v4
    (Eq 4, 5):

    .. math::

       \sigma_1^2 &= \sigma_\mathrm{sys}^2 + \sigma_\mathrm{rand}^2 \\
       x &= 10^{0.4(m-m_5)} \\
       \sigma_\mathrm{rand}^2 &= (0.04 - \gamma) x + \gamma x^2~[\mathrm{mag}^2]

    Parameters
    ----------
    matchedMultiVisitDataset : `lsst.valididate.drp.matchreduce.MatchedMultiVisitDataset`
        A dataset containing matched statistics for stars across multiple
        visits.
    brightSnrMin : `float` or `astropy.unit.Quantity`, optional
        Minimum SNR for a star to be considered "bright."
    medianRef : `float` or `astropy.unit.Quantity`, optional
        Median reference astrometric scatter (millimagnitudes by default).
    matchRef : `int` or `astropy.unit.Quantity`, optional
        Should match at least matchRef stars.

    Returns
    ----------
    blob : `lsst.verify.Blob`
        Blob with datums:

        - ``brightSnrMin``: Threshold in SNR for bright sources used in this  model.
        - ``sigmaSys``: Systematic error floor.
        - ``gamma``: Proxy for sky brightness and read noise.
        - ``m5``: 5-sigma photometric depth (magnitudes).
        - ``photRms``: RMS photometric scatter for 'good' stars (millimagnitudes).

    Notes
    -----
    The scatter and match defaults are appropriate to SDSS are stored here.
    For SDSS, stars with mag < 19.5 should be completely well measured.
    This limit is a band-dependent statement most appropriate to r.
    """
    blob = Blob('PhotometricErrorModel')

    # FIXME add a description field to blobs?
    # _doc['doc'] \
    #     = "Photometric uncertainty model from " \
    #       "http://arxiv.org/abs/0805.2366v4 (Eq 4, 5): " \
    #       "sigma_1^2 = sigma_sys^2 + sigma_rand^2, " \
    #       "sigma_rand^2 = (0.04 - gamma) * x + gamma * x^2 [mag^2] " \
    #       "where x = 10**(0.4*(m-m_5))"

    if not isinstance(medianRef, u.Quantity):
        medianRef = medianRef * u.mmag
    if not isinstance(brightSnrMin, u.Quantity):
        brightSnrMin = brightSnrMin * u.Unit('')
    _compute(blob,
             matchedMultiVisitDataset['snr'].quantity,
             matchedMultiVisitDataset['mag'].quantity,
             matchedMultiVisitDataset['magerr'].quantity,
             matchedMultiVisitDataset['magrms'].quantity,
             matchedMultiVisitDataset['dist'].quantity,
             len(matchedMultiVisitDataset.matchesFaint),
             brightSnrMin,
             medianRef,
             matchRef)
    return blob


def _compute(blob, snr, mag, magErr, magRms, dist, nMatch,
             brightSnrMin, medianRef, matchRef):
    blob['brightSnrMin'] = Datum(quantity=brightSnrMin,
                                 label='Bright SNR',
                                 description='Threshold in SNR for bright sources used in this model')

    bright = np.where(snr > blob['brightSnrMin'].quantity)
    blob['photScatter'] = Datum(quantity=np.median(magRms[bright]),
                                label='RMS',
                                description='RMS photometric scatter for good stars')
    print('Photometric scatter (median) - SNR > {0:.1f} : {1:.1f}'.format(
          blob['brightSnrMin'].quantity, blob['photScatter'].quantity.to(u.mmag)))

    fit_params = fitPhotErrModel(mag[bright], magErr[bright])
    blob['sigmaSys'] = Datum(quantity=fit_params['sigmaSys'],
                             label='sigma(sys)',
                             description='Systematic error floor')
    blob['gamma'] = Datum(quantity=fit_params['gamma'],
                          label='gamma',
                          description='Proxy for sky brightness and read noise')
    blob['m5'] = Datum(quantity=fit_params['m5'],
                       label='m5',
                       description='5-sigma depth')

    if blob['photScatter'].quantity > medianRef:
        msg = 'Median photometric scatter {0:.3f} is larger than ' \
              'reference : {1:.3f}'
        print(msg.format(blob['photScatter'].quantity.value, medianRef))
    if nMatch < matchRef:
        msg = 'Number of matched sources {0:d} is too small ' \
              '(should be > {0:d})'
        print(msg.format(nMatch, matchRef))
