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

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

from lsst.verify import Measurement, Datum


def measureAA1(metric, matchedDataset, magRange=None, verbose=False):
    r"""Measurement of AA1: Median error in the absolute astrometric positions
        (per coordinate, in milliarcseconds).

    Parameters
    ----------
    metric : `lsst.verify.Metric`
        An AA1, `~lsst.verify.Metric` instance.
    matchedDataset : lsst.verify.Blob
        Contains the spatially matched dataset for the measurement
    magRange : 2-element `list`, `tuple`, or `numpy.ndarray`, optional
        brighter, fainter limits of the magnitude range to include.
        Default: ``[17.5, 21.5]`` mag.
    verbose : `bool`, optional
        Output additional information on the analysis steps.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of AA1 and associated metadata.

    Notes
    -----
    This table below is provided ``validate_drp``\ 's :file:`metrics.yaml`.

    LPM-17 dated 2011-07-06

    Specification:
        The LSST astrometric system must transform to an external system (e.g.
        ICRF extension) with the median accuracy of AA1 milliarcsec (Table 20).

        The accuracy of absolute astrometry is driven by the linkage and orbital computations for
        solar system objects. A somewhat weaker constraint is also placed by the need for positional
        cross-correlation with external catalogs. Note that the delivered absolute astrometric
        accuracy may be fundamentally limited by the accuracy of astrometric standard catalogs.


    ========================= ====== ======= =======
    Abolute astrometry          Specification
    ------------------------- ----------------------
                       Metric Design Minimum Stretch
    ========================= ====== ======= =======
            AA1 (milliarcsec)     50     100      20
    ========================= ====== ======= =======

    TABLE 20: The median error in the absolute astrometric positions (per coordinate, in milliarcsec).
    """

    matches = matchedDataset.safeMatches

    datums = {}

    if magRange is None:
        magRange = np.array([17.0, 21.5]) * u.mag
    else:
        assert len(magRange) == 2
        if not isinstance(magRange, u.Quantity):
            magRange = np.array(magRange) * u.mag
    datums['magRange'] = Datum(quantity=magRange, description='Stellar magnitude selection range.')

    # Register measurement extras
    medianRaOffsets, medianDecOffsets, raOffsets, decOffsets = calcPosOffsets(
        matches,
        magRange=magRange,
        verbose=verbose)

    # rmsDistances, separations = calcRmsDistances(
    #    matches,
    #    magRange=magRange,
    #    verbose=verbose)

#    import pdb
#    pdb.set_trace()

    if len(raOffsets) == 0:
        # Should be a proper log message
        print('No matched stars found to calculate AA1.')
        datums['raSeparations'] = Datum(quantity=None, label='raSeparations')
        quantity = np.nan * u.marcsec
        datums['decSeparations'] = Datum(quantity=None, label='decSeparations')
    else:
        datums['raSeparations'] = Datum(quantity=raOffsets, label='raSeparations')
        datums['decSeparations'] = Datum(quantity=decOffsets, label='decSeparations')
        # Requirement calls for each axis separately, but we'll just take the
        #   maximum of the RA, Dec offsets:
        quantity = np.max([medianRaOffsets.value, medianDecOffsets.value])*u.marcsec

    return Measurement(metric.name, quantity, extras=datums)


def calcPosOffsets(groupView, magRange, verbose=False):

    minMag, maxMag = magRange.to(u.mag).value

    def magInRange(cat):
        mag = cat.get('base_PsfFlux_mag')
        w, = np.where(np.isfinite(mag))
        medianMag = np.median(mag[w])
        return minMag <= medianMag and medianMag < maxMag

    groupViewInMagRange = groupView.where(magInRange)

    # import pdb
    # pdb.set_trace()

    # Create an Astropy SkyCoord object for all objects in the groupView:

    ra_src = groupViewInMagRange.apply(returnVal, field='coord_ra')
    dec_src = groupViewInMagRange.apply(returnVal, field='coord_dec')
    # ra_src = groupViewInMagRange.get('coord_ra')
    # dec_src = groupViewInMagRange.get('coord_dec')
    sc_src = SkyCoord(ra_src*u.rad, dec_src*u.rad)

    # Use Astroquery to get Gaia DR2 data in the region:
    minRa, maxRa = np.min(sc_src.ra.degree), np.max(sc_src.ra.degree)
    minDec, maxDec = np.min(sc_src.dec.degree), np.max(sc_src.dec.degree)
    width = maxRa - minRa
    height = maxDec - minDec
    cen = SkyCoord((minRa+width/2.0)*u.deg, (minDec+height/2.0)*u.deg)
    width = u.Quantity(width, u.deg)
    height = u.Quantity(height, u.deg)
    gaia_mch = Gaia.query_object_async(coordinate=cen, width=width, height=height)
    sc_gaia = SkyCoord(gaia_mch['ra'], gaia_mch['dec'])

    # Match the input groupView catalog to Gaia:
    src_match = sc_src.match_to_catalog_sky(sc_gaia)
    sep_match = src_match[1]

    gaia_gmag = gaia_mch['phot_g_mean_mag']
    # gaia_bpmag = gaia_mch['phot_bp_mean_mag']
    # gaia_rpmag = gaia_mch['phot_rp_mean_mag']

    src_mag = groupViewInMagRange.apply(returnVal, field='base_PsfFlux_mag')

    # Things with >2.5" separations are obviously bad matches, so remove them:
    okmch = (sep_match.arcsec < 2.5)
    # matchsep = sep_match[okmch]

    # A "good" match should have a similar magnitude in both catalogs
    #   (we compare to the median difference to account for different bandpasses)
    magdiff = src_mag-gaia_gmag[src_match[0]]
    okmagdiff = (np.abs(magdiff-np.median(magdiff[okmch])) < 1)

    # Calculate delta_ra and delta_dec:
    ra_gaia_match = sc_gaia[src_match[0]].ra
    dec_gaia_match = sc_gaia[src_match[0]].dec
    dra = (sc_src.ra - ra_gaia_match)*np.cos(sc_src.dec.radian)
    ddec = sc_src.dec - dec_gaia_match

    medianRaOffset = np.median(np.abs(dra[okmch & okmagdiff].marcsec))
    medianDecOffset = np.median(np.abs(ddec[okmch & okmagdiff].marcsec))

    # Return the RA, Dec offsets:
    offsetRa = dra[okmch & okmagdiff].marcsec * u.marcsec
    offsetDec = ddec[okmch & okmagdiff].marcsec * u.marcsec

    return medianRaOffset*u.marcsec, medianDecOffset*u.marcsec, offsetRa, offsetDec


def returnVal(cat):
    return cat
