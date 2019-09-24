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
"""Blob classes that reduce a multi-visit dataset and encapsulate data
for measurement classes, plotting functions, and JSON persistence.
"""

__all__ = ['build_matched_dataset']

import numpy as np
import astropy.units as u
from sqlalchemy.exc import OperationalError
import sqlite3

import lsst.geom as geom
import lsst.daf.persistence as dafPersist
from lsst.afw.table import (SourceCatalog, SchemaMapper, Field,
                            MultiMatch, SimpleRecord, GroupView,
                            SOURCE_IO_NO_FOOTPRINTS)
from lsst.afw.fits import FitsError
from lsst.verify import Blob, Datum

from .util import (getCcdKeyName, raftSensorToInt, positionRmsFromCat,
                   ellipticity_from_cat)


def build_matched_dataset(repo, dataIds, matchRadius=None, safeSnr=50.,
                          useJointCal=False, skipTEx=False):
    """Construct a container for matched star catalogs from multple visits, with filtering,
    summary statistics, and modelling.

    `lsst.verify.Blob` instances are serializable to JSON.

    Parameters
    ----------
    repo : `str` or `Butler`
        A Butler instance or a repository URL that can be used to construct
        one.
    dataIds : `list` of `dict`
        List of `butler` data IDs of Image catalogs to compare to reference.
        The `calexp` cpixel image is needed for the photometric calibration.
    matchRadius :  geom.Angle(), optional
        Radius for matching. Default is 1 arcsecond.
    safeSnr : `float`, optional
        Minimum median SNR for a match to be considered "safe".
    useJointCal : `bool`, optional
        Use jointcal/meas_mosaic outputs to calibrate positions and fluxes.
    skipTEx : `bool`, optional
        Skip TEx calculations (useful for older catalogs that don't have
        PsfShape measurements).

    Attributes of returned Blob
    ----------
    filterName : `str`
        Name of filter used for all observations.
    mag : `astropy.units.Quantity`
        Mean PSF magnitudes of stars over multiple visits (magnitudes).
    magerr : `astropy.units.Quantity`
        Median 1-sigma uncertainty of PSF magnitudes over multiple visits
        (magnitudes).
    magrms : `astropy.units.Quantity`
        RMS of PSF magnitudes over multiple visits (magnitudes).
    snr : `astropy.units.Quantity`
        Median signal-to-noise ratio of PSF magnitudes over multiple visits
        (dimensionless).
    dist : `astropy.units.Quantity`
        RMS of sky coordinates of stars over multiple visits (milliarcseconds).

        *Not serialized.*
    goodMatches
        all good matches, as an afw.table.GroupView;
        good matches contain only objects whose detections all have

        1. a PSF Flux measurement with S/N > 1
        2. a finite (non-nan) PSF magnitude. This separate check is largely
           to reject failed zeropoints.
        3. and do not have flags set for bad, cosmic ray, edge or saturated

        *Not serialized.*

    safeMatches
        safe matches, as an afw.table.GroupView. Safe matches
        are good matches that are sufficiently bright and sufficiently
        compact.

        *Not serialized.*
    magKey
        Key for `"base_PsfFlux_mag"` in the `goodMatches` and `safeMatches`
        catalog tables.

        *Not serialized.*
    """
    blob = Blob('MatchedMultiVisitDataset')

    if not matchRadius:
        matchRadius = geom.Angle(1, geom.arcseconds)

    # Extract single filter
    blob['filterName'] = Datum(quantity=set([dId['filter'] for dId in dataIds]).pop(),
                               description='Filter name')

    # Record important configuration
    blob['useJointCal'] = Datum(quantity=useJointCal,
                                description='Whether jointcal/meas_mosaic calibrations were used')

    # Match catalogs across visits
    blob._catalog, blob._matchedCatalog = \
        _loadAndMatchCatalogs(repo, dataIds, matchRadius,
                              useJointCal=useJointCal, skipTEx=skipTEx)

    blob.magKey = blob._matchedCatalog.schema.find("base_PsfFlux_mag").key
    # Reduce catalogs into summary statistics.
    # These are the serialiable attributes of this class.
    _reduceStars(blob, blob._matchedCatalog, safeSnr)
    return blob


def _loadAndMatchCatalogs(repo, dataIds, matchRadius,
                          useJointCal=False, skipTEx=False):
    """Load data from specific visit. Match with reference.

    Parameters
    ----------
    repo : string or Butler
        A Butler or a repository URL that can be used to construct one
    dataIds : list of dict
        List of `butler` data IDs of Image catalogs to compare to
        reference. The `calexp` cpixel image is needed for the photometric
        calibration.
    matchRadius :  geom.Angle(), optional
        Radius for matching. Default is 1 arcsecond.
    useJointCal : `bool`, optional
        Use jointcal/meas_mosaic outputs to calibrate positions and fluxes.
    skipTEx : `bool`, optional
        Skip TEx calculations (useful for older catalogs that don't have
        PsfShape measurements).

    Returns
    -------
    catalog_list : afw.table.SourceCatalog
        List of all of the catalogs
    matched_catalog : afw.table.GroupView
        An object of matched catalog.
    """
    # Following
    # https://github.com/lsst/afw/blob/tickets/DM-3896/examples/repeatability.ipynb
    if isinstance(repo, dafPersist.Butler):
        butler = repo
    else:
        butler = dafPersist.Butler(repo)
    dataset = 'src'

    # 2016-02-08 MWV:
    # I feel like I could be doing something more efficient with
    # something along the lines of the following:
    #    dataRefs = [dafPersist.ButlerDataRef(butler, vId) for vId in dataIds]

    ccdKeyName = getCcdKeyName(dataIds[0])

    # Hack to support raft and sensor 0,1 IDs as ints for multimatch
    if ccdKeyName == 'sensor':
        ccdKeyName = 'raft_sensor_int'
        for vId in dataIds:
            vId[ccdKeyName] = raftSensorToInt(vId)

    schema = butler.get(dataset + "_schema").schema
    mapper = SchemaMapper(schema)
    mapper.addMinimalSchema(schema)
    mapper.addOutputField(Field[float]('base_PsfFlux_snr',
                                       'PSF flux SNR'))
    mapper.addOutputField(Field[float]('base_PsfFlux_mag',
                                       'PSF magnitude'))
    mapper.addOutputField(Field[float]('base_PsfFlux_magErr',
                                       'PSF magnitude uncertainty'))
    mapper.addOutputField(Field[float]('e1',
                                       'Source Ellipticity 1'))
    mapper.addOutputField(Field[float]('e2',
                                       'Source Ellipticity 1'))
    mapper.addOutputField(Field[float]('psf_e1',
                                       'PSF Ellipticity 1'))
    mapper.addOutputField(Field[float]('psf_e2',
                                       'PSF Ellipticity 1'))
    newSchema = mapper.getOutputSchema()
    newSchema.setAliasMap(schema.getAliasMap())

    # Create an object that matches multiple catalogs with same schema
    mmatch = MultiMatch(newSchema,
                        dataIdFormat={'visit': np.int32, ccdKeyName: np.int32},
                        radius=matchRadius,
                        RecordClass=SimpleRecord)

    # create the new extented source catalog
    srcVis = SourceCatalog(newSchema)

    for vId in dataIds:

        if useJointCal:
            try:
                photoCalib = butler.get("jointcal_photoCalib", vId)
            except (FitsError, dafPersist.NoResults) as e:
                print(e)
                print("Could not open photometric calibration for ", vId)
                print("Skipping this dataId.")
                continue
            try:
                wcs = butler.get("jointcal_wcs", vId)
            except (FitsError, dafPersist.NoResults) as e:
                print(e)
                print("Could not open updated WCS for ", vId)
                print("Skipping this dataId.")
                continue
        else:
            try:
                photoCalib = butler.get("calexp_photoCalib", vId)
            except (FitsError, dafPersist.NoResults) as e:
                print(e)
                print("Could not open calibrated image file for ", vId)
                print("Skipping this dataId.")
                continue
            except TypeError as te:
                # DECam images that haven't been properly reformatted
                # can trigger a TypeError because of a residual FITS header
                # LTV2 which is a float instead of the expected integer.
                # This generates an error of the form:
                #
                # lsst::pex::exceptions::TypeError: 'LTV2 has mismatched type'
                #
                # See, e.g., DM-2957 for details.
                print(te)
                print("Calibration image header information malformed.")
                print("Skipping this dataId.")
                continue

        # We don't want to put this above the first "if useJointCal block"
        # because we need to use the first `butler.get` above to quickly
        # catch data IDs with no usable outputs.
        try:
            # HSC supports these flags, which dramatically improve I/O
            # performance; support for other cameras is DM-6927.
            oldSrc = butler.get('src', vId, flags=SOURCE_IO_NO_FOOTPRINTS)
        except (OperationalError, sqlite3.OperationalError):
            oldSrc = butler.get('src', vId)

        print(len(oldSrc), "sources in ccd %s  visit %s" %
              (vId[ccdKeyName], vId["visit"]))

        # create temporary catalog
        tmpCat = SourceCatalog(SourceCatalog(newSchema).table)
        tmpCat.extend(oldSrc, mapper=mapper)
        tmpCat['base_PsfFlux_snr'][:] = tmpCat['base_PsfFlux_instFlux'] \
            / tmpCat['base_PsfFlux_instFluxErr']

        if useJointCal:
            for record in tmpCat:
                record.updateCoord(wcs)
        photoCalib.instFluxToMagnitude(tmpCat, "base_PsfFlux", "base_PsfFlux")

        if not skipTEx:
            _, psf_e1, psf_e2 = ellipticity_from_cat(oldSrc, slot_shape='slot_PsfShape')
            _, star_e1, star_e2 = ellipticity_from_cat(oldSrc, slot_shape='slot_Shape')
            tmpCat['e1'][:] = star_e1
            tmpCat['e2'][:] = star_e2
            tmpCat['psf_e1'][:] = psf_e1
            tmpCat['psf_e2'][:] = psf_e2

        srcVis.extend(tmpCat, False)
        mmatch.add(catalog=tmpCat, dataId=vId)

    # Complete the match, returning a catalog that includes
    # all matched sources with object IDs that can be used to group them.
    matchCat = mmatch.finish()

    # Create a mapping object that allows the matches to be manipulated
    # as a mapping of object ID to catalog of sources.
    allMatches = GroupView.build(matchCat)

    return srcVis, allMatches


def _reduceStars(blob, allMatches, safeSnr=50.0):
    """Calculate summary statistics for each star. These are persisted
    as object attributes.

    Parameters
    ----------
    allMatches : afw.table.GroupView
        GroupView object with matches.
    safeSnr : float, optional
        Minimum median SNR for a match to be considered "safe".
    """
    # Filter down to matches with at least 2 sources and good flags
    flagKeys = [allMatches.schema.find("base_PixelFlags_flag_%s" % flag).key
                for flag in ("saturated", "cr", "bad", "edge")]
    nMatchesRequired = 2

    psfSnrKey = allMatches.schema.find("base_PsfFlux_snr").key
    psfMagKey = allMatches.schema.find("base_PsfFlux_mag").key
    psfMagErrKey = allMatches.schema.find("base_PsfFlux_magErr").key
    extendedKey = allMatches.schema.find("base_ClassificationExtendedness_value").key

    def goodFilter(cat, goodSnr=3):
        if len(cat) < nMatchesRequired:
            return False
        for flagKey in flagKeys:
            if cat.get(flagKey).any():
                return False
        if not np.isfinite(cat.get(psfMagKey)).all():
            return False
        psfSnr = np.median(cat.get(psfSnrKey))
        # Note that this also implicitly checks for psfSnr being non-nan.
        return psfSnr >= goodSnr

    goodMatches = allMatches.where(goodFilter)

    # Filter further to a limited range in S/N and extendedness
    # to select bright stars.
    safeMaxExtended = 1.0

    def safeFilter(cat):
        psfSnr = np.median(cat.get(psfSnrKey))
        extended = np.max(cat.get(extendedKey))
        return psfSnr >= safeSnr and extended < safeMaxExtended

    safeMatches = goodMatches.where(safeFilter)

    # Pass field=psfMagKey so np.mean just gets that as its input
    filter_name = blob['filterName']
    blob['snr'] = Datum(quantity=goodMatches.aggregate(np.median, field=psfSnrKey) * u.Unit(''),
                        label='SNR({band})'.format(band=filter_name),
                        description='Median signal-to-noise ratio of PSF magnitudes over '
                                    'multiple visits')
    blob['mag'] = Datum(quantity=goodMatches.aggregate(np.mean, field=psfMagKey) * u.mag,
                        label='{band}'.format(band=filter_name),
                        description='Mean PSF magnitudes of stars over multiple visits')
    blob['magrms'] = Datum(quantity=goodMatches.aggregate(np.std, field=psfMagKey) * u.mag,
                           label='RMS({band})'.format(band=filter_name),
                           description='RMS of PSF magnitudes over multiple visits')
    blob['magerr'] = Datum(quantity=goodMatches.aggregate(np.median, field=psfMagErrKey) * u.mag,
                           label='sigma({band})'.format(band=filter_name),
                           description='Median 1-sigma uncertainty of PSF magnitudes over '
                                       'multiple visits')
    # positionRmsFromCat knows how to query a group
    # so we give it the whole thing by going with the default `field=None`.
    blob['dist'] = Datum(quantity=goodMatches.aggregate(positionRmsFromCat) * u.milliarcsecond,
                         label='d',
                         description='RMS of sky coordinates of stars over multiple visits')

    # These attributes are not serialized
    blob.goodMatches = goodMatches
    blob.safeMatches = safeMatches
