# LSST Data Management System
# Copyright 2016-2019 AURA/LSST.
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

__all__ = ['build_matched_dataset', 'getKeysFilter', 'filterSources', 'summarizeSources']

import numpy as np
import astropy.units as u
from sqlalchemy.exc import OperationalError
import sqlite3

import lsst.geom as geom
import lsst.daf.persistence as dafPersist
from lsst.afw.table import (SourceCatalog, SchemaMapper, Field,
                            MultiMatch, SimpleRecord, GroupView,
                            SOURCE_IO_NO_FOOTPRINTS)
import lsst.afw.table as afwTable
from lsst.afw.fits import FitsError
import lsst.pipe.base as pipeBase
from lsst.verify import Blob, Datum

from .util import (getCcdKeyName, raftSensorToInt, positionRmsFromCat,
                   ellipticity_from_cat)


def build_matched_dataset(repo, dataIds, matchRadius=None, brightSnrMin=None, brightSnrMax=None,
                          faintSnrMin=None, faintSnrMax=None,
                          doApplyExternalPhotoCalib=False, externalPhotoCalibName=None,
                          doApplyExternalSkyWcs=False, externalSkyWcsName=None,
                          skipTEx=False, skipNonSrd=False):
    """Construct a container for matched star catalogs from multple visits, with filtering,
    summary statistics, and modelling.

    `lsst.verify.Blob` instances are serializable to JSON.

    Parameters
    ----------
    repo : `str` or `lsst.daf.persistence.Butler`
        A Butler instance or a repository URL that can be used to construct
        one.
    dataIds : `list` of `dict`
        List of `butler` data IDs of Image catalogs to compare to reference.
        The `calexp` cpixel image is needed for the photometric calibration.
    matchRadius :  `lsst.geom.Angle`, optional
        Radius for matching. Default is 1 arcsecond.
    brightSnrMin : `float`, optional
        Minimum median SNR for a source to be considered bright; passed to `filterSources`.
    brightSnrMax : `float`, optional
        Maximum median SNR for a source to be considered bright; passed to `filterSources`.
    faintSnrMin : `float`, optional
        Minimum median SNR for a source to be considered faint; passed to `filterSources`.
    faintSnrMax : `float`, optional
        Maximum median SNR for a source to be considered faint; passed to `filterSources`.
    doApplyExternalPhotoCalib : bool, optional
        Apply external photoCalib to calibrate fluxes.
    externalPhotoCalibName : str, optional
        Type of external `PhotoCalib` to apply.  Currently supported are jointcal,
        fgcm, and fgcm_tract.  Must be set if "doApplyExternalPhotoCalib" is True.
    doApplyExternalSkyWcs : bool, optional
        Apply external wcs to calibrate positions.
    externalSkyWcsName : str, optional:
        Type of external `wcs` to apply.  Currently supported is jointcal.
        Must be set if "doApplyExternalSkyWcs" is True.
    skipTEx : `bool`, optional
        Skip TEx calculations (useful for older catalogs that don't have
        PsfShape measurements).
    skipNonSrd : `bool`, optional
        Skip any metrics not defined in the LSST SRD; default False.

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
    matchesFaint : `afw.table.GroupView`
        Faint matches containing only objects that have:

        1. A PSF Flux measurement with sufficient S/N.
        2. A finite (non-nan) PSF magnitude. This separate check is largely
           to reject failed zeropoints.
        3. No flags set for bad, cosmic ray, edge or saturated.
        4. Extendedness consistent with a point source.

        *Not serialized.*
    matchesBright : `afw.table.GroupView`
        Bright matches matching a higher S/N threshold than matchesFaint.

        *Not serialized.*
    magKey
        Key for `"base_PsfFlux_mag"` in the `matchesFaint` and `matchesBright`
        catalog tables.

        *Not serialized.*

    Raises
    ------
    RuntimeError:
        Raised if "doApplyExternalPhotoCalib" is True and "externalPhotoCalibName"
        is None, or if "doApplyExternalSkyWcs" is True and "externalSkyWcsName" is
        None.
    """
    if doApplyExternalPhotoCalib and externalPhotoCalibName is None:
        raise RuntimeError("Must set externalPhotoCalibName if doApplyExternalPhotoCalib is True.")
    if doApplyExternalSkyWcs and externalSkyWcsName is None:
        raise RuntimeError("Must set externalSkyWcsName if doApplyExternalSkyWcs is True.")

    blob = Blob('MatchedMultiVisitDataset')

    if not matchRadius:
        matchRadius = geom.Angle(1, geom.arcseconds)

    # Extract single filter
    blob['filterName'] = Datum(quantity=set([dId['filter'] for dId in dataIds]).pop(),
                               description='Filter name')

    # Record important configuration
    blob['doApplyExternalPhotoCalib'] = Datum(quantity=doApplyExternalPhotoCalib,
                                              description=('Whether external photometric '
                                                           'calibrations were used.'))
    blob['externalPhotoCalibName'] = Datum(quantity=externalPhotoCalibName,
                                           description='Name of external PhotoCalib dataset used.')
    blob['doApplyExternalSkyWcs'] = Datum(quantity=doApplyExternalSkyWcs,
                                          description='Whether external wcs calibrations were used.')
    blob['externalSkyWcsName'] = Datum(quantity=externalSkyWcsName,
                                       description='Name of external wcs dataset used.')

    # Match catalogs across visits
    blob._catalog, blob._matchedCatalog = \
        _loadAndMatchCatalogs(repo, dataIds, matchRadius,
                              doApplyExternalPhotoCalib=doApplyExternalPhotoCalib,
                              externalPhotoCalibName=externalPhotoCalibName,
                              doApplyExternalSkyWcs=doApplyExternalSkyWcs,
                              externalSkyWcsName=externalSkyWcsName,
                              skipTEx=skipTEx, skipNonSrd=skipNonSrd)

    blob.magKey = blob._matchedCatalog.schema.find("base_PsfFlux_mag").key
    # Reduce catalogs into summary statistics.
    # These are the serializable attributes of this class.
    filterResult = filterSources(
        blob._matchedCatalog, brightSnrMin=brightSnrMin, brightSnrMax=brightSnrMax,
        faintSnrMin=faintSnrMin, faintSnrMax=faintSnrMax,
    )
    blob['brightSnrMin'] = Datum(quantity=filterResult.brightSnrMin * u.Unit(''),
                                 label='Bright SNR Min',
                                 description='Minimum median SNR for a source to be considered bright')
    blob['brightSnrMax'] = Datum(quantity=filterResult.brightSnrMax * u.Unit(''),
                                 label='Bright SNR Max',
                                 description='Maximum median SNR for a source to be considered bright')
    summarizeSources(blob, filterResult)

    # import pdb ; pdb.set_trace()

    return blob


def _loadAndMatchCatalogs(repo, dataIds, matchRadius,
                          doApplyExternalPhotoCalib=False, externalPhotoCalibName=None,
                          doApplyExternalSkyWcs=False, externalSkyWcsName=None,
                          skipTEx=False, skipNonSrd=False):
    """Load data from specific visits and returned a calibrated catalog matched
    with a reference.

    Parameters
    ----------
    repo : `str` or `lsst.daf.persistence.Butler`
        A Butler or a repository URL that can be used to construct one.
    dataIds : list of dict
        List of butler data IDs of Image catalogs to compare to
        reference. The calexp cpixel image is needed for the photometric
        calibration.
    matchRadius :  `lsst.geom.Angle`, optional
        Radius for matching. Default is 1 arcsecond.
    doApplyExternalPhotoCalib : bool, optional
        Apply external photoCalib to calibrate fluxes.
    externalPhotoCalibName : str, optional
        Type of external `PhotoCalib` to apply.  Currently supported are jointcal,
        fgcm, and fgcm_tract.  Must be set if doApplyExternalPhotoCalib is True.
    doApplyExternalSkyWcs : bool, optional
        Apply external wcs to calibrate positions.
    externalSkyWcsName : str, optional
        Type of external `wcs` to apply.  Currently supported is jointcal.
        Must be set if "doApplyExternalWcs" is True.
    skipTEx : `bool`, optional
        Skip TEx calculations (useful for older catalogs that don't have
        PsfShape measurements).
    skipNonSrd : `bool`, optional
        Skip any metrics not defined in the LSST SRD; default False.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        A new calibrated SourceCatalog.
    matches : `lsst.afw.table.GroupView`
        A GroupView of the matched sources.

    Raises
    ------
    RuntimeError:
        Raised if "doApplyExternalPhotoCalib" is True and "externalPhotoCalibName"
        is None, or if "doApplyExternalSkyWcs" is True and "externalSkyWcsName" is
        None.
    """

    if doApplyExternalPhotoCalib and externalPhotoCalibName is None:
        raise RuntimeError("Must set externalPhotoCalibName if doApplyExternalPhotoCalib is True.")
    if doApplyExternalSkyWcs and externalSkyWcsName is None:
        raise RuntimeError("Must set externalSkyWcsName if doApplyExternalSkyWcs is True.")

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
    if not skipNonSrd:
        # Needed because addOutputField(... 'slot_ModelFlux_mag') will add a field with that literal name
        aliasMap = schema.getAliasMap()
        # Possibly not needed since base_GaussianFlux is the default, but this ought to be safe
        modelName = aliasMap['slot_ModelFlux'] if 'slot_ModelFlux' in aliasMap.keys() else 'base_GaussianFlux'
        mapper.addOutputField(Field[float](f'{modelName}_mag',
                                           'Model magnitude'))
        mapper.addOutputField(Field[float](f'{modelName}_magErr',
                                           'Model magnitude uncertainty'))
        mapper.addOutputField(Field[float](f'{modelName}_snr',
                                           'Model flux snr'))
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
        if not butler.datasetExists('src', vId):
            print(f'Could not find source catalog for {vId}; skipping.')
            continue

        photoCalib = _loadPhotoCalib(butler, vId,
                                     doApplyExternalPhotoCalib, externalPhotoCalibName)
        if photoCalib is None:
            continue

        if doApplyExternalSkyWcs:
            wcs = _loadExternalSkyWcs(butler, vId, externalSkyWcsName)
            if wcs is None:
                continue

        # We don't want to put this above the first _loadPhotoCalib call
        # because we need to use the first `butler.get` in there to quickly
        # catch dataIDs with no usable outputs.
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

        if doApplyExternalSkyWcs:
            afwTable.updateSourceCoords(wcs, tmpCat)
        photoCalib.instFluxToMagnitude(tmpCat, "base_PsfFlux", "base_PsfFlux")
        if not skipNonSrd:
            tmpCat['slot_ModelFlux_snr'][:] = (tmpCat['slot_ModelFlux_instFlux'] /
                                               tmpCat['slot_ModelFlux_instFluxErr'])
            photoCalib.instFluxToMagnitude(tmpCat, "slot_ModelFlux", "slot_ModelFlux")

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

    match_filename = "matchedCat_%s.fits" % vId['filter']
    matchCat.writeFits(match_filename)
    print('Wrote matched catalog to file %s' % match_filename)

    # import pdb ; pdb.set_trace()

    # Create a mapping object that allows the matches to be manipulated
    # as a mapping of object ID to catalog of sources.
    allMatches = GroupView.build(matchCat)

    return srcVis, allMatches


def getKeysFilter(schema, nameFluxKey=None):
    """ Get schema keys for filtering sources.

    schema : `lsst.afw.table.Schema`
        A table schema to retrieve keys from.
    nameFluxKey : `str`
        The name of a flux field to retrieve

    Returns
    -------
    keys : `lsst.pipe.base.Struct`
        A struct storing schema keys to aggregate over.
    """
    if nameFluxKey is None:
        nameFluxKey = "base_PsfFlux"
        # Filter down to matches with at least 2 sources and good flags

    return pipeBase.Struct(
        flags=[schema.find("base_PixelFlags_flag_%s" % flag).key
               for flag in ("saturated", "cr", "bad", "edge")],
        snr=schema.find(f"{nameFluxKey}_snr").key,
        mag=schema.find(f"{nameFluxKey}_mag").key,
        magErr=schema.find(f"{nameFluxKey}_magErr").key,
        extended=schema.find("base_ClassificationExtendedness_value").key,
    )


def filterSources(allMatches, keys=None, faintSnrMin=None, brightSnrMin=None, safeExtendedness=None,
                  extended=False, faintSnrMax=None, brightSnrMax=None):
    """Filter matched sources on flags and SNR.

    Parameters
    ----------
    allMatches : `lsst.afw.table.GroupView`
        GroupView object with matches.
    keys : `lsst.pipe.base.Struct`
        A struct storing schema keys to aggregate over.
    faintSnrMin : float, optional
        Minimum median SNR for a faint source match; default 5.
    brightSnrMin : float, optional
        Minimum median SNR for a bright source match; default 50.
    safeExtendedness: float, optional
        Maximum (exclusive) extendedness for sources or minimum (inclusive) if extended==True; default 1.
    extended: bool, optional
        Whether to select extended sources, i.e. galaxies.
    faintSnrMax : float, optional
        Maximum median SNR for a faint source match; default `numpy.Inf`.
    brightSnrMax : float, optional
        Maximum median SNR for a bright source match; default `numpy.Inf`.

    Returns
    -------
    filterResult : `lsst.pipe.base.Struct`
        A struct containing good and safe matches and the necessary keys to use them.
    """
    if brightSnrMin is None:
        brightSnrMin = 50
    if brightSnrMax is None:
        brightSnrMax = np.Inf
    if faintSnrMin is None:
        faintSnrMin = 5
    if faintSnrMax is None:
        faintSnrMax = np.Inf
    if safeExtendedness is None:
        safeExtendedness = 1.0
    if keys is None:
        keys = getKeysFilter(allMatches.schema, "slot_ModelFlux" if extended else "base_PsfFlux")
    nMatchesRequired = 2
    snrMin, snrMax = faintSnrMin, faintSnrMax

    def extendedFilter(cat):
        if len(cat) < nMatchesRequired:
            return False
        for flagKey in keys.flags:
            if cat.get(flagKey).any():
                return False
        if not np.isfinite(cat.get(keys.mag)).all():
            return False
        extendedness = cat.get(keys.extended)
        return np.min(extendedness) >= safeExtendedness if extended else \
            np.max(extendedness) < safeExtendedness

    def snrFilter(cat):
        # Note that this also implicitly checks for psfSnr being non-nan.
        snr = np.median(cat.get(keys.snr))
        return snrMax >= snr >= snrMin

    def fullFilter(cat):
        return extendedFilter(cat) and snrFilter(cat)

    # If brightSnrMin range is a subset of faintSnrMin, it's safe to only filter on snr again
    # Otherwise, filter on flags/extendedness first, then snr
    isSafeSubset = faintSnrMax >= brightSnrMax and faintSnrMin <= brightSnrMin
    matchesFaint = allMatches.where(fullFilter) if isSafeSubset else allMatches.where(extendedFilter)
    snrMin, snrMax = brightSnrMin, brightSnrMax
    matchesBright = matchesFaint.where(snrFilter)
    # This means that matchesFaint has had extendedFilter but not snrFilter applied
    if not isSafeSubset:
        snrMin, snrMax = faintSnrMin, faintSnrMax
        matchesFaint = matchesFaint.where(snrFilter)

    return pipeBase.Struct(
        extended=extended, keys=keys, matchesFaint=matchesFaint, matchesBright=matchesBright,
        brightSnrMin=brightSnrMin, brightSnrMax=brightSnrMax,
        faintSnrMin=faintSnrMin, faintSnrMax=faintSnrMax,
    )


def summarizeSources(blob, filterResult):
    """Calculate summary statistics for each source. These are persisted
    as object attributes.

    Parameters
    ----------
    blob : `lsst.verify.blob.Blob`
        A verification blob to store Datums in.
    filterResult : `lsst.pipe.base.Struct`
        A struct containing bright and faint filter matches, as returned by `filterSources`.
    """
    # Pass field=psfMagKey so np.mean just gets that as its input
    typeMag = "model" if filterResult.extended else "PSF"
    filter_name = blob['filterName']
    source_type = f'{"extended" if filterResult.extended else "point"} sources"'
    matches = filterResult.matchesFaint
    keys = filterResult.keys
    blob['snr'] = Datum(quantity=matches.aggregate(np.median, field=keys.snr) * u.Unit(''),
                        label='SNR({band})'.format(band=filter_name),
                        description=f'Median signal-to-noise ratio of {typeMag} magnitudes for {source_type}'
                                    f' over multiple visits')
    blob['mag'] = Datum(quantity=matches.aggregate(np.mean, field=keys.mag) * u.mag,
                        label='{band}'.format(band=filter_name),
                        description=f'Mean of {typeMag} magnitudes for {source_type} over multiple visits')
    blob['magrms'] = Datum(quantity=matches.aggregate(np.std, field=keys.mag) * u.mag,
                           label='RMS({band})'.format(band=filter_name),
                           description=f'RMS of {typeMag} magnitudes for {source_type} over multiple visits')
    blob['magerr'] = Datum(quantity=matches.aggregate(np.median, field=keys.magErr) * u.mag,
                           label='sigma({band})'.format(band=filter_name),
                           description=f'Median 1-sigma uncertainty of {typeMag} magnitudes for {source_type}'
                                       f' over multiple visits')
    # positionRmsFromCat knows how to query a group
    # so we give it the whole thing by going with the default `field=None`.
    blob['dist'] = Datum(quantity=matches.aggregate(positionRmsFromCat) * u.milliarcsecond,
                         label='d',
                         description=f'RMS of sky coordinates of {source_type} over multiple visits')

    # These attributes are not serialized
    blob.matchesFaint = filterResult.matchesFaint
    blob.matchesBright = filterResult.matchesBright


def _loadPhotoCalib(butler, dataId, doApplyExternalPhotoCalib, externalPhotoCalibName):
    """
    Load a photoCalib object.

    Parameters
    ----------
    butler: `lsst.daf.persistence.Butler`
    dataId: Butler dataId `dict`
    doApplyExternalPhotoCalib: `bool`
        Apply external photoCalib to calibrate fluxes.
    externalPhotoCalibName: `str`
        Type of external `PhotoCalib` to apply.  Currently supported are jointcal,
        fgcm, and fgcm_tract.  Must be set if "doApplyExternalPhotoCalib" is True.

    Returns
    -------
    photoCalib: `lsst.afw.image.PhotoCalib` or None
        photoCalib to apply.  None if a suitable one was not found.
    """

    photoCalib = None

    if doApplyExternalPhotoCalib:
        try:
            photoCalib = butler.get(f"{externalPhotoCalibName}_photoCalib", dataId)
        except (FitsError, dafPersist.NoResults) as e:
            print(e)
            print(f'Could not open external photometric calib for {dataId}; skipping.')
            photoCalib = None
    else:
        try:
            photoCalib = butler.get('calexp_photoCalib', dataId)
        except (FitsError, dafPersist.NoResults) as e:
            print(e)
            print(f'Could not open calibrated image file for {dataId}; skipping.')
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
            print(f'Calibration image header information malformed for {dataId}; skipping.')
            photoCalib = None

    return photoCalib


def _loadExternalSkyWcs(butler, dataId, externalSkyWcsName):
    """
    Load a SkyWcs object.

    Parameters
    ----------
    butler: `lsst.daf.persistence.Butler`
    dataId: Butler dataId `dict`
    externalSkyWcsName: `str`
        Type of external `SkyWcs` to apply.  Currently supported is jointcal.
        Must be not None if "doApplyExternalSkyWcs" is True.

    Returns
    -------
    SkyWcs: `lsst.afw.geom.SkyWcs` or None
        SkyWcs to apply.  None if a suitable one was not found.
    """

    try:
        wcs = butler.get(f"{externalSkyWcsName}_wcs", dataId)
    except (FitsError, dafPersist.NoResults) as e:
        print(e)
        print(f'Could not open external WCS for {dataId}; skipping.')
        wcs = None

    return wcs
