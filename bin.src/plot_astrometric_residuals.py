#!/Users/parejkoj/lsst/lsstsw/miniconda/envs/lsst-scipipe/bin/python  # noqa
"""Plot astrometric residuals for ccds in a tract.

Writes output to the current working directory, in `plots/` and `pickles/`,
which are created if they do not exist.

Originally written to make plots for John Parejko's AAS 233 poster.

Example to run on tract 8524 of HSC PDR1 on lsst-dev:

TRACT=8524
VISITS=7286^7288^7298^7300^7302^7304^7310^7338^7340^7350^7352^7356^7358^7364^7366^7370^7372^7378^7384^7386^7390^7392^7394^7396^7400^7402^7416^14124^14126^14128^14130^14142^14144^14146^14164^14166^14176^14178^14196^14198^14206^14208^14210^1623
CCDS=0..8^10..103
DATADIR=/project/parejkoj/hscRerun/DM-15713/WIDE
plot_astrometric_residuals.py $DATADIR --jointcal --id ccd=$CCDS visit=$VISITS tract=$TRACT

Or to run on jointcal's cfht testdata output in your local directory:

DATADIR=/Users/parejkoj/lsst/jointcal/jointcal/tests/.test/JointcalTestCFHT/test_jointcalTask_2_visits_constrainedAstrometry_no_photometry
TRACT=0
VISITS=849375^850587
CCDS=12^13^14^21^22^23
plot_astrometric_residuals.py $DATADIR --jointcal --id ccd=$CCDS visit=$VISITS tract=$TRACT
"""

import collections
import pickle
import os

import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('white')  # noqa: E402
seaborn.set_context("poster")  # noqa: E402

from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
import lsst.afw.table
import lsst.daf.persistence
import lsst.geom
import lsst.meas.astrom
import lsst.pex.config
import lsst.pipe.base
from lsst.validate.drp.util import positionRmsFromCat, averageRaDecFromCat


def filter_matches(allMatches, fluxField):
    """Filter down to matches with at least 2 sources and good flags."""
    flagKeys = [allMatches.schema.find("base_PixelFlags_flag_%s" % flag).key
                for flag in ("saturated", "cr", "bad", "edge")]
    nMatchesRequired = 2

    fluxSnrKey = allMatches.schema.find(fluxField + "_snr").key
    # NOTE: alias oddities mean I have to change "_flux" <-> "_instFlux" depending on catalog version.
    fluxKey = allMatches.schema.find(fluxField + "_flux").key

    def goodFilter(cat, goodSnr=10):
        if len(cat) < nMatchesRequired:
            return False
        for flagKey in flagKeys:
            if cat.get(flagKey).any():
                return False
        if not (cat.get(fluxKey) > 0).all():
            return False
        snr = np.median(cat.get(fluxSnrKey))
        # Note that this also implicitly checks for snr being non-nan.
        return snr >= goodSnr

    return allMatches.where(goodFilter)


def prep_matching(butler, dataRef):
    """Prep MultiMatch with the catalog schema and field types."""
    oldSchema = dataRef.get('src_schema').schema
    fluxField = oldSchema.getAliasMap().get("slot_CalibFlux")

    # make the new schema, with a field for S/N
    mapper = lsst.afw.table.SchemaMapper(oldSchema)
    mapper.addMinimalSchema(oldSchema)
    mapper.addOutputField(lsst.afw.table.Field[float](fluxField + '_snr', 'flux SNR'))
    newSchema = mapper.getOutputSchema()
    newSchema.setAliasMap(oldSchema.getAliasMap())

    types = {"visit": np.int32, "ccd": np.int32}
    multiMatch = lsst.afw.table.MultiMatch(newSchema, types)
    return fluxField, newSchema, mapper, multiMatch


def read_one(dataRef, useJointcal=False):
    # NOTE: I know we're not supposed to access the internals of dataRefs like this,
    # but I can't be bothered to do the "getDetector()" stuff, and you can't
    # get at a visit ID in any other way anyway...
    visit = dataRef.dataId['visit']
    ccd = dataRef.dataId['ccd']
    try:
        oldCat = dataRef.get('src')
    except lsst.daf.persistence.butlerExceptions.NoResults:
        # ignore missing data
        print("No data:", visit, ccd)
        return None
    if useJointcal:
        wcs = dataRef.get('jointcal_wcs')
        lsst.afw.table.updateSourceCoords(wcs, oldCat)

    return visit, ccd, oldCat


def do_match(multiMatch, butler, dataRefs, fluxField, newSchema, mapper, useJointcal=False):
    """Make the multiMatch, identify good matches, and compute aggregate statistics."""
    # import concurrent.futures
    # import itertools
    # max_workers = 1
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     mapped = executor.map(read_one, dataRefs, itertools.repeat(useJointcal))

    for dataRef in dataRefs:
        # for (visit, ccd, oldCat), dataRef in zip(mapped, dataRefs):
        visit, ccd, oldCat = read_one(dataRef, useJointcal)
        catalog = lsst.afw.table.SourceCatalog(newSchema)
        tmpCat = lsst.afw.table.SourceCatalog(lsst.afw.table.SourceCatalog(newSchema).table)
        tmpCat.extend(oldCat, mapper=mapper)
        tmpCat[fluxField + '_snr'][:] = tmpCat[fluxField + '_instFlux'] / tmpCat[fluxField + '_instFluxErr']
        catalog.extend(tmpCat, False)

        multiMatch.add(catalog, dataId=dataRef.dataId)
        print("Loaded:", visit, ccd, len(tmpCat))

    matchCat = multiMatch.finish()
    allMatches = lsst.afw.table.GroupView.build(matchCat)
    print("Found matches, groups:", len(matchCat), len(allMatches))

    goodMatches = filter_matches(allMatches, fluxField)
    print("Good groups:", len(goodMatches))

    averageCoord = goodMatches.aggregate(averageRaDecFromCat,
                                         dtype=[('ra', np.float64), ('dec', np.float64)])
    distance = goodMatches.aggregate(positionRmsFromCat) * u.milliarcsecond
    return goodMatches, averageCoord, distance


def count_ccds(goodMatches):
    """Count how many objects are on each ccd."""
    counts = collections.defaultdict(int)
    for group in goodMatches.groups:
        for x in group:
            counts[x['ccd']] += 1
    print("ccd counts:", ', '.join("%s: %s"%(k, v) for k, v in counts.items()))
    return counts


def compute_errors(counts, goodMatches, averageCoord, detector, focalplane=False):
    """Return the ra/dec error for each centroided object.

    If focalplane is set, return x,y in focal plane coordinates, instead of pixels.
    """
    detectorId = detector.getId()
    if focalplane:
        pixelToFocal = detector.getTransform(PIXELS, FOCAL_PLANE)

    xx = np.zeros(counts[detectorId])
    yy = np.zeros(counts[detectorId])
    uu = np.zeros(counts[detectorId])
    vv = np.zeros(counts[detectorId])
    i = 0

    for group, coord in zip(goodMatches.groups, averageCoord):
        good = group['ccd'] == detectorId
        n = good.sum()
        if focalplane:
            pixels = [lsst.geom.Point2D(x, y) for x, y in zip(group[good].getX(), group[good].getY())]
            focals = pixelToFocal.applyForward(pixels)
            xx[i:i + n] = [f.getX() for f in focals]
            yy[i:i + n] = [f.getY() for f in focals]
        else:
            xx[i:i + n] = group[good].getX()
            yy[i:i + n] = group[good].getY()
        uu[i:i + n] = group[good]['coord_ra'] - coord[0]
        vv[i:i + n] = group[good]['coord_dec'] - coord[1]
        i += n
    uu = (uu*u.radian).to_value(u.milliarcsecond)
    vv = (vv*u.radian).to_value(u.milliarcsecond)

    bbox = detector.getBBox()
    if focalplane:
        # convert the detector bbox to focal plane coordinates
        focalBBox = lsst.geom.Box2D()
        for point in pixelToFocal.applyForward(lsst.geom.Box2D(bbox).getCorners()):
            focalBBox.include(point)
        bbox = focalBBox

    return xx, yy, uu, vv, bbox


def uv_mean(xlim, ylim, xx, yy, uu, vv):
    """Compute the mean of uu and vv on a grid within bbox."""
    nx = 20
    ny = 40
    uMean = np.zeros((nx-1, ny-1))
    vMean = np.zeros((nx-1, ny-1))
    xMean = np.zeros((nx-1, ny-1))
    yMean = np.zeros((nx-1, ny-1))
    ww = np.linspace(xlim[0], xlim[1], nx)
    hh = np.linspace(ylim[0], ylim[1], ny)
    for i, (w0, w1) in enumerate(zip(ww[:-1], ww[1:])):
        inx = (xx >= w0) & (xx <= w1)
        for j, (h0, h1) in enumerate(zip(hh[:-1], hh[1:])):
            iny = (yy >= h0) & (yy <= h1)
            inside = inx & iny
            xMean[i, j] = (w0 + w1)/2
            yMean[i, j] = (h0 + h1)/2
            uMean[i, j] = np.mean(uu[inside])
            vMean[i, j] = np.mean(vv[inside])

    return xMean, yMean, uMean, vMean


def plot_quiver(xx, yy, uu, vv, ccd, label, focal):
    """Make a quiver plot of the astrometry error vectors."""
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    color = cycle[0] if 'jointcal' in label else cycle[1]
    scale = 0.25 if 'mean' in label else 1
    width = 7.0
    if focal:
        width = .1
        scale *= 70

    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(111)

    # units of the vectors are not, in fact "x", but it's simpler to treat them that way.
    # uu, vv should be in arcsec, xx,yy in pixels or mm (if focal)
    Q = ax.quiver(xx, yy, uu, vv, units='x', pivot='tail', scale=scale, width=width,
                  headwidth=4, clip_on=False, color=color)
    length = 5/scale if 'mean' in label else 100
    key = ax.quiverkey(Q, 0.85, 0.90, length, '%s mas'%(length), angle=45,
                       coordinates='figure', labelpos='W', fontproperties={'size': 24})

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title('{}'.format(ccd))
    filename = "plots/quiver-%s-%s.png"%(ccd, label)
    plt.savefig(filename, bbox_extra_artists=(key,))
    plt.close(fig)


def main():
    parser = lsst.pipe.base.InputOnlyArgumentParser("plot_astrometric_residuals",
                                                    description=__doc__)
    parser.add_argument("--jointcal", action='store_true',
                        help="Use the jointcal_wcs dataset to update each ccd's coordinates.")
    parser.add_argument("--focalplane", action='store_true',
                        help="Make plots and pickle files in focal plane coordinates.")
    # We need to specify "jointcal_wcs" here so that the butler understands what a `tract` is.
    parser.add_id_argument(name="--id", datasetType="jointcal_wcs",
                           help="data IDs, e.g. --id visit=12345 ccd=1^2 tract=1234")

    args = parser.parse_args(config=lsst.pex.config.Config())
    name = 'jointcal' if args.jointcal else 'single'
    butler = args.butler

    if len(args.id.refList) == 0:
        raise RuntimeError("No data to process! Check your id list and/or input data path.")
    fluxField, newSchema, mapper, multiMatch = prep_matching(butler, args.id.refList[0])
    goodMatches, averageCoord, distance = do_match(multiMatch, butler, args.id.refList,
                                                   fluxField, newSchema, mapper, useJointcal=args.jointcal)
    counts = count_ccds(goodMatches)

    os.makedirs('plots', exist_ok=True)
    os.makedirs('pickles', exist_ok=True)

    for ccd in counts.keys():
        detector = butler.get('calexp_detector', ccd=ccd)
        xx, yy, uu, vv, bbox = compute_errors(counts, goodMatches, averageCoord, detector,
                                              focalplane=args.focalplane)

        xlim = (bbox.getMinX(), bbox.getMaxX())
        ylim = (bbox.getMinY(), bbox.getMaxY())
        filename = "pickles/quiverData-%s-%s.pickle"%(name, ccd)
        with open(filename, 'wb') as outfile:
            pickle.dump((xx, yy, uu, vv, xlim, ylim, ccd),
                        outfile,
                        protocol=pickle.HIGHEST_PROTOCOL)

        plot_quiver(xx, yy, uu, vv, ccd, 'all-'+name, args.focalplane)
        xMean, yMean, uMean, vMean = uv_mean(xlim, ylim, xx, yy, uu, vv)
        plot_quiver(xMean, yMean, uMean, vMean, ccd, 'mean-'+name, args.focalplane)


if __name__ == "__main__":
    main()
