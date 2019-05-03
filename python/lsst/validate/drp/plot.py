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
"""Matplotlib plots describing lsst.validate.drp metric measurements, as well
as analytic models of photometric and astrometric repeatability.
"""

__all__ = ['plotOutlinedAxline',
           'plotAstrometryErrorModel',
           'plotAstromErrModelFit', 'plotPhotErrModelFit',
           'plotPhotometryErrorModel', 'plotPA1', 'plotAMx']


import matplotlib.pylab as plt
import numpy as np
import astropy.units as u
import scipy.stats
from .astromerrmodel import astromErrModel
from .photerrmodel import photErrModel
from lsst.verify import Name


# Plotting defaults
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['figure.titlesize'] = 30

color = {'all': 'grey', 'bright': 'blue',
         'iqr': 'green', 'rms': 'red'}


def makeFilename(prefix, formatStr, **kwargs):
    """Return a filename for writing to.

    Return prefix_formatStr.format(kwargs) if prefix is not empty, otherwise
    just return formatStr.format(kwargs).
    """
    formatted = formatStr.format(**kwargs)
    if prefix is None or prefix == "":
        return formatted
    else:
        return "{}_{}".format(prefix, formatted)


def plotOutlinedAxline(axMethod, x, **kwargs):
    """Plot an axis line with a white shadow for better contrast.

    Parameters
    ----------
    axMethod : `matplotlib.pyplot.axhline` or `matplotlib.pyplot.axvline`
        A horizontal or vertical axis line plotting function.
    x : float
        Axis coordinate
    **kwargs :
        Keyword arguments for `~matplotlib.pyplot.axhline` or
        `~matplotlib.pyplot.axvline`.
    """
    shadowArgs = dict(kwargs)
    foregroundArgs = dict(kwargs)

    if 'linewidth' not in foregroundArgs:
        foregroundArgs['linewidth'] = 3

    if 'linewidth' in shadowArgs:
        shadowArgs['linewidth'] += 1
    else:
        shadowArgs['linewidth'] = 4
    shadowArgs['color'] = 'w'
    shadowArgs['label'] = None

    axMethod(x, **shadowArgs)
    axMethod(x, **foregroundArgs)


def plotAstrometryErrorModel(dataset, astromModel, outputPrefix='',
                             showTitle=True, plotExt='png'):
    """Plot angular distance between matched sources from different exposures.

    Creates a file containing the plot with a filename beginning with
    `outputPrefix`.

    Parameters
    ----------
    dataset : `lsst.verify.Blob`
        Blob with the multi-visit photometry model.
    photomModel : `lsst.verify.Blob`
        A `Blob` containing the analytic photometry model.
    outputPrefix : str, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., ``outputPrefix='Cfht_output_r_'`` will result in a file
        named ``'Cfht_output_r_check_astrometry.png'``.
    """
    bright, = np.where(dataset['snr'].quantity > astromModel['brightSnr'].quantity)

    dist = dataset['dist'].quantity
    numMatched = len(dist)
    dist_median = np.median(dist)
    bright_dist_median = np.median(dist[bright])

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18, 12))

    ax[0].hist(dist, bins=100, color=color['all'],
               histtype='stepfilled', orientation='horizontal')
    ax[0].hist(dist[bright], bins=100, color=color['bright'],
               histtype='stepfilled', orientation='horizontal')

    ax[0].set_ylim([0., 500.])
    ax[0].set_ylabel("Distance [{unit:latex}]".format(unit=dist.unit))
    plotOutlinedAxline(
        ax[0].axhline,
        dist_median.value,
        color=color['all'],
        label="Median RMS: {v.value:.1f} {v.unit:latex}".format(v=dist_median))
    plotOutlinedAxline(
        ax[0].axhline,
        bright_dist_median.value,
        color=color['bright'],
        label="SNR > {snr:.0f}\nMedian RMS: {v.value:.1f} {v.unit:latex}".format(
            snr=astromModel['brightSnr'].quantity.value,
            v=bright_dist_median))
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel('#/bin')

    snr = dataset['snr'].quantity
    ax[1].scatter(snr, dist,
                  s=10, color=color['all'], label='All')
    ax[1].scatter(snr[bright], dist[bright], s=10,
                  color=color['bright'],
                  label='SNR > {0:.0f}'.format(astromModel['brightSnr'].quantity.value))
    ax[1].set_xlabel("SNR")
    ax[1].set_xscale("log")
    ax[1].set_ylim([0., 500.])
    matchCountTemplate = '\n'.join([
        'Matches:',
        '{nBright:d} high SNR,',
        '{nAll:d} total'])
    ax[1].text(0.6, 0.6, matchCountTemplate.format(nBright=len(bright),
                                                   nAll=numMatched),
               transform=ax[1].transAxes, ha='left', va='baseline')

    w, = np.where(dist < 200 * u.marcsec)
    plotAstromErrModelFit(snr[w], dist[w], astromModel,
                          ax=ax[1])

    ax[1].legend(loc='upper right')
    ax[1].axvline(astromModel['brightSnr'].quantity,
                  color='red', linewidth=4, linestyle='dashed')
    plotOutlinedAxline(
        ax[0].axhline,
        dist_median.value,
        color=color['all'])
    plotOutlinedAxline(
        ax[0].axhline,
        bright_dist_median.value,
        color=color['bright'])

    # Using title rather than suptitle because I can't get the top padding
    if showTitle:
        plt.suptitle("Astrometry Check : %s" % outputPrefix,
                     fontsize=30)

    pathFormat = "{name}.{ext}"
    plotPath = makeFilename(outputPrefix, pathFormat, name="check_astrometry", ext=plotExt)
    plt.savefig(plotPath, format=plotExt)
    plt.close(fig)
    print("Wrote plot:", plotPath)


def plotAstromErrModelFit(snr, dist, model,
                          color='red', ax=None, verbose=True):
    """Plot model of photometric error from LSST Overview paper
    http://arxiv.org/abs/0805.2366v4

    Astrometric Errors
    error = C * theta / SNR

    Parameters
    ----------
    snr : list or numpy.array
        S/N of photometric measurements
    dist : list or numpy.array
        Separation from reference [mas]
    model : `lsst.verify.Blob`
        A `Blob` holding the analytic astrometric model.
    """
    if ax is None:
        ax = plt.figure()
        xlim = [10, 30]
    else:
        xlim = ax.get_xlim()

    x_model = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), num=100)
    fit_model_mas_err = astromErrModel(x_model,
                                       theta=model['theta'].quantity,
                                       sigmaSys=model['sigmaSys'].quantity,
                                       C=model['C'].quantity)
    ax.plot(x_model, fit_model_mas_err,
            color=color, linewidth=2,
            label='Model')

    modelLabelTemplate = '\n'.join([
        r'$C = {C:.2g}$',
        r'$\theta$ = {theta:.4g}',
        r'$\sigma_\mathrm{{sys}}$ = {sigmaSys.value:.2g} {sigmaSys.unit:latex}'])
    modelLabel = modelLabelTemplate.format(
        C=model['C'].quantity,
        theta=model['theta'].quantity,
        sigmaSys=model['sigmaSys'].quantity)
    ax.text(0.6, 0.4, modelLabel,
            transform=ax.transAxes, va='baseline', ha='left', color=color)
    # Set the x limits back to their original values.
    ax.set_xlim(xlim)


def plotPhotErrModelFit(mag, mmag_err, photomModel, color='red', ax=None,
                        verbose=True):
    """Plot model of photometric error from LSST Overview paper (Eq. 4 & 5)

    Parameters
    ----------
    mag : list or numpy.array
        Magnitude
    mmag_err : list or numpy.array
        Magnitude uncertainty or variation in *mmag*.
    photomModel : `lsst.verify.Blob`
        A `Blob` holding the parameters to display.
    ax : matplotlib.Axis, optional
        The Axis object to plot to.
    verbose : bool, optional
        Produce extra output to STDOUT
    """

    if ax is None:
        ax = plt.figure()
        xlim = [10, 30]
    else:
        xlim = ax.get_xlim()

    x_model = np.linspace(*xlim, num=100)
    fit_model_mag_err = photErrModel(x_model,
                                     sigmaSys=photomModel['sigmaSys'].quantity.to(u.mag).value,
                                     gamma=photomModel['gamma'].quantity.value,
                                     m5=photomModel['m5'].quantity.to(u.mag).value)
    fit_model_mag_err = fit_model_mag_err * u.mag
    ax.plot(x_model, fit_model_mag_err.to(u.mmag).value,
            color=color, linewidth=2,
            label='Model')

    labelFormatStr = '\n'.join([
        r'$\sigma_\mathrm{{sys}}$ = {sigmaSysMmag:.4f} mmag',
        r'$\gamma = {gamma:.4f}$',
        r'$m_5 =$ {m5:.4f}'])
    label = labelFormatStr.format(sigmaSysMmag=1000*photomModel['sigmaSys'].quantity.to(u.mag).value,
                                  gamma=photomModel['gamma'].quantity.value,
                                  m5=photomModel['m5'].quantity.value)
    ax.text(0.1, 0.8, label, color=color,
            transform=ax.transAxes, ha='left', va='top')


def plotPhotometryErrorModel(dataset, photomModel,
                             filterName='', outputPrefix='',
                             showTitle=True, plotExt='png'):
    """Plot photometric RMS for matched sources.

    Parameters
    ----------
    dataset : `lsst.verify.Blob`
        A `Blob` with the multi-visit photometry model.
    photomModel : `lsst.verify.Blob`
        A `Blob` hlding the analytic photometry model parameters.
    filterName : str, optional
        Name of the observed filter to use on axis labels.
    outputPrefix : str, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., ``outputPrefix='Cfht_output_r_'`` will result in a file
        named ``'Cfht_output_r_check_photometry.png'``.
    """
    bright, = np.where(dataset['snr'].quantity > photomModel['brightSnr'].quantity)

    numMatched = len(dataset['mag'].quantity)
    magrms = dataset['magrms'].quantity
    mmagRms = magrms.to(u.mmag)
    mmagRmsHighSnr = mmagRms[bright]
    magerr = dataset['magerr'].quantity
    mmagErr = magerr.to(u.mmag)
    mmagErrHighSnr = mmagErr[bright]

    mmagrms_median = np.median(mmagRms)
    bright_mmagrms_median = np.median(mmagRmsHighSnr)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(18, 16))

    ax[0][0].hist(mmagRms,
                  bins=100, range=(0, 500), color=color['all'],
                  histtype='stepfilled', orientation='horizontal')
    ax[0][0].hist(mmagRmsHighSnr,
                  bins=100, range=(0, 500),
                  color=color['bright'],
                  histtype='stepfilled', orientation='horizontal')
    plotOutlinedAxline(
        ax[0][0].axhline,
        mmagrms_median.value,
        color=color['all'],
        label="Median RMS: {v:.1f}".format(v=mmagrms_median))
    plotOutlinedAxline(
        ax[0][0].axhline,
        bright_mmagrms_median.value,
        color=color['bright'],
        label="SNR > {snr:.0f}\nMedian RMS: {v:.1f}".format(
            snr=photomModel['brightSnr'].quantity.value,
            v=bright_mmagrms_median))

    ax[0][0].set_ylim([0, 500])
    ax[0][0].set_ylabel("{label}) [{mmagrms.unit:latex}]".format(
        label=dataset['magrms'].label.split('\n')[0], mmagrms=mmagRms))
    ax[0][0].legend(loc='upper right')
    # HACK ALERT
    ax[0][0].set_xlabel('#/bin')

    mag = dataset['mag'].quantity
    ax[0][1].scatter(mag, mmagRms,
                     s=10, color=color['all'], label='All')
    ax[0][1].scatter(mag[bright], mmagRmsHighSnr,
                     s=10, color=color['bright'],
                     label='{label} > {value:.0f}'.format(
                         label=photomModel['brightSnr'].label,
                         value=photomModel['brightSnr'].quantity.value))
    ax[0][1].set_xlabel("{label} [{unit:latex}]".format(
#       label=filterName,
        label='r',
        unit=mag.unit))
    ax[0][1].set_ylabel("{label}) [{unit:latex}]".format(label=dataset['magrms'].label.split('\n')[0],
                                                        unit=mmagRmsHighSnr.unit))
    ax[0][1].set_xlim([17, 24])
    ax[0][1].set_ylim([0, 500])
    ax[0][1].legend(loc='upper left')
    plotOutlinedAxline(
        ax[0][1].axhline,
        mmagrms_median.value,
        color=color['all'])
    plotOutlinedAxline(
        ax[0][1].axhline,
        bright_mmagrms_median.value,
        color=color['bright'])
    matchCountTemplate = '\n'.join([
        'Matches:',
        '{nBright:d} high SNR,',
        '{nAll:d} total'])
    ax[0][1].text(0.1, 0.6, matchCountTemplate.format(nBright=len(bright),
                                                      nAll=numMatched),
                  transform=ax[0][1].transAxes, ha='left', va='top')

    ax[1][0].scatter(mmagRms, mmagErr,
                     s=10, color=color['all'], label=None)
    ax[1][0].scatter(mmagRmsHighSnr, mmagErrHighSnr,
                     s=10, color=color['bright'],
                     label=None)
    ax[1][0].set_xscale('log')
    ax[1][0].set_yscale('log')
    ax[1][0].plot([0, 1000], [0, 1000],
                  linestyle='--', color='black', linewidth=2)
    ax[1][0].set_xlabel("{label}) [{unit:latex}]".format(
        label=dataset['magrms'].label.split('\n')[0],
        unit=mmagRms.unit))
    ax[1][0].set_ylabel("Median Reported Magnitude Err [{unit:latex}]".format(
        unit=mmagErr.unit))

    brightSnrMag = 2.5*np.log10(1 + (1/photomModel['brightSnr'].quantity.value)) * u.mag
    label = r'$SNR > {snr:.0f} \equiv \sigma <  {snrMag:0.1f}$'.format(
        snr=photomModel['brightSnr'].quantity.value,
        snrMag=brightSnrMag.to(u.mmag))
    ax[1][0].axhline(brightSnrMag.to(u.mmag).value,
                     color='red', linewidth=4,
                     linestyle='dashed',
                     label=label)
    ax[1][0].set_xlim([1, 500])
    ax[1][0].set_ylim([1, 500])
    ax[1][0].legend(loc='upper center')

    ax[1][1].scatter(mag, mmagErr,
                     color=color['all'], label=None)
    ax[1][1].set_yscale('log')
    ax[1][1].scatter(np.asarray(mag)[bright],
                     mmagErrHighSnr,
                     s=10, color=color['bright'],
                     label=None)
    ax[1][1].set_xlabel("{label} [{unit:latex}]".format(
#        label=filterName,
        label='r',
        unit=mag.unit))
    ax[1][1].set_ylabel("Median Reported Magnitude Err [{unit:latex}]".format(
        unit=mmagErr.unit))
    ax[1][1].set_xlim([17, 24])
    ax[1][1].set_ylim([1, 500])
    ax[1][1].axhline(brightSnrMag.to(u.mmag).value,
                     color='red', linewidth=4,
                     linestyle='dashed',
                     label=None)

    w, = np.where(mmagErr < 200. * u.mmag)
    plotPhotErrModelFit(mag[w].to(u.mag).value,
                        magerr[w].to(u.mmag).value,
                        photomModel, ax=ax[1][1])
    ax[1][1].legend(loc='upper left')

    if showTitle:
        plt.suptitle("Photometry Check : %s" % outputPrefix,
                     fontsize=30)

    pathFormat = "{name}.{ext}"
    plotPath = makeFilename(outputPrefix, pathFormat, name="check_photometry", ext=plotExt)
    plt.savefig(plotPath, format=plotExt)
    plt.close(fig)
    print("Wrote plot:", plotPath)


def plotPA1(pa1, outputPrefix="", showTitle=True, plotExt='png'):
    """Plot the results of calculating the LSST SRC requirement PA1.

    Creates a file containing the plot with a filename beginning with
    `outputPrefix`.

    Parameters
    ----------
    pa1 : `lsst.verify.Measurement`
        A `Measurement` of the PA1 `Metric`.
    outputPrefix : `str`, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., outputPrefix='Cfht_output_r_' will result in a file
        named ``'Cfht_output_r_AM1_D_5_arcmin_17.0-21.5.png'``
        for an ``AMx.name=='AM1'`` and ``AMx.magRange==[17, 21.5]``.
    """
    diffRange = (-100, +100)
    magDiff = pa1.extras['magDiff'].quantity
    magMean = pa1.extras['magMean'].quantity
    rms = pa1.extras['rms'].quantity
    iqr = pa1.extras['iqr'].quantity

    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(magMean[0],
                magDiff[0],
                s=10, color=color['bright'], linewidth=0)
    # index 0 because we show only the first sample from multiple trials
    ax1.axhline(+rms[0].value, color=color['rms'], linewidth=3)
    ax1.axhline(-rms[0].value, color=color['rms'], linewidth=3)
    ax1.axhline(+iqr[0].value, color=color['iqr'], linewidth=3)
    ax1.axhline(-iqr[0].value, color=color['iqr'], linewidth=3)

    ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
    ax2.hist(magDiff[0], bins=25, range=diffRange,
             orientation='horizontal', histtype='stepfilled',
             normed=True, color=color['bright'])
    ax2.set_xlabel("relative # / bin")

    labelTemplate = r'PA1({label}) = {q.value:4.2f} {q.unit:latex}'
    yv = np.linspace(diffRange[0], diffRange[1], 100)
    ax2.plot(scipy.stats.norm.pdf(yv, scale=rms[0]), yv,
             marker='', linestyle='-', linewidth=3, color=color['rms'],
             label=labelTemplate.format(label='RMS', q=rms[0]))
    ax2.plot(scipy.stats.norm.pdf(yv, scale=iqr[0]), yv,
             marker='', linestyle='-', linewidth=3, color=color['iqr'],
             label=labelTemplate.format(label='IQR', q=iqr[0]))
    ax2.set_ylim(*diffRange)
    ax2.legend()
    ax1.set_xlabel("psf magnitude")
    ax1.set_ylabel(r"psf magnitude diff ({0.unit:latex})".format(magDiff))
    for label in ax2.get_yticklabels():
        label.set_visible(False)

    plt.tight_layout()  # fix padding

    pathFormat = "{name}.{ext}"
    plotPath = makeFilename(outputPrefix, pathFormat, name="PA1", ext=plotExt)
    plt.savefig(plotPath, format=plotExt)
    plt.close(fig)
    print("Wrote plot:", plotPath)


def plotAMx(job, amx, afx, filterName, amxSpecName='design', outputPrefix="",
            showTitle=True, plotExt='png'):
    """Plot a histogram of the RMS in relative distance between pairs of
    stars.

    Creates a file containing the plot with a filename beginning with
    `outputPrefix`.

    Parameters
    ----------
    job : `lsst.verify.Job`
        `~lsst.verify.Job` providing access to metrics, specs and measurements
    amx : `lsst.verify.Measurement`
    afx : `lsst.verify.Measurement`
    filterName : `str`
    amxSpecName : `str`, optional
        Name of the AMx specification to reference in the plot.
        Default: ``'design'``.
    outputPrefix : `str`, optional
        Prefix to use for filename of plot file.  Will also be used in plot
        titles. E.g., ``outputPrefix='Cfht_output_r_'`` will result in a file
        named ``'Cfht_output_r_AM1_D_5_arcmin_17.0-21.5.png'``
        for an ``AMx.name=='AM1'`` and ``AMx.magRange==[17, 21.5]``.
    """
    if np.isnan(amx.quantity):
        print("Skipping %s -- no measurement"%str(amx.metric_name))
        return

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    histLabelTemplate = 'D: [{inner.value:.1f}{inner.unit:latex}-{outer.value:.1f}{outer.unit:latex}]\n'\
                        'Mag: [{magBright:.1f}-{magFaint:.1f}]'
    annulus = amx.extras['annulus'].quantity
    magRange = amx.extras['magRange'].quantity
    ax1.hist(amx.extras['rmsDistMas'].quantity, bins=25, range=(0.0, 100.0),
             histtype='stepfilled',
             label=histLabelTemplate.format(
                 inner=annulus[0],
                 outer=annulus[1],
                 magBright=magRange[0],
                 magFaint=magRange[1]))
    metric_name = amx.metric_name
    amxSpec = job.specs[Name(package=metric_name.package, metric=metric_name.metric, spec=amxSpecName)]
#    amxSpecLabelTemplate = '{amx.datum.label} {specname}: {amxSpec.threshold:.1f}'
    amxSpecLabelTemplate = '{label} {specname}: {amxSpec.threshold:.1f}'
    amxSpecLabel = amxSpecLabelTemplate.format(
        amx=amx,
        label=amx.datum.label.split('.')[1],
        specname=amxSpecName,
        amxSpec=amxSpec)
    ax1.axvline(amxSpec.threshold.value, 0, 1, linewidth=2, color='red',
                label=amxSpecLabel)

    if amxSpec.check(amx.quantity):
        amxStatus = 'passed'
    else:
        amxStatus = 'failed'
#    amxLabelTemplate = '{amx.datum.label} measured: {amx.quantity:.1f} ({status})'
    amxLabelTemplate = '{label} measured: {amx.quantity:.1f} ({status})'
    amxLabel = amxLabelTemplate.format(
        amx=amx,
        label=amx.datum.label.split('.')[1],
        status=amxStatus)
    ax1.axvline(amxSpec.threshold.value, 0, 1, linewidth=2, color='black',
                label=amxLabel)

    afxSpec = job.specs[Name(package=afx.metric_name.package, metric=afx.metric_name.metric, spec='srd')]
    if afxSpec.check(afx.quantity):
        afxStatus = 'passed'
    else:
        afxStatus = 'failed'
#    afxLabelTemplate = '{afx.datum.label} {afxSpec.name}: {afxSpec.threshold}%\n' + \
#                       '{afx.datum.label} measured: {afx.quantity:.1f}% ({status})'
    afxLabelTemplate = '{label} {specname}: {afxSpec.threshold}\n' + \
                       '{label} measured: {afx.quantity:.1f} ({status})'
    afxLabel = afxLabelTemplate.format(
        afx=afx,
        label=afx.datum.label.split('.')[1].split('_')[0],
        afxSpec=afxSpec,
        specname=amxSpecName,
        status=afxStatus)
    ax1.axvline((amx.quantity + afx.extras['ADx'].quantity).value,
                0, 1, linewidth=2, color='green',
                label=afxLabel)

    title = '{metric} Astrometric Repeatability over {D.value:.0f}{D.unit:latex}'.format(
        metric=amx.datum.label,
        D=amx.extras['D'].quantity)
    if showTitle:
        ax1.set_title(title)

    ax1.set_xlim(0.0, 100.0)
    ax1.set_xlabel(
        '{rmsDistMas.label} ({unit})'.format(
            rmsDistMas=amx.extras['rmsDistMas'], unit=amx.extras['rmsDistMas'].quantity.unit._repr_latex_()))
    ax1.set_ylabel('# pairs / bin')

    ax1.legend(loc='upper right', fontsize=16)

    pathFormat = '{metric}_D_{D:d}_{Dunits}_' + \
        '{magBright.value}_{magFaint.value}_{magFaint.unit}.{ext}'
    plotPath = makeFilename(outputPrefix,
                            pathFormat,
                            metric=amx.datum.label,
                            D=int(amx.extras['D'].quantity.value),
                            Dunits=amx.extras['D'].quantity.unit,
                            magBright=magRange[0],
                            magFaint=magRange[1],
                            ext=plotExt)

    plt.tight_layout()  # fix padding
    plt.savefig(plotPath, dpi=300, format=plotExt)
    plt.close(fig)
    print("Wrote plot:", plotPath)


def plotTEx(job, tex, filterName, texSpecName='design', outputPrefix='',
            showTitle=True, plotExt='png'):
    """Plot TEx correlation function measurements and thresholds.

    Parameters
    ----------
    job : `lsst.verify.Job`
        `Job` providing access to metrics, specs, and measurements
    tex : `lsst.verify.Measurement
        The ellipticity residual correlation `Measurement` object
    filterName : str
        Name of the filter of the images
    texSpecName : str
        Level of requirement to compare against.
        Must be a into the metrics specified in the tex Measurement object
        Typically one of 'design', 'minimum', 'stretch'
    outputPrefix : str, optional
        Prefix to use for filename of plot file.

    Effects
    -------
    Saves an output plot file to that starts with specified outputPrefix.

    """
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    # Plot correlation vs. radius
    radius = tex.extras['radius'].quantity
    xip = tex.extras['xip'].quantity
    xip_err = tex.extras['xip_err'].quantity
    D = tex.extras['D'].quantity
    bin_range_operator = tex.extras['bin_range_operator'].quantity

    ax1.errorbar(radius.value, xip.value, yerr=xip_err.value)
    ax1.set_xscale('log')
    ax1.set_xlabel('Separation (arcmin)', size=19)
    ax1.set_ylabel('Median Residual Ellipticity Correlation', size=19)

    # Overlay requirements level
    metric_name = tex.metric_name
    texSpec = job.specs[Name(package=metric_name.package, metric=metric_name.metric, spec=texSpecName)]
    texSpecLabel = '{tex.datum.label} {specname}: {texSpec:.2g}'.format(
        tex=tex,
        texSpec=texSpec.threshold,
        specname=texSpecName)
    ax1.axhline(texSpec.threshold.value, 0, 1, linewidth=2, color='red',
                label=texSpecLabel)

    # Overlay measured KPM whether it passed or failed.
    if texSpec.check(tex.quantity):
        texStatus = 'passed'
    else:
        texStatus = 'failed'
    texLabelTemplate = '{tex.datum.label} measured: {tex.quantity:.2g} ({status})'
    texLabel = texLabelTemplate.format(tex=tex, status=texStatus)

    ax1.axhline(tex.quantity.value, 0, 1, linewidth=2, color='black',
                label=texLabel)

    titleTemplate = """
        {metric} Residual PSF Ellipticity Correlation
        {bin_range_operator:s} {D.value:.1f}{D.unit:latex}
        """
    title = titleTemplate.format(metric=tex.datum.label,
                                 bin_range_operator=bin_range_operator,
                                 D=D)
    if showTitle:
        ax1.set_title(title)

    ax1.set_xlim(0.0, 20.0)
    ax1.set_xlabel(
        '{radius.label} ({unit})'.format(
            radius=tex.extras['radius'], unit=radius.unit._repr_latex_()))
    ax1.set_ylabel('Correlation')

    ax1.legend(loc='upper right', fontsize=16)

    pathFormat = '{metric}_D_{D:d}_{Dunits}.{plotExt}'
    plotPath = makeFilename(outputPrefix,
                            pathFormat,
                            metric=tex.datum.label,
                            D=int(D.value),
                            Dunits=D.unit,
                            ext=plotExt)

    plt.tight_layout()  # fix padding
    plt.savefig(plotPath, dpi=300, ext=plotExt)
    plt.close(fig)
    print("Wrote plot:", plotPath)
