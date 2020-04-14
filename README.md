# lsst.validate.drp

**Validate an LSST DM processCcd.py output repository against a set of [LSST Science Requirements Document](https://ls.st/srd) Key Performance Metrics.**

Also assess expected analytic models for photometric and astrometric performance following the [LSST Overview paper](http://arxiv.org/abs/0805.2366v4).

`validate_drp` is part of the LSST Science Pipelines.
You can learn how to install the Pipelines at https://pipelines.lsst.io/install/index.html.

## Quick run with CFHT observations

```
setup validate_drp
validateDrp.py CFHT/output
```

This will analyse the processed data in a Butler repository called `CFHT/output` and produce metric printouts, plots, and JSON files. Metrics will be separately calculated for each filter represented in the repository.

Replace `CFHT/output` with your favorite processed data repository and you will get reasonable output.

You can run `validateDrp.py` in any of the following modes:

1. Use no configuration file (as above).
2. Pass a configuration file with just validation parameters (such as `brightSnrMin`) but no `dataId` specifications.
3. Pass a configuration file that specifies validation parameters and the `dataIds` to process.  See examples below for use with a `--configFile`

## Full processCcd examples

This package also includes examples that run processCcd task on some
CFHT data and DECam data
and validate the astrometric and photometric repeatability
and the residual PSF ellipticity correlation of the results.

Pre-requisites for the examples: install and declare the following

1. `pipe_base` from the LSST DM stack (note that `pipe_base` is included with `lsst_apps`, which is the usual thing to install)
2. `obs_decam` from https://github.com/lsst/obs_decam
3. `obs_cfht` from https://github.com/lsst/obs_cfht
4. `validation_data_cfht` from https://github.com/lsst/validation_data_cfht
5. `validation_data_decam` from https://github.com/lsst/validation_data_decam

The `obs_decam`, `obs_cfht`, `validation_data_cfht`, `validation_data_decam`, `validate_drp` products are also buildable by the standard LSST DM stack tools: `lsstsw` or `eups distrib`.  But they (intentionally) aren't in the dependency tree of `lsst_apps`.

If you have a stack already [installed with `lsst_apps`](https://pipelines.lsst.io/install/index.html), you can install these in the same manner.

Using [eups distrib](https://pipelines.lsst.io/install/newinstall.html):

```
eups distrib install obs_decam obs_cfht validation_data_decam validation_data_cfht validate_drp
```

Alternatively, using [lsstsw](https://pipelines.lsst.io/install/lsstsw.html):

```
rebuild -u obs_decam obs_cfht validation_data_decam validation_data_cfht validate_drp
```

### Example: CFHT

To set up for a run with CFHT:

```
setup pipe_tasks
setup obs_cfht
setup validation_data_cfht
setup validate_drp
```

As usual, if any of these packages are not declared current you will also need to specify a version or tag.

`validation_data_cfht` contains both the test CFHT data and selected SDSS reference catalogs in astrometry.net format.  It also contains an already-processed set of output data products.

Analyze the `validation_data_cfht` data products with

```
validateDrp.py ${VALIDATION_DATA_CFHT_DIR}/data
```

To add a custom set of requirements to pass that differ from the LSST Science Requirements Document, pass a configuration file option:
```
validateDrp.py ${VALIDATION_DATA_CFHT_DIR}/data --configFile ${VALIDATE_DRP_DIR}/examples/Cfht.yaml
```

Run the measurement algorithm processing and astrometry test with

```
$VALIDATE_DRP_DIR/examples/runCfhtTest.sh
```

This will create a repository in your current working directory called CFHT.

**Note:** You can, instead, run:

```
examples/runCfhtQuickTest.sh
```

This quick test processes on a single CCD, which is useful for debugging.

### Example: HSC

To setup for a run with data from the HSC camera on Subaru:

```
setup obs_subaru
setup pipe_drivers
setup validation_data_hsc
setup validate_drp
```

Note that `pipe_drivers` is required for the HSC processing; it is not required for the Cfht or DECam processing.

As usual, if any of these packages are not declared current you will also need to specify a version or tag.

`validation_data_decam` contains both the test DECam data and selected SDSS reference catalogs in astrometry.net format.  It also contains an already-processed set of output data products.

Analyze the `validation_data_decam` data products with

```
validateDrp.py ${VALIDATION_DATA_DECAM_DIR}/data
```

To add a custom set of requirements to pass that differ from the LSST Science Requirements Document, pass a configuration file option:
```
validateDrp.py ${VALIDATION_DATA_DECAM_DIR}/data --configFile ${VALIDATE_DRP_DIR}/examples/decam.yaml
```

Run the measurement algorithm processing and astrometry test with

```
$VALIDATE_DRP_DIR/examples/runHscTest.sh
```

This will create a repository in your current working directory called `data_hsc`.

**Note:** You can, instead, run:

```
setup obs_subaru
setup pipe_drivers
setup validate_drp

# The ci_hsc doesn't install any data in the stack dir.
# It's meant to be a an integration test itself.
# If you have used lsstsw to "install" ci_hsc,
# the actual data we want is in the build dir.
setup -k -r "${LSST_BUILD_DIR}"/../build/ci_hsc

$VALIDATE_DRP_DIR/examples/runHscQuickTest.sh
```

This quick test processes on just a few CCDs that are in the data in `ci_hsc`.  A repository will be created in the current working directory as `data_hsc_quick`. The `validation_data_hsc` data set is not required for this quick test.

### Details on the processing

While `examples/run{Cfht,DECam,HSC}Test.sh` respectively do all of the processing and validation analysis, below are some examples of running the processing/measurement steps individually.  While these examples are from  the CFHT validation example, analogous commands would work for DECam.  The HSC processing uses `singleFrameDriver.py` instead of `processCcd.py`.

1. Make sure the astrometry.net environment variable is pointed to the right place for this validation set:
    ```
    export ASTROMETRY_NET_DATA_DIR=${VALIDATION_DATA_CFHT_DIR}/astrometry_net_data
    ```

2. Ingest the files into the repository
    ```
    mkdir -p CFHT/input
    echo lsst.obs.cfht.MegacamMapper > CFHT/input/_mapper
    ingestImages.py CFHT/input "${VALIDATION_DATA_CFHT_DIR}"/raw/*.fz --mode link
    ```

3. Create the `runCfht.list` file from the YAML configuration file
    ```
    makeRunList.py "${VALIDATE_DRP_DIR}"/examples/runCfht.yaml > "${VALIDATE_DRP_DIR}"/examples/runCfht.list
    ```

Once these basic steps are completed, then you can run any of the following:

* To process all CCDs with the standard AstrometryTask and 6 threads use newAstrometryConfig.py:
    ```
    processCcd.py CFHT/input @examples/runCfht.list --configfile config/newAstrometryConfig.py --clobber-config -j 6 --output CFHT/output
    ```

* To process all CCDs with the old ANetAstrometryTask and 6 threads:
    ```
    processCcd.py CFHT/input @examples/runCfht.list --configfile config/anetAstrometryConfig.py --clobber-config -j 6 --output CFHT/output
    validateDrp.py CFHT/output --configFile examples/runCfht.yaml
    ```

* To process one CCD with the new AstrometryTask:
    ```
    processCcd.py CFHT/input  --id visit=850587 ccd=21 --configfile config/newAstrometryConfig.py --clobber-config --output tempout
    ```

* Or process one CCD with the ANetAstrometryTask:
    ```
    processCcd.py CFHT/input --id visit=850587 ccd=21 --configfile config/anetAstrometryConfig.py --clobber-config --output tempout
    ```

* Run the validation test
    ```
    validateDrp.py CFHT/output --configFile examples/runCfht.yaml
    ```

Note that the example validation test selects several of the CCDs and will fail if you just pass it a repository with 1 visit or just 1 CCD.

## Reference

### validate.py commmand line usage

`validateDrp.py` is the command line interface to `validate_drp`.

```
usage: validateDrp.py [-h] [--configFile CONFIGFILE]
                      [--metricsFile METRICSFILE] [--verbose] [--noplot]
                      [--level LEVEL]
                      repo

Calculate and plot validation Key Project Metrics from the LSST SRD.
http://ls.st/LPM-17 Produces results to: STDOUT Summary of key metrics
`REPONAME*.png` Plots of key metrics. Generated in current working directory.
`REPONAME*.json` JSON serialization of each KPM. where REPONAME is based on the
repository name but with path separators replaced with underscores. E.g.,
`Cfht/output` -> `Cfht_output_`.

positional arguments:
  repo                  path to a repository containing the output of
                        processCcd

optional arguments:
  -h, --help            show this help message and exit
  --configFile CONFIGFILE, -c CONFIGFILE
                        YAML configuration file validation parameters and
                        dataIds.
  --metricsFile METRICSFILE
                        Path of YAML file with LPM-17 metric definitions.
  --verbose, -v         Display additional information about the analysis.
  --noplot              Skip making plots of performance.
  --level LEVEL         Level of SRD requirement to meet: "minimum", "design",
                        "stretch"
```

### YAML configuration files (--configFile)

You can configure `validateDrp.py` by assigning a YAML file's path to the `--configFile` command argument.
See `validate_drp`'s `examples/` directory for examples.
The fields include:

- `brightSnrMin` (float): Only consider stars with a S/N high than this limit.
- `visits` (array of ints): List of visit identifiers to process.
- `filter` (array of strings, or string): Filter to process, or a filter name for each visit (see [Specifying multiple filters](#specifying-multiple-filters)).
- `ccd` (array of ints): List of CCD chips to process from each visit.

#### Specifying multiple filters

Multiple filters can be processed by specifying a filter name for each visit.  See `examples/DecamCosmos.yaml` for an example.  In brief:

```
# Visit - filter are matched pairs.
visits: [176837, 176839, 176840, 176841, 176842, 176843, 176844, 176845, 176846,
         177341, 177342, 177343, 177344, 177345, 177346,]
filter: ['z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z',
         'r', 'r', 'r', 'r', 'r', 'r', ]
# ccd list is iterated through for each visit-filter pair
ccdnum: [10, 11, 12, 13, 14, 15, 16, 17, 18]
```

### metrics.yaml (--metricsFile)

Metrics and their specification levels are defined in a YAML file.
You can choose this file by assigning its path to `validateDrp.py`'s `--metricsFile` argument.
By default, `validateDrp.py` uses the `metrics.yaml` file in the `etc/` directory of its repository.
See the `validate_base` documentation for details on the file's schema.

To add a custom specification level for an existing metric:

1. Make a copy of `validate_drp`'s `metrics.yaml`.
2. Add the new specification level (see the `validate_base` documentation, and also follow the existing YAML schema).
3. Use this custom metric file to `validateDrp.py` with the `--metricsFile` argument.


## Notes

* Will likely not successfully run on more than 500 catalogs per band due to memory limits and inefficiencies in the current matching approach.
* The astrometric and photometric error models are formally valid for individual images.  However, they are being applied here to the results from the set of images, which is implicitly looking at some sort of mean performance.
E.g., the expected astrometric uncertainty is intimately related to the seeing of the image.  For collections of images where most have a similar seeing, these estimates are useful and reasonable.  However, if the data set analyzed consisted of a set of images distributed across a wide range of seeing values, then the fits here have less direct meaning.

## Files

* `etc/metrics.yaml`: Definitions of metrics measured by `validate_drp`.
* `bin.src/validateDrp.py`: Analyze output data produced by processCcd.py
* `config/cfhtConfig.py`: empty config overrides for Cfht.  Edit to easily include config parameters in the examples.
* `config/decamConfig.py`: empty config overrides for Decam.  Edit to easily include config parameters in the examples.
* `examples/runCfhtTest.sh`: CFHT Run initialization, ingest, measurement, and astrometry validation.
* `examples/runDecamTest.sh`: DECam Run initialization, ingest, measurement, and astrometry validation.
* `examples/runHscTest.sh`: HSC Run initialization, ingest, measurement, and astrometry validation.
* `examples/runCfhtQuickTest.sh`: Quick version.
* `examples/runDecamQuickTest.sh`: Quick version.
* `examples/runHscQuickTest.sh`: Quick run version.  Uses `ci_hsc` data instead of `validation_data_hsc`.
* `examples/runExample.sh`: General example runner.
* `examples/Cfht.yaml`: CFHT YAML file with visits, ccd, paramaters for validateDrp.
* `examples/Decam.yaml`: DECam YAML file with visits, ccd, paramaters for validateDrp.
* `examples/DecamCosmos.yaml`: DECam COSMOS YAML file with visits, ccd, paramaters for validateDrp.
* `python/lsst/validate/drp/validate.py`: Main driver for metric measurements, plotting, and summary printouts.
* `python/lsst/validate/drp/calcsrd`: Includes modules that calculate each metric
* `python/lsst/validate/drp/matchreduce.py`: Matches star catalogs across mulitple visits; this catalog is used by metric measurements including AMx, ADx, AFx, PA1, PA2, PF1, TE1, and TE2.
* `python/lsst/validate/drp/astromerrmodel.py`: Model of astrometric errors.
* `python/lsst/validate/drp/photerrmodel.py`: Model of photometric errors.
* `python/lsst/validate/drp/plot.py`: Generate matplotlib visualizations metrics and error models.
* `python/lsst/validate/drp/util.py`: Utility routines.
* `README.md`: THIS FILE.  Guide and examples.
