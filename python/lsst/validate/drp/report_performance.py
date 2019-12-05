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

from astropy.table import Column, Table
import json
import numpy as np

from lsst.verify import Job, SpecificationSet, MetricSet
from .validate import get_specs_metrics


def run(validation_drp_report_filenames, output_file,
        srd_level=None,
        release_specs_package=None, release_level=None,
        metrics_package='verify_metrics'):
    """
    Parameters
    ---
    validation_drp_report_filenames : list or str
        filepaths for JSON files.
    output_file : str
        filepath of output RST file.
    srd_level : str
        SRD level to quote.  One of ['design', 'minimum', 'stretch']
    release_specs_package : str, optional
        Name of package to use in constructing the release level specs.
    release_level : str, A specification level in the 'release_specs_file'
       E.g., 'FY17' or 'ORR'

    Products
    ---
    Writes table of performance metrics to an RST file.
    """
    input_objects = ingest_data(validation_drp_report_filenames, metrics_package)
    input_table = objects_to_table(input_objects, level=srd_level)
    if input_table is None:
        msg = "Table from Job is None.  Returning without writing table"
        print(msg)
        return
    if release_specs_package is not None and release_level is not None:
        tmp_specs = SpecificationSet.load_metrics_package(release_specs_package, subset='validate_drp')
        release_specs = tmp_specs.subset(spec_tags=['release'])
        add_release_spec(input_table, release_specs, release_level, srd_level)

    write_report(input_table, output_file)


def ingest_data(filenames, metrics_package):
    """Load JSON files into a list of lsst.validate.base measurement Jobs.

    Parameters
    ----------
    filenames : list of str
        Filenames of JSON files to load.

    Returns
    -------
    job_list : list of lsst.validate.base.Job
        Each element is the Job representation of the JSON file.
    """
    jobs = {}
    # Read in JSON output from metrics run
    for filename in filenames:
        with open(filename) as fh:
            data = json.load(fh)
            job = Job.deserialize(**data)
        filter_name = job.meta['filter_name']
        metrics = MetricSet.load_metrics_package(metrics_package)
        job.metrics.update(metrics)
        specs = SpecificationSet.load_metrics_package(metrics_package)
        job.specs.update(specs)
        jobs[filter_name] = job

    return jobs


# Identify key data from JSON
def objects_to_table(input_objects, level='design'):
    """Take lsst.validate.base.Job objects and convert to astropy.table.Table

    Parameters
    ----------
    input_objects : list of Job objects
    level : str
        The requirement level to compare to for each metric.
        This is required because there are metrics with dependencies
        E.g., AD1 depends on AF1.  Thus you have to specify a level
        to even get a single AD1 metric.

    Returns
    -------
    report : astropy.table.Table
        Table with columns needed for final report.
    """
    rows = []
    for filter_name, job in input_objects.items():
        specs, metrics = get_specs_metrics(job)
        for key, m in job.measurements.items():
            parts = key.metric.split("_")
            metric = parts[0]  # For compound metrics
            if len(parts) > 1:
                if level not in ".".join(parts):
                    continue
            spec_set = specs[metric]
            spec = None
            for spec_key in spec_set:
                if level in job.specs[spec_key].tags:
                    spec = job.specs[spec_key]
            # If spec is not tagged, ispect the name.
            if spec is None:
                for spec_key in spec_set:
                    if level in spec_key.spec:
                        spec = job.specs[spec_key]
            if spec is not None:
                if np.isnan(m.quantity):
                    meas_quantity_value = "**"  # -- is reserved in rst for headers
                else:
                    meas_quantity_value = m.quantity.value
                this_row = [metric, filter_name, meas_quantity_value, m.quantity.unit,
                            spec.operator_str, spec.threshold.value, job.meta['instrument']]
                rows.append(this_row)

    if len(rows) == 0:
        msg_format = "No rows loaded from Job at level: '{:s}'"
        print(msg_format.format(level))
        return None

    srd_requirement_col_name = 'SRD Requirement: %s' % level
    col_names = ('Metric', 'Filter', 'Value', 'Unit',
                 'Operator', srd_requirement_col_name, 'Instrument')
    output = Table(rows=rows, names=col_names)
    output.add_column(Column(['']*len(output), dtype=str, name='Comments'))
    return output


# Calculate numbers in table
def add_release_spec(data, release_specs, release_specs_level, srd_level):
    """Add columns of additional metric thresholds.

    Intended use is for specifying a set of release metrics that converge
    over time to the SRD metrics.

    If release_specs_level is not present in release_specs,
    then the original data is unchanged.
    """
    release_targets = []
    for row in data:

        specs = release_specs.subset(required_meta={'filter_name': row['Filter'],
                                                    'instrument': row['Instrument']},
                                     spec_tags=['chromatic'])
        specs.update(release_specs.subset(required_meta={'instrument': row['Instrument']},
                                          spec_tags=['achromatic']))
        value = None
        for spec in specs:
            parts = spec.metric.split('_')  # for compound specs
            if len(parts) == 2 and parts[1] != srd_level:
                continue
            if parts[0] == row['Metric'] and release_specs_level in spec.spec:
                value = specs[spec].threshold.value
        release_targets.append(value)

    release_targets_col = Column(
        release_targets,
        dtype=float,
        name='Release Target: %s' % release_specs_level)
    data.add_column(release_targets_col)


def float_or_dash(f, format_string='{:.3g}'):
    """Return string of formatted float, or -- if None.

    Intended use is to provide formatting output for columns
    where where None or non-float value indicates a missing measurement.
    """
    # This try/except handles both None and non-numeric strings.
    try:
        f = float(f)
        return format_string.format(f)
    except Exception:
        # dashes are reserved
        return '**'


def blank_none(s):
    """Return a blank for  None or 'None', else return string of input.

    Intended use is to provide formatting output for columns where an empty
    or None value is totally reasonable and expected and should be display
    as a blank ''.
    """
    if s is None:
        return ''
    if s == 'None':
        return ''

    return str(s)


def find_col_name(prefix, colnames):
    """Return the first entry in 'colnames' that starts with 'prefix'."""
    for c in colnames:
        if c.startswith(prefix):
            return c


# Output table
def write_report(data, filename='test.rst', format='ascii.rst'):
    """Write performance report to RST file.

    Parameters
    ----------
    data : astropy.table.Table
    filename : str, optional
        Filepath of output RST file.
    format : str, optional
        astropy.table format for output table.

    Creates
    -------
    test.rst
        Output file with RST version of data Table.
    """
    # Find the 'Release Target XYZ' column name
    release_target_col_name = find_col_name('Release Target', data.colnames)
    # Find the 'SRD Requirement XYZ' column name
    srd_requirement_col_name = find_col_name('SRD Requirement', data.colnames)

    col_names = ['Metric', 'Filter', 'Unit', 'Operator',
                 srd_requirement_col_name,
                 release_target_col_name,
                 'Value', 'Comments']
    use_col_names = [c for c in col_names if c in data.colnames]
    # Provide default formats
    for spec_col in (release_target_col_name, srd_requirement_col_name):
        if spec_col in data:
            data[spec_col].info.format = '.2f'
    data['Value'].info.format = float_or_dash
    data['Unit'].info.format = blank_none
    # Astropy 1.2.1 (the current miniconda stack install) doesn't support
    #  overwrite=True for Tables
    # But in Astropy 2.0 (the modern version) reliance on automatically overwriting
    # ASCII files is deprecated and you get a warning if you don't specify it.
    # But it does still write.  So for now we'll leave overwrite=True out of
    # the argument list below, but someday it will likely be required to include it.
    data[use_col_names].write(filename=filename, format=format,
                              include_names=use_col_names)
