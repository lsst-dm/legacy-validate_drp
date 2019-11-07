# LSST Data Management System
# Copyright 2008-2019 AURA/LSST.
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

from lsst.validate.drp.repeatability import measurePhotRepeat


def measurePA1(*args, **kwargs):
    """Measurement of the PA1 metric: photometric repeatability of
    measurements across a set of observations.

    Parameters
    ----------
    *args
        Arguments to pass to `lsst.validate.drp.measureRepeat`.
    **kwargs
        Additional keyword arguments to pass to `lsst.validate.drp.measureRepeat`.

    Returns
    -------
    measurement : `lsst.verify.Measurement`
        Measurement of PA1 and associated metadata.

    """

    return measurePhotRepeat(*args, **kwargs)
