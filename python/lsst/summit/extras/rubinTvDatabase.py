# This file is part of summit_extras.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import glob
import os
import pandas as pd
import logging

from lsst.summit.utils.utils import getSite

__all__ = [
    'getRubinTvDatabase',
    'SUPPORTED_INSTRUMENTS'
]

SUPPORTED_INSTRUMENTS = (
    'LATISS',
    'LSSTCam',
    'ts8'
    # 'TMA',  # TODO: Add support for TMA and then add here
)


def _getLibraryPathForInstrument(instrument):
    """Get the path for the metadata json files for the instrument.

    Automatically works out the location using getSite() as only
    places for which getSite() returns are supported by this tool
    anyway.

    Parameters
    ----------
    instrument : `str`
        The instrument to get the file path for.

    Returns
    -------
    path : `str`
        The path containin the json files

    Raises:
        ValueError: Raised if the site is not supported/cannot be determined.
    """
    site = None

    try:
        site = getSite()
    except ValueError:
        # XXX remove this before merging and just let it raise
        # just for use on Merlin's laptop
        print('defaulting to Merlins laptop for testing')
        site = 'merlins_laptop'

    match site:
        case 'summit':
            match instrument:
                case 'LATISS':
                    return '/project/rubintv/sidecar_metadata'
                case 'LSSTCam':
                    raise RuntimeError('LSSTCam not supported at the summit (yet, and hopefully ever!)')
                case 'ts8':
                    raise RuntimeError('No data for ts8 at the summit')
                case _:
                    raise ValueError(f'{instrument=} is unsupported at site {site}')

        case 'rubin-devl':
            match instrument:
                case 'LATISS':
                    return '/sdf/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/'
                case 'LSSTCam':
                    return '/sdf/scratch/rubin/rapid-analysis/LSSTCam/sidecar_metadata/'
                case 'ts8':
                    return '/sdf/scratch/rubin/rapid-analysis/ts8/sidecar_metadata/'
                case _:
                    raise ValueError(f'{instrument=} is unsupported at site {site}')

        case 'staff-rsp':
            match instrument:
                case 'LATISS':
                    return '/home/m/mfl/u/rubinTvDataProducts/sidecar_metadata/'
                case 'LSSTCam':
                    return '/scratch/rubin/rapid-analysis/LSSTCam/sidecar_metadata/'
                case 'ts8':
                    return '/scratch/rubin/rapid-analysis/ts8/sidecar_metadata/'
                case _:
                    raise ValueError(f'{instrument=} is unsupported at site {site}')

        case 'merlins_laptop':
            match instrument:
                case 'LATISS':
                    return '/Users/merlin/rsync/LATISS'
                case 'LSSTCam':
                    return '/Users/merlin/rsync/LSSTCam'
                case 'ts8':
                    return '/Users/merlin/rsync/ts8'
                case _:
                    raise ValueError(f'{instrument=} is unsupported at site {site}')

        case _:
            raise ValueError(f'Unsupported site: {site}')


def _getFilesFromLibrary(path):
    """Get the metadata files from the specified path

    Parameters
    ----------
    path : `str`
        The path to get the files from

    Returns
    -------
    files : `list` of `str`
        The files.

    Raises:
        RuntimeError: Raised if no files are found.
    """
    globPattern = os.path.join(path, '*.json')
    files = sorted(glob.glob(globPattern))
    if not files:
        raise RuntimeError(f'No files found in {path}')
    return files


def getRubinTvDatabase(instrument):
    """Get all of the RubinTV data as a single dataframe.

    The 'Exposure Id' is used as the primary key.

    This function will only work on the summit and at USDF. On the summit it
    will be up to date, but only go back some arbitrary distance. At USDF it is
    kept up-to-date by Merlin by hand, and so will probably not contain the
    most recent data.

    Note: the only supported sites are the summit and USDF. This is a VERY
    TEMPORARY workaround for people who are currently using web-scrapers to
    access this data. This is NOT a replacement for the visit database! It is
    kept up to date by hand at USDF, and will be retired *as soon as* the visit
    database is available!

    Parameters
    ----------
    instrument : `str`
        The instrument to get the dataframe for. Must be one of:
            - LSSTCam
            - LATISS
            - ts8
            - TMA  # TODO: Add support for TMA and remove this comment

    Returns
    -------
    data : `pd.dataframe`
        The dataframe containing all the available data.
    """
    # warn loudly in every conceivable way so that anyone running this will definitely see!
    msg = ("WARNING - This interface will disappear as soon as we have a visit database! "
           "This is a temporary convenience *only* - DO NOT build software which relies on this!")
    logging.warning(msg)  # go straight to the root logger to ensure this comes out
    logger = logging.getLogger(__file__)
    logger.warning(msg)

    if instrument not in SUPPORTED_INSTRUMENTS:
        raise RuntimeError(f'Instrument {instrument} is not supported - choose from: {SUPPORTED_INSTRUMENTS}')

    libraryPath = _getLibraryPathForInstrument(instrument)
    files = _getFilesFromLibrary(libraryPath)
    logger.info(f'Loading data for {len(files)} days...')
    dfs = []
    for filename in files:
        df = pd.read_json(filename).T
        df = df.sort_index()
        dfs.append(df)

    # TODO: need to create an 'Exposure Id' column for LSSTCam and ts8, derived
    # from the obsid, I think.

    dfsIndexed = [df.set_index('Exposure id') for df in dfs]
    combinedDf = pd.concat(dfsIndexed, verify_integrity=True)
    return combinedDf
