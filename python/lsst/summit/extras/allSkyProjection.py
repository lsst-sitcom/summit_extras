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

import os
import datetime
import json
import glob
from PIL import Image
import logging


def getDirName(dataRoot, date):
    """
    Returns the directory name for a given date in the format 'utYYMMDD'.

    Parameters
    ----------
    dataRoot : `str`
        The root directory where the data is stored.
    date : `datetime.datetime`
        The datetime for the dayObs of the directory.

    Returns
    -------
    path : `str`
        The full path of the directory name in the format 'dataRoot/utYYMMDD'.
    """
    dirName = f'ut{date.year - 2000}{date.month:02}{date.day:02}'
    return os.path.join(dataRoot, dirName)


def getDayObsFromDirName(dirName):
    """
    Return the dayObs integer for the directory name.

    Parameters
    ----------
    dirName : `str`
        The directory name.

    Returns
    -------
    dayObs : `int`
        The day of observation in the format YYYYMMDD.

    Raises
    ------
    ValueError
        If the directory name is not in the format 'utYYMMDD'.
    """
    baseDir = os.path.basename(dirName)
    if not baseDir.startswith("ut") or len(baseDir) != 8:
        raise ValueError("Invalid directory format")

    return int(f'20{baseDir[2:]}')


def getJpgCreatedDatetime(filename):
    """Get the creation datetime of a JPG image file from its Exif metadata.

    Parameters
    ----------
    filename : `str`
        The path to the JPG image file.

    Returns
    -------
    creationDate : `datetime.datetime` or `None`
        The creation datetime of the JPG image file if it exists in the Exif
        metadata, otherwise None.
    """
    try:
        with Image.open(filename) as img:
            # need the private _getexif, the public one doesn't
            # exposure the 36867 key!
            exifData = img._getexif()

            # The Exif tag 36867 (0x9003) holds the date and
            # time when the original image data was generated
            if 36867 in exifData:
                dateStr = exifData[36867]

                dateObject = datetime.datetime.strptime(dateStr, '%Y:%m:%d %H:%M:%S')
                return dateObject
            else:
                print(f'Failed to calculated the datetime for {filename}')
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None


class AllSkyDatabase:
    """A class to hold the data for all sky image acquisition times.

    Finds the closest image to a specified datetime.

    Parameters
    ----------
    dataPath : `str`
        The path to the directory containing the data.
    warningThreshold : `float`
        The threshold in seconds for the warning message when the closest image
        is more than this number of seconds away from the specified datetime.
    """
    def __init__(self,
                 dataPath='/sdf/data/rubin/offline/allsky/storage',
                 warningThreshold=5*60):
        self._data = {}
        self.warningThreshold = warningThreshold  # Threshold in seconds
        self.log = logging.getLogger(__name__)
        logging.basicConfig(level=logging.WARNING)  # Set the default logging level to WARNING

    def findClosest(self, targetDatetime, failDistance=10*60):
        """Find the filen taken most closely in time to the specified datetime.

        Parameters
        ----------
        targetDatetime : `datetime.datetime`
            The datetime to find the closest image to.

        Returns
        -------
        filename : `str` or `None`
            The filename of the closest image, or `None` if this image is more
            than `failDistance` seconds away from the specified datetime.
        """
        closestDt = min(self._data.keys(), key=lambda dt: abs((dt - targetDatetime).total_seconds()))
        timeDifference = abs((closestDt - targetDatetime).total_seconds())

        if timeDifference > failDistance:
            self.log.warning(f"Closest image is {timeDifference:.1f} seconds away from the specified"
                             " datetime.")
            return None

        # Check if the time difference exceeds the warning threshold
        if timeDifference > self.warningThreshold:
            self.log.warning(f"Closest image is {timeDifference:.1f} seconds away from the specified"
                             " datetime.")

        return self._data[closestDt]

    def update(self):
        """Crawl the directories and update the database with any new files.
        """
        scannedFiles = set(self._data.values())
        dirs = glob.glob('/sdf/data/rubin/offline/allsky/storage/ut2*')
        nNewFiles = 0
        self.log.info(f"Updating database with files from {len(dirs)} directories")
        for dirName in dirs:
            pattern = os.path.join(dirName, '*.jpg')
            filesInDir = set(glob.glob(pattern))
            newFiles = filesInDir - scannedFiles
            for file in newFiles:
                timeTaken = getJpgCreatedDatetime(file)
                self._data[timeTaken] = file
                nNewFiles += 1
        self.log.info(f"Found {nNewFiles} new files")

    def save(self, filepath):
        """Save the database to a file.

        Parameters
        ----------
        filepath : `str`
            The path to the file to save the database to.
        """
        with open(filepath, 'w') as f:
            serializedData = {dt.strftime('%Y-%m-%dT%H:%M:%S'): filename
                              for dt, filename in self._data.items()}
            json.dump(serializedData, f)

    @classmethod
    def load(cls, filepath):
        """Load the database from a file and return an instance of the class.

        Parameters
        ----------
        filepath : `str`
            The path to the file to load the database from.
        """
        with open(filepath, 'r') as f:
            serializedData = json.load(f)
            instance = cls()
            instance._data = {datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S'): filename
                              for dt, filename in serializedData.items()}
            return instance
