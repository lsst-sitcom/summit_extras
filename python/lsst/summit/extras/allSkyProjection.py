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
from __future__ import annotations

import datetime
import glob
import json
import logging
import os
from typing import TYPE_CHECKING

import astropy.units as u
import matplotlib.pylab as plt
import numpy as np
from astropy import wcs
from astropy.coordinates import EarthLocation, SkyCoord, get_body
from PIL import Image

if TYPE_CHECKING:
    from lsst.daf.butler import DimensionRecord

FOVS = {"LSSTCam": 3.5 * u.deg, "LSSTComCam": 0.7 * u.deg, "LATISS": 6.7 * u.arcmin}


def getDirName(dataRoot: str, date: datetime.datetime) -> str:
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
    dirName = f"ut{date.year - 2000}{date.month:02}{date.day:02}"
    return os.path.join(dataRoot, dirName)


def getDayObsFromDirName(dirName: str) -> int:
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

    return int(f"20{baseDir[2:]}")


def getJpgCreatedDatetime(filename: str) -> datetime.datetime | None:
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

                dateObject = datetime.datetime.strptime(dateStr, "%Y:%m:%d %H:%M:%S")
                return dateObject
            else:
                print(f"Failed to calculated the datetime for {filename}")
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

    def __init__(
        self, dataPath: str = "/sdf/data/rubin/offline/allsky/storage", warningThreshold: float = 5 * 60
    ) -> None:
        self._data = {}
        self.dataPath = dataPath
        self.warningThreshold = warningThreshold  # Threshold in seconds
        self.log = logging.getLogger(__name__)
        logging.basicConfig(level=logging.WARNING)  # Set the default logging level to WARNING

    def findClosest(self, targetDatetime: datetime.datetime, failDistance: float = 10 * 60) -> str | None:
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
            self.log.warning(
                f"Closest image is {timeDifference:.1f} seconds away from the specified" " datetime."
            )
            return None

        # Check if the time difference exceeds the warning threshold
        if timeDifference > self.warningThreshold:
            self.log.warning(
                f"Closest image is {timeDifference:.1f} seconds away from the specified" " datetime."
            )

        return self._data[closestDt]

    def update(self) -> None:
        """Crawl the directories and update the database with any new files."""
        scannedFiles = set(self._data.values())
        dirs = glob.glob(os.path.join(self.dataPath, "/ut2*"))
        nNewFiles = 0
        self.log.info(f"Updating database with files from {len(dirs)} directories")
        for dirName in dirs:
            pattern = os.path.join(dirName, "*.jpg")
            filesInDir = set(glob.glob(pattern))
            newFiles = filesInDir - scannedFiles
            for file in newFiles:
                timeTaken = getJpgCreatedDatetime(file)
                self._data[timeTaken] = file
                nNewFiles += 1
        self.log.info(f"Found {nNewFiles} new files")

    def save(self, filepath: str) -> None:
        """Save the database to a file.

        Parameters
        ----------
        filepath : `str`
            The path to the file to save the database to.
        """
        with open(filepath, "w") as f:
            serializedData = {
                dt.strftime("%Y-%m-%dT%H:%M:%S"): filename for dt, filename in self._data.items()
            }
            json.dump(serializedData, f)

    @classmethod
    def load(cls, filepath: str) -> AllSkyDatabase:
        """Load the database from a file and return an instance of the class.

        Parameters
        ----------
        filepath : `str`
            The path to the file to load the database from.
        """
        with open(filepath, "r") as f:
            serializedData = json.load(f)
            instance = cls()
            instance._data = {
                datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S"): filename
                for dt, filename in serializedData.items()
            }
            return instance


def getAllSkyWcs() -> wcs.WCS:
    # Solution from Peter
    w = wcs.WCS(naxis=2)
    x0 = np.array(
        [
            2.24999998e03,
            1.50000029e03,
            -4.92326522e-02,
            4.81183444e-02,
            1.80740466e-01,
            9.89446006e-01,
            -1.00321437e00,
            1.86998471e-01,
        ]
    )
    w.wcs.crpix = [x0[0], x0[1]]
    w.wcs.cdelt = [x0[2], x0[3]]
    w.wcs.pc = x0[4:8].reshape((2, 2))
    # Declare the ctype alt az as ALAT, ALON, use Zenith Equal Area projection
    w.wcs.ctype = ["ALON-ZEA", "ALAT-ZEA"]
    # Values at the reference pixel
    w.wcs.crval = [180, 90]

    return w


def makeCircle(
    az: u.quantity.Quantity, alt: u.quantity.Quantity, radius: u.quantity.Quantity, points: int = 100
) -> tuple[u.quantity.Quantity, u.quantity.Quantity]:
    """Generate a circle on a unit sphere centered at az,alt and with specified
    radius."""
    # 3d vector along altaz
    v1 = np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])

    # get perpendicular vectors
    if v1[0] == 0 and v1[1] == 0:
        v2 = np.array([1, 0, 0])
        v3 = np.array([0, 1, 0])
    else:
        v2 = np.array([-v1[1], v1[0], 0])
        v3 = np.array([-v1[2] * v1[0], -v1[2] * v1[1], v1[0] ** 2 + v1[1] ** 2])
        v2 /= np.sqrt(np.dot(v2, v2))
        v3 /= np.sqrt(np.dot(v3, v3))

    # Spin around
    circle_points = np.empty((points, 3))
    for i, th in enumerate(np.linspace(0, 2 * np.pi, points)):
        circle_points[i] = np.cos(th) * v2 + np.sin(th) * v3
    circle_points *= np.sin(radius)
    circle_points += v1 * np.cos(radius)

    # back to altaz
    alt = np.arcsin(circle_points[:, 2]) * u.radian
    az = np.arctan2(circle_points[:, 1], circle_points[:, 0]) * u.radian
    return az.to(u.deg), alt.to(u.deg)


def getBrightObjects() -> tuple[list[SkyCoord], list[str]]:
    # Objects of possible interest
    lmc = SkyCoord.from_name("LMC")
    smc = SkyCoord.from_name("SMC")
    sag = SkyCoord.from_name(
        "Sag A*",
    )
    scp = SkyCoord(ra=0 * u.deg, dec=-90 * u.deg)
    objs = [lmc, smc, sag, scp]
    names = ["LMC", "SMC", "Sag A*", "SCP"]

    cp = EarthLocation.of_site("Cerro Pachon")
    for obj in objs:
        obj.location = cp
    return objs, names


def plotAllSkyProjection(
    expRecords: list[DimensionRecord], allSkyDatabase: AllSkyDatabase
) -> plt.Figure | None:

    try:
        iter(expRecords)
    except TypeError:
        expRecords = [expRecords]

    w = getAllSkyWcs()

    # Use the last record in the list to get the closest allSky camera image.
    obstime = expRecords[-1].timespan.begin
    imageFilename = allSkyDatabase.findClosest(obstime.to_datetime())
    if imageFilename is None:
        print("No close images found")
        return None
    print(imageFilename)
    image = Image.open(imageFilename)
    print(f"datetime = {image._getexif()[306]}")
    print(f"exptime  = {image._getexif()[33434]} s")
    objs, names = getBrightObjects()
    for obj in objs:
        obj.obstime = obstime
    img = np.mean(image, axis=-1)  # squash rgb into grey

    # Use center of image to set vmax
    # Here's the mask for "center"
    xx = np.arange(4464.0)
    xx -= np.mean(xx)
    yy = np.arange(2976.0)
    yy -= np.mean(yy)
    xx, yy = np.meshgrid(xx, yy)
    rr = np.sqrt(xx**2 + yy**2)
    ww = rr < 1400
    vmax = np.quantile(img[ww], 0.99)

    fig = plt.figure(figsize=(4.46 * 1.5, 2.98 * 1.5))
    plt.imshow(img, vmax=vmax)

    for obj, name in zip(objs, names):
        plt.text(*w.world_to_pixel(obj.altaz.az, obj.altaz.alt), name, c="r")

    for name, sym in [("moon", "☽︎"), ("sun", "☉︎"), ("jupiter", "♃"), ("venus", "♀")]:
        body = get_body(name, obstime)
        body.location = EarthLocation.of_site("Cerro Pachon")
        if body.altaz.alt > -5 * u.deg:
            plt.text(*w.world_to_pixel(body.altaz.az, body.altaz.alt), sym, c="r", fontsize=15)

    fadeOut = np.linspace(0.2, 1, len(expRecords))
    for i, expRecord in enumerate(expRecords):
        az = expRecord.azimuth
        el = 90 - expRecord.zenith_angle
        radius = FOVS[expRecord.instrument] / 2
        circle = makeCircle(az * u.deg, el * u.deg, radius)
        plt.plot(*w.world_to_pixel(*circle), c="m", alpha=(fadeOut[i] if len(expRecords) > 1 else 1))
        plt.text(
            *w.world_to_pixel(az * u.deg, el * u.deg),
            f" {expRecord.instrument}",
            c="m",
            fontsize=10,
            alpha=(fadeOut[i] if len(expRecords) > 1 else 1),
        )

    # Add cardinal directions
    for name, az in [
        ("N", 0),
        ("E", 90),
        ("S", 180),
        ("W", 270),
    ]:
        plt.text(*w.world_to_pixel(az * u.deg, 5 * u.deg), name, c="r", ha="center")

    return fig
