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

import io
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime

import easyocr
import numpy as np
import pandas as pd
import requests
import tables  # noqa: F401 required for HDFStore append mode
from packaging import version

# Check Pillow version to determine the correct resampling filter
from PIL import Image, ImageEnhance
from PIL import __version__ as PILLOW_VERSION
from requests.exceptions import HTTPError

from lsst.summit.utils.utils import getSite, getSunAngle

if version.parse(PILLOW_VERSION) >= version.parse("9.1.0"):
    resample_filter = Image.LANCZOS
else:
    resample_filter = Image.ANTIALIAS

# coordinates as (x0, y0), (x1, y1) for cropping the various image parts
USER_COORDINATES = {
    "dateImage": ((42, 164), (222, 186)),
    "seeingImage": ((180, 128), (233, 154)),
    "freeAtmSeeingImage": ((180, 103), (233, 128)),
    "groundLayerImage": ((180, 80), (233, 103)),
}

SOAR_IMAGE_URL = "http://www.ctio.noirlab.edu/~soarsitemon/soar_seeing_monitor.png"

STORE_FILE = {
    "rubin-devl": "/sdf/scratch/rubin/rapid-analysis/SOAR_seeing/seeing_conditions.h5",
    "summit": "/project/rubintv/SOAR_seeing/seeing_conditions.h5",
}
ERROR_FILE = {
    "rubin-devl": "/sdf/scratch/rubin/rapid-analysis/SOAR_seeing/seeing_errors.log",
    "summit": "/project/rubintv/SOAR_seeing/seeing_errors.log",
}
FAILED_FILES_DIR = {
    "rubin-devl": "/sdf/scratch/rubin/rapid-analysis/SOAR_seeing/failed_files",
    "summit": "/project/rubintv/SOAR_seeing/failed_files",
}


@dataclass
class SeeingConditions:
    timestamp: datetime
    seeing: float
    freeAtmSeeing: float
    groundLayer: float

    def __repr__(self):
        return (
            f"SeeingConditions @ {self.timestamp.isoformat()}\n"
            f"  Seeing          = {self.seeing}\n"
            f"  Free Atm Seeing = {self.freeAtmSeeing}\n"
            f"  Ground layer    = {self.groundLayer}\n"
        )


class SoarSeeingMonitor:
    def __init__(self):
        site = getSite()
        self.STORE_FILE = STORE_FILE[site]

    def _reload(self):
        self.df = pd.read_hdf(self.STORE_FILE, key="data")

    def getSeeingAtTime(self, time):
        raise NotImplementedError


class SoarDatabaseBuiler:
    def __init__(self):
        site = getSite()
        self.STORE_FILE = STORE_FILE[site]
        self.ERROR_FILE = ERROR_FILE[site]
        self.FAILED_FILES_DIR = FAILED_FILES_DIR[site]

        logging.getLogger("easyocr").setLevel(logging.ERROR)
        self.reader = easyocr.Reader(["en"])

        # Ensure the HDFStore file exists
        if not os.path.exists(self.STORE_FILE):
            with pd.HDFStore(self.STORE_FILE, mode="w") as store:  # noqa: F841
                pass  # Create an empty store

        print(f"Writing to database at {self.STORE_FILE}")

        self.last_etag = None
        self.lastModified = None

    def getCurrentSeeingFromWebsite(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SoarSeeingMonitor/1.0; +http://tellmeyourseeing.com/bot)"
        }

        # Add conditional headers if we have previous values
        if self.last_etag:
            headers["If-None-Match"] = self.last_etag
        if self.lastModified:
            headers["If-Modified-Since"] = self.lastModified

        try:
            with requests.get(SOAR_IMAGE_URL, stream=True) as response:
                if response.status_code == 304:
                    # Resource has not changed; no need to process
                    print("Image has not changed since the last check.")
                    return None

                response.raise_for_status()

                self.last_etag = response.headers.get("ETag")
                self.lastModified = response.headers.get("Last-Modified")

                if "image" not in response.headers.get("Content-Type", ""):
                    raise ValueError("URL did not return an image.")

                imageData = response.content
                seeingConditions = self.getSeeingConditionsFromBytes(imageData)

                return seeingConditions

        except HTTPError as httpErr:
            print(f"HTTP error occurred: {httpErr}")
            # Handle HTTP errors (e.g., 403 Forbidden)
            # You can implement backoff strategies or logging here
            return None
        except Exception as e:
            # Log the exception
            print(f"An error occurred: {e}")
            with open(self.ERROR_FILE, "a") as f:
                f.write(f"Exception at {datetime.now()}:\n")
                traceback.print_exc(file=f)
                if "imageData" in locals():
                    filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".png"
                    failedImagePath = os.path.join(self.FAILED_FILES_DIR, filename)
                    with open(failedImagePath, "wb") as f:
                        f.write(imageData)
        return None

    @staticmethod
    def adjustCoords(coords, height):
        (x0, y0), (x1, y1) = coords
        # Adjust y coordinates
        y0_new = height - y0
        y1_new = height - y1
        # Ensure that upper < lower for the crop function
        upper = min(y0_new, y1_new)
        lower = max(y0_new, y1_new)
        return (x0, upper, x1, lower)

    def getSeeingConditionsFromBytes(self, imageData):
        inputImage = Image.open(io.BytesIO(imageData))
        _, height = inputImage.size  # Get image dimensions
        coordinates = {name: self.adjustCoords(coords, height) for name, coords in USER_COORDINATES.items()}

        # Crop and store sub-images in variables
        dateImage = inputImage.crop(coordinates["dateImage"])
        seeingImage = inputImage.crop(coordinates["seeingImage"])
        freeAtmSeeingImage = inputImage.crop(coordinates["freeAtmSeeingImage"])
        groundLayerImage = inputImage.crop(coordinates["groundLayerImage"])

        date = self._getDateTime(dateImage)
        seeing = self._getSeeingNumber(seeingImage)
        freeAtmSeeing = self._getSeeingNumber(freeAtmSeeingImage)
        groundLayer = self._getSeeingNumber(groundLayerImage)

        return SeeingConditions(date, seeing, freeAtmSeeing, groundLayer)

    def getLastTimestamp(self):
        """Retrieve the last timestamp from the HDFStore."""
        with pd.HDFStore(self.STORE_FILE, mode="r") as store:
            if "/data" in store.keys():
                last_row = store.select("data", start=-1)
                if not last_row.empty:
                    return last_row.index[-1]
        return None

    def run(self):
        lastTimestamp = self.getLastTimestamp()
        while True:
            if getSunAngle() > -2:
                print("Sun is too high, waiting for the SOAR seeing monitor to have a chance...")
                time.sleep(60)
                continue

            print("Fetching the current seeing conditions... ", end="")
            seeing = self.getCurrentSeeingFromWebsite()
            if seeing is None:
                print("Something went wrong in the data scraping - check the error logs.")
                time.sleep(30)
                continue
            newTimestamp = seeing.timestamp

            # Check if the new timestamp is newer than the last recorded one
            if lastTimestamp is None or newTimestamp > lastTimestamp:
                # Create a DataFrame for the new data
                print(f'seeing = {seeing.seeing}" @ {newTimestamp} UTC')
                df = pd.DataFrame(
                    {
                        "seeing": [seeing.seeing],
                        "freeAtmSeeing": [seeing.freeAtmSeeing],
                        "groundLayer": [seeing.groundLayer],
                    },
                    index=[newTimestamp],
                )

                # Append the new data to the HDFStore
                with pd.HDFStore(self.STORE_FILE, mode="a") as store:
                    store.append("data", df, format="table", data_columns=True)

                lastTimestamp = newTimestamp
            else:
                print("no updates since last time.")

            time.sleep(30)

    def _getDateTime(self, dateImage):
        dateImage = self._preprocessImage(dateImage)
        # dateImage_np = np.array(dateImage)
        results = self.reader.readtext(dateImage, detail=0)
        dateString = " ".join(results).strip()
        # replace common OCR errors
        dateString = dateString.replace(".", ":")
        dateString = dateString.replace(";", ":")
        dateString = dateString.replace("o", "0")
        dateString = dateString.replace("O", "0")
        dateString = dateString.replace("i", "1")
        dateString = dateString.replace("l", "1")
        dateString = dateString.replace("I", "1")
        dateString = dateString.replace("::", ":")
        dateString = dateString.replace("*", ":")
        dateString = dateString.replace(" :", ":")

        try:
            date_obj = datetime.strptime(dateString, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # If the format doesn't match, try alternative formats
            # For example, if there's an underscore instead of a space
            date_obj = datetime.strptime(dateString, "%Y-%m-%d_%H:%M:%S")

        return date_obj

    @staticmethod
    def thresholdImage(image, threshold):
        return image.point(lambda x: 0 if x < threshold else 255, "L")

    def _preprocessImage(self, image):
        scaleFactor = 4  # You can adjust this factor as needed
        newSize = (image.width * scaleFactor, image.height * scaleFactor)
        image = image.resize(newSize, resample=resample_filter)

        imageGrayscale = image.convert("L")

        enhancer = ImageEnhance.Contrast(imageGrayscale)
        imageGrayContrastEnhanced = enhancer.enhance(2.0)  # Increase contrast by a factor of 2

        threshold = 128

        imageBw = self.thresholdImage(imageGrayContrastEnhanced, threshold)
        imageBwNumpy = np.array(imageBw).astype(np.uint8)
        return imageBwNumpy

    def _getSeeingNumber(self, image):
        image = self._preprocessImage(image)

        results = self.reader.readtext(image, detail=0, contrast_ths=0.1, adjust_contrast=0.5)
        seeing_text = " ".join(results).strip()

        match = re.search(r"[-+]?\d*\.\d+|\d+", seeing_text)
        if match:
            seeing_value = float(match.group())
            return seeing_value
        else:
            print("No numerical value found in the OCR result.")
            return float("nan")
