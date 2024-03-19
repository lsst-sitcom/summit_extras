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

from time import sleep
from typing import Any

import numpy as np

import lsst.afw.cameraGeom.utils as cgUtils
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.pex.exceptions import NotFoundError
from lsst.summit.utils.bestEffort import BestEffortIsr
from lsst.summit.utils.butlerUtils import (
    getExpIdFromDayObsSeqNum,
    getExpRecordFromDataId,
    getMostRecentDataId,
    makeDefaultLatissButler,
)

# TODO: maybe add option to create display and return URL?


class Monitor:
    """Create a monitor for AuxTel.

    Scans the butler repo for new images and sends each one, after running
    bestEffortIsr, to the display.

    Now largely superceded by RubinTV.

    Parameters
    -------
    fireflyDisplay : `lsst.afw.display.Display`
        A Firefly display instance.
    """

    cadence = 1  # in seconds
    runIsr = True

    def __init__(self, fireflyDisplay: afwDisplay, **kwargs: Any):
        """"""
        self.butler = makeDefaultLatissButler()
        self.display = fireflyDisplay
        self.bestEffort = BestEffortIsr(**kwargs)
        self.writeQuickLookImages = None
        self.overlayAmps = False  # do the overlay?
        self.measureFromChipCenter = False

    def _getLatestImageDataIdAndExpId(self) -> tuple:
        """Get the dataId and expId for the most recent image in the repo."""
        dataId = getMostRecentDataId(self.butler)
        expId = getExpIdFromDayObsSeqNum(self.butler, dataId)["exposure"]
        return dataId, expId

    def _calcImageStats(self, exp: afwImage.Exposure) -> list[str]:
        elements = []
        median = np.median(exp.image.array)
        elements.append(f"Median={median:.2f}")
        mean = np.mean(exp.image.array)
        # elements.append(f"Median={median:.2f}")
        elements.append(f"Mean={mean:.2f}")

        return elements

    def _makeImageInfoText(self, dataId: dict, exp: afwImage.Exposure, asList: bool = False) -> list | str:
        # TODO: add the following to the display:
        # az, el, zenith angle
        # main source centroid
        # PSF
        # num saturated pixels (or maybe just an isSaturated bool)
        # main star max ADU (post isr)

        elements = []

        expRecord = getExpRecordFromDataId(self.butler, dataId)
        imageType = expRecord.observation_type
        obj = None
        if imageType.upper() not in ["BIAS", "DARK", "FLAT"]:
            try:
                obj = expRecord.target_name
                obj = obj.replace(" ", "")
            except Exception:
                pass

        for k, v in dataId.items():  # dataId done per line for vertical display
            elements.append(f"{k}:{v}")

        if obj:
            elements.append(f"{obj}")
        else:
            elements.append(f"{imageType}")

        expTime = exp.getInfo().getVisitInfo().getExposureTime()
        filt = exp.filter.physicalLabel

        elements.append(f"{expTime}s exp")
        elements.append(f"{filt}")

        elements.extend(self._calcImageStats(exp))

        if asList:
            return elements
        return " ".join([e for e in elements])

    def _printImageInfo(self, elements: list) -> None:
        size = 3
        top = 3850  # just under title for size=3
        xnom = -600  # 0 is the left edge of the image
        vSpacing = 100  # about right for size=3, can make f(size) if needed

        # TODO: add a with buffering and a .flush()
        # Also maybe a sleep as it seems buggy
        for i, item in enumerate(elements):
            y = top - (i * vSpacing)
            x = xnom + (size * 18.5 * len(item) // 2)
            self.display.dot(str(item), x, y, size, ctype="red", fontFamily="courier")

    def run(self, durationInSeconds: int = -1) -> None:
        """Run the monitor, displaying new images as they are taken.

        Parameters
        ----------
        durationInSeconds : `int`, optional
            How long to run for. Use -1 for infinite.
        """

        if durationInSeconds == -1:
            nLoops = int(1e9)
        else:
            nLoops = int(durationInSeconds // self.cadence)

        lastDisplayed = -1
        for i in range(nLoops):
            try:
                dataId, expId = self._getLatestImageDataIdAndExpId()

                if lastDisplayed == expId:
                    sleep(self.cadence)
                    continue

                if self.runIsr:
                    exp = self.bestEffort.getExposure(dataId)
                else:
                    exp = self.butler.get("raw", dataId=dataId)

                # TODO: add logic to deal with amp overlay and chip center
                # being mutually exclusive
                if self.measureFromChipCenter:  # after writing only!
                    exp.setXY0(geom.PointI(-2036, -2000))

                print(f"Displaying {dataId}...")
                imageInfoText = self._makeImageInfoText(dataId, exp, asList=True)
                # too long of a title breaks Java FITS i/o
                fireflyTitle = " ".join([s for s in imageInfoText])[:67]
                try:
                    self.display.scale("asinh", "zscale")
                    self.display.mtv(exp, title=fireflyTitle)
                except Exception as e:  # includes JSONDecodeError, HTTPError, anything else
                    print(f"Caught error {e}, skipping this image")  # TODO: try again maybe?

                if self.overlayAmps:
                    cgUtils.overlayCcdBoxes(exp.getDetector(), display=self.display, isTrimmed=True)

                self._printImageInfo(imageInfoText)
                lastDisplayed = expId

            except NotFoundError as e:  # NotFoundError when filters aren't defined
                print(f"Skipped displaying {dataId} due to {e}")
        return
