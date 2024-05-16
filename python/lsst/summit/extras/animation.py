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

import gc
import logging
import math
import os
import shutil
import subprocess
import uuid
from typing import Any

import matplotlib.pyplot as plt

import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.daf.butler as dafButler
import lsst.meas.algorithms as measAlg
from lsst.atmospec.utils import airMassFromRawMetadata, getTargetCentroidFromWcs
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask, QuickFrameMeasurementTaskConfig
from lsst.summit.utils.butlerUtils import (
    getDayObs,
    getExpRecordFromDataId,
    getLatissOnSkyDataIds,
    getSeqNum,
    makeDefaultLatissButler,
    updateDataIdOrDataCord,
)
from lsst.summit.utils.utils import dayObsIntToString, setupLogging

logger = logging.getLogger("lsst.summit.extras.animation")
setupLogging()


class Animator:
    """Animate the list of dataIds in the order in which they are specified
    for the data product specified."""

    def __init__(
        self,
        butler: dafButler.Butler,
        dataIdList: list[dict],
        outputPath: str,
        outputFilename: str,
        *,
        remakePngs: bool = False,
        clobberVideoAndGif: bool = False,
        keepIntermediateGif: bool = False,
        smoothImages: bool = True,
        plotObjectCentroids: bool = True,
        useQfmForCentroids: bool = False,
        dataProductToPlot: str = "calexp",
        debug: bool = False,
    ):
        self.butler = butler
        self.dataIdList = dataIdList
        self.outputPath = outputPath
        self.outputFilename = os.path.join(outputPath, outputFilename)
        if not self.outputFilename.endswith(".mp4"):
            self.outputFilename += ".mp4"
        self.pngPath = os.path.join(outputPath, "pngs/")

        self.remakePngs = remakePngs
        self.clobberVideoAndGif = clobberVideoAndGif
        self.keepIntermediateGif = keepIntermediateGif
        self.smoothImages = smoothImages
        self.plotObjectCentroids = plotObjectCentroids
        self.useQfmForCentroids = useQfmForCentroids
        self.dataProductToPlot = dataProductToPlot
        self.debug = debug

        # zfilled at the start as animation is alphabetical
        # if you're doing more than 1e6 files you've got bigger problems
        self.toAnimateTemplate = "%06d-%s-%s.png"
        self.basicTemplate = "%s-%s.png"

        qfmTaskConfig = QuickFrameMeasurementTaskConfig()
        self.qfmTask = QuickFrameMeasurementTask(config=qfmTaskConfig)

        afwDisplay.setDefaultBackend("matplotlib")
        self.fig = plt.figure(figsize=(15, 15))
        self.disp = afwDisplay.Display(self.fig)
        self.disp.setImageColormap("gray")
        self.disp.scale("asinh", "zscale")

        self.pngsToMakeDataIds: list[dict] = []

        self.preRun()  # sets the above list

    @staticmethod
    def _strDataId(dataId: dict) -> str:
        """Make a dataId into a string suitable for use as a filename.

        Parameters
        ----------
        dataId : `dict`
            The data id.

        Returns
        -------
        strId : `str`
            The data id as a string.
        """
        if (dayObs := getDayObs(dataId)) and (seqNum := getSeqNum(dataId)):  # nicely ordered if easy
            return f"{dayObsIntToString(dayObs)}-{seqNum:05d}"

        # General case (and yeah, I should probably learn regex someday)
        dIdStr = str(dataId)
        dIdStr = dIdStr.replace(" ", "")
        dIdStr = dIdStr.replace("{", "")
        dIdStr = dIdStr.replace("}", "")
        dIdStr = dIdStr.replace("'", "")
        dIdStr = dIdStr.replace(":", "-")
        dIdStr = dIdStr.replace(",", "-")
        return dIdStr

    def dataIdToFilename(self, dataId: dict, includeNumber: bool = False, imNum: int | None = None) -> str:
        """Convert dataId to filename.

        Returns a full path+filename by default. if includeNumber then
        returns just the filename for use in temporary dir for animation."""
        if includeNumber:
            assert imNum is not None

        dIdStr = self._strDataId(dataId)

        if includeNumber:  # for use in temp dir, so not full path
            filename = self.toAnimateTemplate % (imNum, dIdStr, self.dataProductToPlot)
            return os.path.join(filename)
        else:
            filename = self.basicTemplate % (dIdStr, self.dataProductToPlot)
            return os.path.join(self.pngPath, filename)

    def exists(self, obj: Any) -> bool:
        if isinstance(obj, str):
            return os.path.exists(obj)
        raise RuntimeError("Other type checks not yet implemented")

    def preRun(self) -> None:
        # check the paths work
        if not os.path.exists(self.pngPath):
            os.makedirs(self.pngPath)
        assert os.path.exists(self.pngPath), f"Failed to create output dir: {self.pngPath}"

        if self.exists(self.outputFilename):
            if self.clobberVideoAndGif:
                os.remove(self.outputFilename)
            else:
                raise RuntimeError(f"Output file {self.outputFilename} exists and clobber==False")

        # make list of found & missing files
        dIdsWithPngs = [d for d in self.dataIdList if self.exists(self.dataIdToFilename(d))]
        dIdsWithoutPngs = [d for d in self.dataIdList if d not in dIdsWithPngs]
        if self.debug:
            logger.info(f"dIdsWithPngs = {dIdsWithPngs}")
            logger.info(f"dIdsWithoutPngs = {dIdsWithoutPngs}")

        # check the datasets exist for the pngs which need remaking
        missingData = [
            d for d in dIdsWithoutPngs if not self.butler.exists(self.dataProductToPlot, d, detector=0)
        ]

        logger.info(f"Of the provided {len(self.dataIdList)} dataIds:")
        logger.info(f"{len(dIdsWithPngs)} existing pngs were found")
        logger.info(f"{len(dIdsWithoutPngs)} do not yet exist")

        if missingData:
            for dId in missingData:
                msg = f"Failed to find {self.dataProductToPlot} for {dId}"
                logger.warning(msg)
                self.dataIdList.remove(dId)
            logger.info(
                f"Of the {len(dIdsWithoutPngs)} dataIds without pngs, {len(missingData)}"
                " did not have the corresponding dataset existing"
            )

        if self.remakePngs:
            self.pngsToMakeDataIds = [d for d in self.dataIdList if d not in missingData]
        else:
            self.pngsToMakeDataIds = [d for d in dIdsWithoutPngs if d not in missingData]

        msg = f"So {len(self.pngsToMakeDataIds)} will be made"
        if self.remakePngs and len(dIdsWithPngs) > 0:
            msg += " because remakePngs=True"
        logger.info(msg)

    def run(self) -> str | None:
        # make the missing pngs
        if self.pngsToMakeDataIds:
            logger.info("Creating necessary pngs...")
            for i, dataId in enumerate(self.pngsToMakeDataIds):
                logger.info(f"Making png for file {i+1} of {len(self.pngsToMakeDataIds)}")
                self.makePng(dataId, self.dataIdToFilename(dataId))

        # stage files in temp dir with numbers prepended to filenames
        if not self.dataIdList:
            logger.warning("No files to animate - nothing to do")
            return None

        logger.info("Copying files to ordered temp dir...")
        pngFilesOriginal = [self.dataIdToFilename(d) for d in self.dataIdList]
        for filename in pngFilesOriginal:  # these must all now exist, but let's assert just in case
            assert self.exists(filename)
        tempDir = os.path.join(self.pngPath, f"{uuid.uuid1()}/"[0:8])
        os.makedirs(tempDir)
        pngFileList = []  # list of number-prepended files in the temp dir
        for i, dId in enumerate(self.dataIdList):
            srcFile = self.dataIdToFilename(dId)
            destFile = os.path.join(tempDir, self.dataIdToFilename(dId, includeNumber=True, imNum=i))
            shutil.copy(srcFile, destFile)
            pngFileList.append(destFile)

        # # create gif in temp dir
        # outputGifFilename = os.path.join(tempDir, 'animation.gif')
        # self.pngsToGif(pngFileList, outputGifFilename)

        # # gif turn into mp4, optionally keep gif by moving up to output dir
        # logger.info('Turning gif into mp4...')
        # outputMp4Filename = self.outputFilename
        # self.gifToMp4(outputGifFilename, outputMp4Filename)

        # self.tidyUp(tempDir)
        # logger.info('Finished!')

        # create gif in temp dir

        logger.info("Making mp4 of pngs...")
        self.pngsToMp4(tempDir, self.outputFilename, 10, verbose=False)
        self.tidyUp(tempDir)
        logger.info(f"Finished! Output at {self.outputFilename}")
        return self.outputFilename

    def _titleFromExp(self, exp: afwImage.Exposure, dataId: dict) -> str:
        expRecord = getExpRecordFromDataId(self.butler, dataId)
        obj = expRecord.target_name
        expTime = expRecord.exposure_time
        filterCompound = expRecord.physical_filter
        filt, grating = filterCompound.split("~")
        rawMd = self.butler.get("raw.metadata", dataId)
        airmass = airMassFromRawMetadata(rawMd)  # XXX this could be improved a lot
        if not airmass:
            airmass = -1
        dayObs = dayObsIntToString(getDayObs(dataId))
        timestamp = expRecord.timespan.begin.to_datetime().strftime("%H:%M:%S")  # no microseconds
        ms = expRecord.timespan.begin.to_datetime().strftime("%f")  # always 6 chars long, 000000 if zero
        timestamp += f".{ms[0:2]}"
        title = f"seqNum {getSeqNum(dataId)} - {dayObs} {timestamp}TAI - "
        title += f"Object: {obj} expTime: {expTime}s Filter: {filt} Grating: {grating} Airmass: {airmass:.3f}"
        return title

    def getStarPixCoord(
        self, exp: Any, doMotionCorrection: bool = True, useQfm: bool = False
    ) -> tuple[float, float] | None:
        target = exp.visitInfo.object

        if self.useQfmForCentroids:
            try:
                result = self.qfmTask.run(exp)
                pixCoord = result.brightestObjCentroid
                expId = exp.info.id
                logger.info(f"expId {expId} has centroid {pixCoord}")
            except Exception:
                return None
        else:
            pixCoord = getTargetCentroidFromWcs(exp, target, doMotionCorrection=doMotionCorrection)
        return pixCoord

    def makePng(self, dataId: dict, saveFilename: str) -> None:
        if self.exists(saveFilename) and not self.remakePngs:  # should not be possible due to prerun
            assert False, f"Almost overwrote {saveFilename} - how is this possible?"

        if self.debug:
            logger.info(f"Creating {saveFilename}")

        self.fig.clear()

        # must always keep exp unsmoothed for the centroiding via qfm
        try:
            exp = self.butler.get(self.dataProductToPlot, dataId)
        except Exception:
            # oh no, that should never happen, but it does! Let's just skip
            logger.warning(f"Skipped {dataId}, because {self.dataProductToPlot} retrieval failed!")
            return
        toDisplay = exp
        if self.smoothImages:
            toDisplay = exp.clone()
            toDisplay = self._smoothExp(toDisplay, 2)

        try:
            self.disp.mtv(toDisplay.image, title=self._titleFromExp(exp, dataId))
            self.disp.scale("asinh", "zscale")
        except RuntimeError:  # all-nan images slip through and don't display
            self.disp.scale("linear", 0, 1)
            self.disp.mtv(toDisplay.image, title=self._titleFromExp(exp, dataId))
            self.disp.scale("asinh", "zscale")  # set back for next image
            pass

        if self.plotObjectCentroids:
            try:
                pixCoord = self.getStarPixCoord(exp)
                if pixCoord:
                    self.disp.dot("x", *pixCoord, ctype="C1", size=50)
                    self.disp.dot("o", *pixCoord, ctype="C1", size=50)
                else:
                    self.disp.dot("x", 2000, 2000, ctype="red", size=2000)
            except Exception:
                logger.warning(f"Failed to find OBJECT location for {dataId}")

        deltaH = -0.05
        deltaV = -0.05
        plt.subplots_adjust(right=1 + deltaH, left=0 - deltaH, top=1 + deltaV, bottom=0 - deltaV)
        self.fig.savefig(saveFilename)
        logger.info(f"Saved png for {dataId} to {saveFilename}")

        del toDisplay
        del exp
        gc.collect()

    def pngsToMp4(self, indir: str, outfile: str, framerate: float, verbose: bool = False) -> None:
        """Create the movie with ffmpeg, from files."""
        # NOTE: the order of ffmpeg arguments *REALLY MATTERS*.
        # Reorder them at your own peril!
        pathPattern = f'"{os.path.join(indir, "*.png")}"'
        if verbose:
            ffmpeg_verbose = "info"
        else:
            ffmpeg_verbose = "error"
        cmd = [
            "ffmpeg",
            "-v",
            ffmpeg_verbose,
            "-f",
            "image2",
            "-y",
            "-pattern_type glob",
            "-framerate",
            f"{framerate}",
            "-i",
            pathPattern,
            "-vcodec",
            "libx264",
            "-b:v",
            "20000k",
            "-profile:v",
            "main",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "10",
            "-r",
            f"{framerate}",
            os.path.join(outfile),
        ]

        subprocess.check_call(r" ".join(cmd), shell=True)

    def tidyUp(self, tempDir: str) -> None:
        shutil.rmtree(tempDir)
        return

    def _smoothExp(self, exp: afwImage.Exposure, smoothing: float, kernelSize: int = 7) -> afwImage.Exposure:
        """Use for DISPLAY ONLY!

        Return a smoothed copy of the exposure
        with the original mask plane in place."""
        psf = measAlg.DoubleGaussianPsf(kernelSize, kernelSize, smoothing / (2 * math.sqrt(2 * math.log(2))))
        newExp = exp.clone()
        originalMask = exp.mask

        kernel = psf.getKernel()
        afwMath.convolve(newExp.maskedImage, newExp.maskedImage, kernel, afwMath.ConvolutionControl())
        newExp.mask = originalMask
        return newExp


def animateDay(
    butler: dafButler.Butler, dayObs: int, outputPath: str, dataProductToPlot: str = "quickLookExp"
) -> str | None:
    outputFilename = f"{dayObs}.mp4"

    onSkyIds = getLatissOnSkyDataIds(butler, startDate=dayObs, endDate=dayObs)
    logger.info(f"Found {len(onSkyIds)} on sky ids for {dayObs}")

    onSkyIds = [updateDataIdOrDataCord(dataId, detector=0) for dataId in onSkyIds]

    animator = Animator(
        butler,
        onSkyIds,
        outputPath,
        outputFilename,
        dataProductToPlot=dataProductToPlot,
        remakePngs=False,
        debug=False,
        clobberVideoAndGif=True,
        plotObjectCentroids=True,
        useQfmForCentroids=True,
    )
    filename = animator.run()
    return filename


if __name__ == "__main__":
    outputPath = "/home/mfl/animatorOutput/main/"
    butler = makeDefaultLatissButler()

    day = 20211104
    animateDay(butler, day, outputPath)
