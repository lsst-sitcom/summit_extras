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

import gc
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import lsst.afw.display as afwDisplay
from lsst.pipe.tasks.visualizeVisit import VisualizeMosaicExpTask

from .utils import dayObsIntToString, getCameraFromInstrumentName
from .plotting import plot

if TYPE_CHECKING:
    from lsst.afw.image import Exposure

    from lsst.daf.butler import Butler, DimensionRecord


def exists(filename: str) -> bool:
    return os.path.exists(filename)


class Animator:
    """Animate the list of dataIds in the order in which they are specified
    for the data product specified."""

    def __init__(
        self,
        butler: Butler,
        pngPath: str,
        *,
        dataProductToPlot: str = "calexp",
        binning: int = 8,
    ) -> None:
        self.log = logging.getLogger("lsst.summit.extras.animation")
        self.butler = butler
        self.pngPath = pngPath
        if not os.path.exists(self.pngPath):
            os.makedirs(self.pngPath)
        assert os.path.exists(self.pngPath), f"Failed to create output dir: {self.pngPath}"

        self.dataProductToPlot = dataProductToPlot

        afwDisplay.setDefaultBackend("matplotlib")
        self.fig = plt.figure(figsize=(15, 15))
        self.mosaicTask = VisualizeMosaicExpTask()
        # self.disp = afwDisplay.Display(self.fig)
        # self.disp.setImageColormap("gray")
        # self.disp.scale("asinh", "zscale")

    def animate(
        self,
        records: list[DimensionRecord],
        outputFilename: str,
        framerate: float = 10,
        forceReRender: bool = False,
        ignoreCalibs: bool = False,
        ignoreCwfs: bool = False,
    ) -> None:
        if not outputFilename.endswith(".mp4"):
            # TODO: check this is a valid path & file?
            raise ValueError("Output file must be a fully qualified path to an .mp4 file")
        if exists(self.outputFilename):
            os.remove(self.outputFilename)

        if len(set(r.instrument for r in records)) != 1:
            raise ValueError("Cannot mix instruments in animation")

        if ignoreCalibs:
            records = [r for r in records if not isCalibration(r)]
        if ignoreCwfs:
            records = [r for r in records if not isWepImage(r)]

        recordsToRender = [r for r in records if not exists(self.recordToFilename(r))] if not forceReRender else records
        existing = list(set(records) - set(recordsToRender))
        # XXX deal with missing data here
        # including incomplete focal planes (important for LSSTCam)
        # maybe also want to flag which are partial and re-render those each time but not complete ones?
        # missingData = [d for d in dIdsWithoutPngs if not self.butler.exists(self.dataProductToPlot, d, detector=0)]
        self.log.info(f"Of the {len(records)} records, {len(existing)} existing pngs were found, {len(recordsToRender)} to render")

        toCopy = []
        for i, record in enumerate(recordsToRender):
            self.log.info(f"Making png for file {i+1} of {len(recordsToRender)}")
        created = self.makePng(record)
        toCopy.append(created)

        # TODO see if the files can be supplied in another way to ffmepg to
        # avoid this copying. Note though that it must maintain the ordering,
        # and also support extremely long arg lists: the filenames are long,
        # and might number in the thousands so be careful the new approach is
        # fully general if replacing this.
        with tempfile.TemporaryDirectory() as tempDir:
            self.log.info("Copying files to ordered temp dir...")
            for i, srcFile in enumerate(toCopy):
                destFile = os.path.join(tempDir, f"{i:06}.png")
                shutil.copy(srcFile, destFile)

            self.log.info("Making mp4 of pngs...")
            self.pngsToMp4(tempDir, self.outputFilename, framerate, verbose=False)
            self.log.info(f"Finished making movie of {len(records)} images, file written to {outputFilename}")
        return

    def recordToFilename(self, record: DimensionRecord) -> str:
        """Convert the dimension record to a filename.
        """
        filename = f"{record.id}-{self.dataProductToPlot}.png"
        return os.path.join(self.pngPath, str(record.day_obs), filename)

    def makeTitle(self, record: DimensionRecord) -> str:
        # XXX check record type here
        # and check what's available on a visit record, ensure this works for both
        obj = record.target_name
        expTime = record.exposure_time
        _filter = record.physical_filter
        # airmass ?
        dayObs = dayObsIntToString(record.day_obs)
        seqNum = record.seq_num  # XXX deal with visit record here
        timestamp = record.timespan.begin.to_datetime().strftime("%H:%M:%S")  # no microseconds
        ms = record.timespan.begin.to_datetime().strftime("%f")  # always 6 chars long, 000000 if zero
        timestamp += f".{ms[0:2]}"
        title = f"seqNum {seqNum} - {dayObs} {timestamp}TAI - "
        title += f"Object: {obj} expTime: {expTime}s Filter: {_filter}"  # Airmass: {airmass:.3f}"
        return title

    def getMosaicImage(self, record: DimensionRecord) -> tuple[Exposure, bool]:
        camera = getCameraFromInstrumentName(record.instrument)
        dRefs = self.butler.registry.queryDatasets(self.dataProductToPlot, dataId=record.dataId)
        # XXX need to workout how to get the pre-binned things here
        # probably just force using that dataset type on init. Add check
        # that the binning matches
        t0 = time.time()
        exps = [butler.get(d) for d in dRefs]
        self.log.info(f"Loading the {len(dRefs)} exposures took {(time.time()-t0):.2f} seconds")

        complete = len(exps) == len(camera)

        t0 = time.time()
        result = self.mosaicTask.run(exps, camera)
        self.log.info(f"Making the mosaic itself took {(time.time()-t0):.2f} seconds")
        return result.image, complete  # it's called .image but it's an Exposure

    def makePng(self, record: DimensionRecord) -> str:
        # XXX add option for mask planes? Probably not
        saveFilename = self.recordToFilename(record)
        if exists(saveFilename):
            os.remove(saveFilename)

        exposure, complete = self.getMosaicImage(record)
        self.fig.clear()

        # XXX if no adjustment needed below just pass saving option in here
        plot(exposure, figure=self.fig, stretch='zscale')

        # try:
        #     self.disp.mtv(exposure.image, title=self.makeTitle(record))
        #     self.disp.scale("asinh", "zscale")
        # except RuntimeError:  # all-nan images slip through and raise with asinh
        #     self.disp.scale("linear", 0, 1)
        #     self.disp.mtv(exposure.image, title=self.makeTitle(record))
        #     self.disp.scale("asinh", "zscale")  # set back for next image

        # deltaH = -0.05
        # deltaV = -0.05
        # XXX adjust may not be necessary once not usin afwDisplay
        # fig.subplots_adjust(right=1 + deltaH, left=0 - deltaH, top=1 + deltaV, bottom=0 - deltaV)
        self.fig.savefig(saveFilename)

        del exposure
        gc.collect()
        return saveFilename

    def pngsToMp4(self, orderedPngDir: str, outfile: str, framerate: float, verbose: bool = False) -> None:
        """Create the movie with ffmpeg, from files."""
        # NOTE: the order of ffmpeg arguments *REALLY MATTERS*.
        # Reorder them at your own peril!
        pathPattern = f'"{os.path.join(orderedPngDir, "*.png")}"'
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

    def removePngsForDayObs(self, dayObs: str) -> None:
        path = os.path.join(self.pngPath, str(dayObs))
        shutil.rmtree(path)


def animateDay(
    butler: Butler,
    dayObs: int,
    instrument: str,
    pngPath: str,
    outputFilename: str,
    ignoreCalibs: bool = True,
    dataProductToPlot: str = "postISRCCD"
) -> None:
    where = f"exposure.day_obs={dayObs} AND instrument='{instrument}'"
    records = list(butler.registry.queryDimensionRecords('exposure', where=where))
    records = sorted(records, key=lambda x: (x.day_obs, x.seq_num))

    animator = Animator(
        butler=butler,
        pngPath=pngPath,
        dataProductToPlot=dataProductToPlot,
    )
    animator.animate(records, outputFilename, ignoreCalibs=ignoreCalibs)


if __name__ == "__main__":
    dayObs = 20241206
    pngPath = "/home/mfl/animatorOutput/main/"
    outFile = f"/home/mfl/animatorOutput/main/{dayObs}.mp4"
    import lsst.summit.utils.butlerUtils as butlerUtils
    butler = butlerUtils.makeDefaultButler("LSSTComCam")

    animateDay(butler, dayObs, "LSSTComCam", pngPath, outFile)
