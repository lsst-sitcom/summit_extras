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
import typing
from dataclasses import dataclass

import galsim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.summit.utils.starTracker import (
    dayObsSeqNumFrameNumFromFilename,
    fastCam,
    getRawDataDirForDayObs,
    isStreamingModeFile,
    openFile,
)
from lsst.summit.utils.utils import bboxToMatplotlibRectanle, detectObjectsInExp, getBboxAround, getSite
from lsst.utils.iteration import ensure_iterable

__all__ = (
    "getStreamingSequences",
    "getFlux",
    "getBackgroundLevel",
    "countOverThresholdPixels",
    "sortSourcesByFlux",
    "findFastStarTrackerImageSources",
    "checkResultConsistency",
    "plotSourceMovement",
    "plotSource",
    "plotSourcesOnImage",
    "Source",
    "NanSource",
)


@dataclass(slots=True)
class Source:
    """A dataclass for FastStarTracker analysis results."""

    dayObs: int  # mandatory attribute - the dayObs
    seqNum: int  # mandatory attribute - the seqNum
    frameNum: int  # mandatory attribute - the sub-sequence number, the position in the sequence

    # raw numbers
    centroidX: float = np.nan  # in image coordinates
    centroidY: float = np.nan  # in image coordinates
    rawFlux: float = np.nan
    nPix: int | float = np.nan
    bbox: geom.Box2I | None = None
    cutout: np.ndarray | None = None
    localCentroidX: float = np.nan  # in cutout coordinates
    localCentroidY: float = np.nan  # in cutout coordinates

    # numbers from the hsm moments fit
    hsmFittedFlux: float = np.nan
    hsmCentroidX: float = np.nan
    hsmCentroidY: float = np.nan
    moments: galsim.hsm.ShapeData | None = None  # keep the full fit even though we pull some things out too

    imageBackground: float = np.nan
    imageStddev: float = np.nan
    nSourcesInImage: int | float = np.nan
    parentImageWidth: int | float = np.nan
    parentImageHeight: int | float = np.nan
    expTime: float = np.nan

    def __repr__(self):
        """Print everything except the full details of the moments."""
        retStr = ""
        for itemName in self.__slots__:
            v = getattr(self, itemName)
            if isinstance(v, int):  # print ints as ints
                retStr += f"{itemName} = {v}\n"
            elif isinstance(v, float):  # but round floats at 3dp
                retStr += f"{itemName} = {v:.3f}\n"
            elif itemName == "moments":  # and don't spam the full moments
                retStr += f"moments = {type(v)}\n"
            elif itemName == "bbox":  # and don't spam the full moments
                retStr += f"bbox = lsst.geom.{repr(v)}\n"
            elif itemName == "cutout":  # and don't spam the full moments
                if v is None:
                    retStr += "cutout = None\n"
                else:
                    retStr += f"cutout = {type(v)}\n"
        return retStr


class NanSource:
    def __getattribute__(self, name: str):
        return np.nan


def getStreamingSequences(dayObs: int) -> dict[int, list[str]]:
    """Get the streaming sequences for a dayObs.

    Note that this will need rewriting very soon once the way the data is
    organised on disk is changed.

    Parameters
    ----------
    dayObs : `int`
        The dayObs.

    Returns
    -------
    sequences : `dict` [`int`, `list`]
        The streaming sequences in a dict, keyed by sequence number, with each
        value being a list of the files in that sequence.
    """
    site = getSite()
    if site in ["rubin-devl", "staff-rsp"]:
        rootDataPath = "/sdf/data/rubin/offline/s3-backup/lfa/"
    elif site == "summit":
        rootDataPath = "/project"
    else:
        raise ValueError(f"Finding StarTracker data isn't supported at {site}")

    dataDir = getRawDataDirForDayObs(rootDataPath, fastCam, dayObs)
    files = glob.glob(os.path.join(dataDir, "*.fits"))
    regularFiles = [f for f in files if not isStreamingModeFile(f)]
    streamingFiles = [f for f in files if isStreamingModeFile(f)]
    print(f"Found {len(regularFiles)} regular files on dayObs {dayObs}")

    data = {}
    if dayObs < 20240311:
        # after this is when we changed the data layout on disk for streaming
        # mode data in the GenericCamera
        for filename in sorted(streamingFiles):
            basename = os.path.basename(filename)
            seqNum = int(basename.split("_")[3])
            if seqNum not in data:
                data[seqNum] = [filename]
            else:
                data[seqNum].append(filename)
    else:
        # dirNames here doesn't contain the full path, it's just the individual
        # directory name and needs joining with dataDir for the full path
        dirNames = sorted(d for d in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, d)))
        for d in dirNames:
            files = sorted(glob.glob(os.path.join(dataDir, d, "*.fits")))
            seqNum = int(d.split("_")[3])
            data[seqNum] = files

    print(f"Found {len(data)} streaming sequences on dayObs {dayObs}:")
    for seqNum, files in data.items():
        print(f"seqNum {seqNum} with {len(files)} frames")

    return data


def getFlux(cutout: np.ndarray[float], backgroundLevel: float = 0) -> float:
    """Get the flux inside a cutout, subtracting the image-background.

    Here the flux is simply summed, and if the image background level is
    supplied, it is subtracted off, assuming it is constant over the cutout. A
    more accurate(?) flux is obtained by the hsm model fit.

    Parameters
    ----------
    cutout : `np.array`
        The cutout as a raw array.
    backgroundLevel : `float`, optional
        If supplied, this is subtracted as a constant background level.

    Returns
    -------
    flux : `float`
        The flux of the source in the cutout.
    """
    rawFlux = np.sum(cutout)
    if not backgroundLevel:
        return rawFlux

    return rawFlux - (cutout.size * backgroundLevel)


def getBackgroundLevel(exp: afwImage.Exposure, nSigma: float = 3) -> tuple[float, float]:
    """Calculate the clipped image mean and stddev of an exposure.

    Testing shows on images like this, 2 rounds of sigma clipping is more than
    enough so this is left fixed here.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    nSigma : `float`, optional
        The number of sigma to clip to for the background estimation.

    Returns
    -------
    mean : `float`
        The clipped mean, as an estimate of the background level
    stddev : `float`
        The clipped standard deviation, as an estimate of the background noise.
    """
    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(nSigma)
    sctrl.setNumIter(2)  # this is always plenty here
    statTypes = afwMath.MEANCLIP | afwMath.STDEVCLIP
    stats = afwMath.makeStatistics(exp.maskedImage, statTypes, sctrl)
    std, _ = stats.getResult(afwMath.STDEVCLIP)
    mean, _ = stats.getResult(afwMath.MEANCLIP)
    return mean, std


def countOverThresholdPixels(cutout: np.ndarray, bgMean: float, bgStd: float, nSigma: float = 15) -> int:
    """Get the number of pixels in the cutout which are 'in the source'.

    From the one image I've looked at so far, the drop-off is quite slow
    probably due to some combination of focus, plate scale, star brightness,
    pointing quality etc, so the default nSigma is 15 here as that looked about
    right when I plotted it by eye.

    Parameters
    ----------
    cutout : `np.array`
        The cutout to measure.
    bgMean : `float`
        The background level.
    bgStd : `float`
        The clipped standard deviation in the image.
    nSigma : `float`, optional
        The number of sigma above background at which to count pixels as being
        over threshold.

    Returns
    -------
    nPix : `int`
        The number of pixels above threshold.
    """
    inds = np.where(cutout > (bgMean + 0 * bgStd))
    return len(inds[0])


def sortSourcesByFlux(sources: list[Source], reverse: bool = False) -> list[Source]:
    """Sort the sources by flux, returning the brightest first.

    Parameters
    ----------
    sources : `list` of
              `lsst.summit.extras.fastStarTrackerAnalysis.Source`
        The list of sources to sort.
    reverse : `bool`, optional
        Return the brightest at the start of the list if ``reverse`` is
        ``False``, or the brightest last if ``reverse`` is ``True``.

    Returns
    -------
    sources : `list` of
              `lsst.summit.extras.fastStarTrackerAnalysis.Source`
        The sources, sorted by flux.
    """
    # invert reverse because we want brightest first by default, but want the
    # reverse arg to still behave as one would expect
    return sorted(sources, key=lambda s: s.rawFlux, reverse=not reverse)


def findFastStarTrackerImageSources(
    filename: str, boxSize: int, attachCutouts: bool = True
) -> list[Source | NanSource]:
    """Analyze a single FastStarTracker image.

    Parameters
    ----------
    filename : `str`
        The full name and path of the file.
    boxSize : `int`
        The size of the box to put around each source for measurement.
    attachCutouts : `bool`, optional
        Attach the cutouts to the ``Source`` objects? Useful for
        debug/plotting but adds memory usage.

    Returns
    -------
    sources : `list` of
              `lsst.summit.extras.fastStarTrackerAnalysis.Source`
        The sources in the image, sorted by rawFlux.
    """
    exp = openFile(filename)
    # if the upstream exposure reading code hasn't set the
    # visitInfo.exposureTime then this will return nan, as desired
    expTime = exp.visitInfo.exposureTime
    footprintSet = detectObjectsInExp(exp)
    footprints = footprintSet.getFootprints()
    bgMean, bgStd = getBackgroundLevel(exp)

    dayObs, seqNum, frameNum = dayObsSeqNumFrameNumFromFilename(filename)

    sources = []
    if len(footprints) == 0:
        sources = [NanSource()]
        return sources

    for footprint in footprints:
        source = Source(dayObs=dayObs, seqNum=seqNum, frameNum=frameNum)
        source.expTime = expTime
        source.nSourcesInImage = len(footprints)
        source.parentImageWidth, source.parentImageHeight = exp.getDimensions()

        centroid = footprint.getCentroid()
        bbox = getBboxAround(centroid, boxSize, exp)
        source.bbox = bbox
        cutout = exp.image[bbox].array
        if attachCutouts:
            source.cutout = cutout
        source.centroidX = centroid[0]
        source.centroidY = centroid[1]
        source.rawFlux = getFlux(cutout, bgMean)
        source.imageBackground = bgMean
        source.imageStddev = bgStd
        source.nPix = countOverThresholdPixels(cutout, bgMean, bgStd)

        moments = galsim.hsm.FindAdaptiveMom(galsim.Image(cutout))
        source.moments = moments
        source.hsmFittedFlux = moments.moments_amp
        source.hsmCentroidX = moments.moments_centroid.x + bbox.minX - 1
        source.hsmCentroidY = moments.moments_centroid.y + bbox.minY - 1
        source.localCentroidX = moments.moments_centroid.x - 1
        source.localCentroidY = moments.moments_centroid.y - 1
        sources.append(source)
    return sortSourcesByFlux(sources)


def checkResultConsistency(
    results: dict[int, list[Source]],
    maxAllowableShift: float = 5,
    silent: bool = False,
) -> bool:
    """Check if a set of results are self-consistent.

    Check the number of detected sources are the same in each image, that no
    sources have been removed from each image's source list, and that all the
    input images were the same size (because we read out sub frames, and
    analyzing these with full frame data invalidates the centroid coordinates).

    Also displays the maximum (x, y) movements between adjacent exposures, and
    the mean and stddev of the main source's flux.

    Parameters
    ----------
    results : `dict` of `list` of
              `lsst.summit.extras.fastStarTrackerAnalysis.Source`
        A dict, keyed by sequence number, with each value being a list of the
        sources found in the image, e.g. as returned by
        ``findFastStarTrackerImageSources()``.
    maxAllowableShift : `float`
        The biggest centroid shift between adjacent images allowable before
        something is considered to have gone wrong.
    silent : `bool`, optional
        Print some useful checks and measurements if ``False``, otherwise just
        return whether the results appear nominally OK silently (for use when
        being called by other code rather than users).

    Returns
    -------
    consistent : `bool`
        Are the results nominally consistent?
    """
    if isinstance(results, typing.ValuesView):  # in case we're passed a .values()
        results = list(results)

    sourceCounts = set([len(sourceSet) for sourceSet in results])
    if sourceCounts == {0}:  # none of the images contain any detections
        if not silent:
            print("No images contain any sources. Results are technically consistent, but also useless.")
        # this is technically consistent, so return True, but any downstream
        # code which tries to make plots with these will fail, of course.
        return True

    if 0 in ([len(sourceSet) for sourceSet in results]):
        if not silent:
            print(
                "Some results contain no sources. Results are therefore fundamentally inconsistent"
                " and other checks cannot be run"
            )
        return False

    consistent = True
    toPrint = []
    nSources = set([sourceSet[0].nSourcesInImage for sourceSet in results])
    if len(nSources) != 1:
        toPrint.append(f"❌ Images contain a variable number of sources: {nSources}")
        consistent = False
    else:
        n = nSources.pop()
        toPrint.append(f"✅ All images contain the same nominal number of sources at detection stage: {n}")

    nSourcesCounted = set([len(sourceSet) for sourceSet in results])
    if len(nSourcesCounted) != 1:
        toPrint.append(
            f"❌ Number of actual sources in each sourceSet varies, got: {nSourcesCounted}."
            " If some were manually removed you can ignore this"
        )
        consistent = False
    else:
        n = nSourcesCounted.pop()
        toPrint.append(f"✅ All results contain the same number of actual sources per image: {n}")

    widths = set([sourceSet[0].parentImageWidth for sourceSet in results])
    heights = set([sourceSet[0].parentImageHeight for sourceSet in results])
    if len(widths) != 1 or len(heights) != 1:
        toPrint.append(f"❌ Input images were of variable dimenions! {widths=}, {heights=}")
        consistent = False
    else:
        toPrint.append("✅ All input images were of the same dimensions")

    if len(results) > 1:  # can't np.diff an array of length 1 so these are not useful/defined
        # now the basic checks have passed, do some sanity checks on the
        # maximum deltas for the primary sources
        sources = [sourceSet[0] for sourceSet in results]
        dx = np.diff([s.centroidX for s in sources])
        dy = np.diff([s.centroidY for s in sources])
        maxMovementX = np.max(dx)
        maxMovementY = np.max(dy)
        happyOrSad = "✅"
        if max(maxMovementX, maxMovementY) > maxAllowableShift:
            consistent = False
            happyOrSad = "❌"

        toPrint.append(
            f"{happyOrSad} Maximum centroid movement of brightest object between images in (x, y)"
            f" = ({maxMovementX:.2f}, {maxMovementY:.2f}) pix"
        )

        fluxStd = np.nanstd([s.rawFlux for s in sources])
        fluxMean = np.nanmean([s.rawFlux for s in sources])
        toPrint.append(f"Mean and stddev of flux from brightest object = {fluxMean:.1f} ± {fluxStd:.1f} ADU")

    if not silent:
        for line in toPrint:
            print(line)

    return consistent


def plotSourceMovement(
    results: dict[int, list[Source]],
    sourceIndex: int = 0,
    allowInconsistent: bool = False,
) -> list[matplotlib.figure.Figure]:
    """Plot the centroid movements and fluxes etc for a set of results.

    By default the brightest source in each image is plotted, but this can be
    changed by setting ``sourceIndex`` to values greater than 0 to move through
    the list of sources in each image.

    Parameters
    ----------
    results : `dict` of `list` of
              `lsst.summit.extras.fastStarTrackerAnalysis.Source`
        A dict, keyed by sequence number, with each value being a list of the
        sources found in the image, e.g. as returned by
        ``findFastStarTrackerImageSources()``.
    sourceIndex : `int`, optional
        If there is more than one source in every image, which source number
        should the plot be made for? Defaults to zero, which is the brightest
        source by default.
    allowInconsistent : `bool`, optional
        Make the plots even if the input results appear to be inconsistent?

    Returns
    -------
    figs : `list` of `matplotlib.figure.Figure`
        The figures. The first is the source's flux and x, y movement over the
        image sequence, and the second is a scatter plot of the x and y, with
        the color showing the position in the sequence.
    """
    opts = {
        "marker": "o",
        "markersize": 6,
        "linestyle": "-",
    }

    consistent = checkResultConsistency(results.values(), silent=True)
    if not consistent and not allowInconsistent:
        checkResultConsistency(results.values(), silent=False)  # print the problem if we're raising
        raise ValueError("The sources were found to be inconsistent and allowInconsistent=False")

    sourceDict = {k: v[sourceIndex] for k, v in results.items()}
    frameNums = [s.frameNum for s in sourceDict.values()]
    sources = list(sourceDict.values())

    allDayObs = set(s.dayObs for s in sources)
    allSeqNums = set(s.seqNum for s in sources)
    if len(allDayObs) > 1 or len(allSeqNums) > 1:
        raise ValueError(
            "The sources are from multiple days or sequences, found"
            f" {allDayObs} dayObs and {allSeqNums} seqNum values."
        )
    dayObs = allDayObs.pop()
    seqNum = allSeqNums.pop()
    startFrame = min(frameNums)
    endFrame = max(frameNums)

    title = f"dayObs {dayObs}, seqNum {seqNum}, frames {startFrame}-{endFrame}"

    axisLabelSize = 18

    figs = []
    fig = plt.figure(figsize=(10, 16))
    ax1, ax2, ax3 = fig.subplots(3, sharex=True)
    fig.subplots_adjust(hspace=0)

    ax1.plot(frameNums, [s.rawFlux for s in sources], label="Raw Flux", **opts)
    ax1.plot(frameNums, [s.hsmFittedFlux for s in sources], label="Fitted Flux", **opts)
    ax1.set_ylabel("Flux (ADU)", size=axisLabelSize)
    ax1.set_title(title)
    ax1.legend()

    ax2.plot(frameNums, [s.centroidX for s in sources], label="Raw centroid x", **opts)
    ax2.plot(
        frameNums,
        [s.hsmCentroidX for s in sources],
        label="Fitted centroid x",
        **opts,
    )
    ax2.set_ylabel("x-centroid (pixels)", size=axisLabelSize)
    ax2.legend()

    ax3.plot(frameNums, [s.centroidY for s in sources], label="Raw centroid y", **opts)
    ax3.plot(
        frameNums,
        [s.hsmCentroidY for s in sources],
        label="Fitted centroid y",
        **opts,
    )
    ax3.set_ylabel("y-centroid (pixels)", size=axisLabelSize)
    ax3.set_xlabel("Frame number", size=axisLabelSize)
    ax3.legend()

    figs.append(fig)

    fig = plt.figure(figsize=(10, 10))
    ax4 = fig.subplots(1)

    colors = np.arange(len(sources))
    # gnuplot2 has a nice balance of nothing white, and having an intuitive
    # progression of colours so the eye can pick out trends on the point cloud.
    axRef = ax4.scatter(
        [s.centroidX for s in sources],
        [s.centroidY for s in sources],
        c=colors,
        cmap="gnuplot2",
    )
    ax4.set_xlabel("x-centroid (pixels)", size=axisLabelSize)
    ax4.set_ylabel("y-centroid (pixels)", size=axisLabelSize)
    ax4.set_aspect("equal", "box")
    # move the colorbar
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(axRef, cax=cax)
    ax4.set_title(title)
    cbar.set_label("Frame number in series", size=axisLabelSize * 0.75)
    figs.append(fig)

    return figs


# -------------- plotting tools


def plotSourcesOnImage(
    parentFilename: str,
    sources: Source | list[Source],
) -> None:
    """Plot one of more source on top of an image.

    Parameters
    ----------
    parentFilename : `str`
        The full path to the parent (.tif) file.
    sources : `list` of
              `lsst.summit.extras.fastStarTrackerAnalysis.Source` or
              `lsst.summit.extras.fastStarTrackerAnalysis.Source`
        The sources found in the image.
    """
    exp = openFile(parentFilename)
    data = exp.image.array

    fig = plt.figure(figsize=(16, 8))
    ax = fig.subplots(1)

    plt.imshow(data, interpolation="None", origin="lower")

    sources = ensure_iterable(sources)
    patches = []
    for source in sources:
        ax.scatter(source.centroidX, source.centroidY, color="red", marker="x")  # mark the centroid
        patch = bboxToMatplotlibRectanle(source.bbox)
        patches.append(patch)

    # move the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)

    # plot the bboxes on top
    pc = PatchCollection(patches, edgecolor="r", facecolor="none")
    ax.add_collection(pc)

    plt.tight_layout()


def plotSource(source: Source) -> None:
    """Plot a single source.

    Parameters
    ----------
    source : `lsst.summit.extras.fastStarTrackerAnalysis.Source`
        The source to plot.
    """
    if source.cutout is None:
        raise RuntimeError(
            "Can only plot sources with attached cutouts. Either set attachCutouts=True "
            "in findFastStarTrackerImageSources() or try using plotSourcesOnImage() instead"
        )

    fig = plt.figure(figsize=(16, 8))
    ax = fig.subplots(1)

    plt.imshow(source.cutout, interpolation="None", origin="lower")  # plot the image
    ax.scatter(source.localCentroidX, source.localCentroidY, color="red", marker="x", s=200)  # mark centroid

    # move the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)

    plt.tight_layout()
