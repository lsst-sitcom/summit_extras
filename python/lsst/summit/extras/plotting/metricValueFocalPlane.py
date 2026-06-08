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

__all__ = [
    "getMetricValues",
    "makeFocalPlanePlot",
]


import copy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import gaussian_sigma_to_fwhm
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon, Rectangle

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, Camera
from lsst.daf.butler import DatasetNotFoundError
from lsst.obs.lsst import LsstCam

if TYPE_CHECKING:
    from logging import Logger

    from lsst.afw.table import ExposureCatalog
    from lsst.daf.butler import Butler


def getMetricValues(
    visitSummary: ExposureCatalog,
    metricName: str,
    log: Logger | None = None,
    butler: Butler | None = None,
) -> dict[int, float]:
    """Extract or derive per-detector metric values from a visit summary
    table.

    Note that if the metricValue involves the pixelScale, if the value in the
    visitSummary table is not finite, this likely means that astrometry failed.
    If asking directly for the pixelScale, having a NaN is valid. However, if
    the requested metric is the PSF FWHM, which uses the pixelScale to convert
    from pixles to arcsec, a rough value for the pixelScale is preferrable to
    a NaN, so in such cases, attempt to obtain the camera model based wcs
    attached to the raw image (in which case a butler much be provided in
    order to locate the raw image).

    Parameters
    ----------
    visitSummary : `lsst.afw.table.ExposureCatalog`
        The visit summary table containing the metric values (or those
        required to derive the metric).
    metricName : `str`
        The name of the metric to plot. If this name is not equal to a column
        present in the ``visitSummary`` table, it must be special cased to
        provide directives on how to compute the metric from existing column
        values (i.e. it is a derived metric).
    log : `logging.Logger` or `None`, optional
    butler : `lsst.daf.butler.Butler` or `None`, optional
        Butler with access to the raw data. Only used to compute a fallback
        pixelScale from the WCS attached to the raw image if the pixelScale
        value in the ``visitSummary`` table is non-finite (noting that the
        pixelScale is used to convert from the psfSigma in pixels to a FWHM
        in arcsec).

    Returns
    -------
    metricValues : `dict` [`int`, `float`]
        A dictionary mapping detector IDs to requested metric values.  There
        are two special cased values for ``metricName``, fwhm and momentsScore,
        which are derived from other columns in the ``visitSummary`` table.
        Otherwise, ``metricName`` is assumed to be the name of an existing
        column in the ``visitSummary`` table. Detectors missing from the
        ``visitSummary`` table are omitted.

    Raises
    ------
    RuntimeError
        Raised if ``metricName`` is not in ``visitSummary`` table and is not
        special cased as a metric derived from existing columns.
    """
    camera = LsstCam().getCamera()
    visit = visitSummary[0]["visit"]

    metricValues: dict[int, float] = {}
    for row in visitSummary:
        metricValue = float("nan")
        detectorId = row["id"]
        if metricName not in ["fwhm", "momentsScore"]:
            if metricName not in visitSummary.schema.getNames():
                raise RuntimeError(
                    f"Metric name {metricName} is not in visitSummary table and is "
                    "not special cased as a metric derived from existing columns."
                )
            metricValue = row[metricName]
        else:
            if metricName == "fwhm":
                pixelScale = row["pixelScale"]
                if ~np.isfinite(pixelScale) and butler is not None:
                    # Astrometry failed so the pixelScale is NaN. Try to get
                    # the camera model based value from the raw image.
                    try:
                        wcs = butler.get("raw.wcs", exposure=visit, detector=detectorId)
                        det = camera[detectorId]
                        pixelScale = wcs.getPixelScale(det.getBBox().getCenter()).asArcseconds()
                    except DatasetNotFoundError as e:
                        if log is not None:
                            log.warning(
                                "Failed to obtain the WCS from the raw exposure for detectorId "
                                "%d: %s. Pixel scale will remain non-finite, so PSF FWHM will be "
                                "reported as NaN.",
                                detectorId,
                                e,
                            )
                    if log is not None:
                        log.warning(
                            "Non-finite pixelScale in visitSummary table for detector: %d. "
                            "This likely indicates that the astrometry failed. Using the "
                            "value from the camera model based WCS attached to the raw "
                            "image: %.4f (arcsec/pixel).",
                            detectorId,
                            pixelScale,
                        )
                psfSigma = row["psfSigma"]
                # Convert Gaussian sigma in pixels to FWHM in arcsec:
                # FWHM = sigma * 2*sqrt(2*ln(2)), plate scale = 0.2"/pixel.
                metricValue = gaussian_sigma_to_fwhm * psfSigma * pixelScale
            if metricName == "momentsScore":
                # Compute the moments score from the higher order moments
                metricValue = 3.0 * (row["starComa1Median"] ** 2 + row["starComa2Median"] ** 2) + (
                    row["starTrefoil1Median"] ** 2 + row["starTrefoil2Median"] ** 2
                )
        metricValues[detectorId] = float(metricValue)

    return metricValues


def makeFocalPlanePlot(
    fig: plt.Figure,
    ax: plt.Axes,
    metricValues: dict[int, float],
    camera: Camera,
    doMilli: bool = False,
    vMin: float | None = None,
    vMax: float | None = None,
    doUnderColor: bool = False,
    doOverColor: bool = True,
    underColor: str = "darkmagenta",
    overColor: str = "red",
    metricLabel: str = "",
    baseFontSize: int = 4,
    coordPlane: str = "Focal Plane",
    title: str = "",
    saveAs: str = "",
) -> plt.Figure:
    """Plot a per-detector map of the metricValues across the focal plane.

    Each detector is drawn as its projected polygon in Focal Plane
    coordinates (in mm) or Feild Angle coordinates (in deg) and colored
    by its metric value; the mean, median, and standard deviation of
    the metric values across detectors are annotated in the top-right
    corner of the axes. If ``vMin``/``vMax`` are not supplied, the
    full range of the supplied metric values is used.

    If ``vMax`` is not `None` but all metric values would exceded this
    limit, switch to a Reds colormap (red to indicate "bad") and increase
    the ``vMin``/``vMax`` values.  This is to provide more information
    than just the uniform overColor for each detector.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    ax : `matplotlib.axes.Axes`
        The axes to plot on.
    metricValues : `dict` [`int`, `float`]
        Dictionary mapping detector IDs to metric values.
    camera : `lsst.afw.cameraGeom.Camera`
        The camera geometry object used to look up detector polygons.
    doMilli : `bool`
        Whether to multiply the metric value by 1000 (to convert to
        milli-unit).
    vMin : `float`, optional
        Minimum value for the color map. Defaults to
        ``nanmin(metricValues)``.
    vMax : `float`, optional
        Maximum value for the color map. Defaults to
        ``nanmax(metricValues)``.
    doUnderColor : `bool`, optional
        If `True`, set a specific color as defined in ``underColor``
        (defaults to red) for the underflow of the colorbar as set
        by ``vMin``.
    doOverColor : `bool`, optional
        If `True`, set a specific color as defined in ``overColor``
        (defaults to darkmagenta) for the underflow of the colorbar as set
        by ``vMax``.
    underColor : `str`, optional
        The color to use for the underflow, i.e. metric values lower than
        ``vMin`` if it is not `None`.
    overColor : `str`, optional
        The color to use for the overflow, i.e. metric values higher than
        ``vMax`` if it is not `None`.
    metricLabel : `str`, optional
        The string to use for the metric label.
    baseFontSize : `int`, optional
        The base font size.  The font size of all text will be scaled
        relative to this one.
    coordPlane : `str`, optional
        The coordiate plane projection to plot. Can be either
        "Focal Plane" (the default) or "Field Angle".
    title : `str`, optional
        Suptitle for the plot. If empty, no title is set.
    saveAs : `str`, optional
        If provided, save the figure to this file path.

    Raises
    ------
    RuntimeError
        Raised if an unknown coordinate plane is specified (must be
        one of "Focal Plane" or "Field Angle".

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The resulting figure.
    """
    if coordPlane not in ["Focal Plane", "Field Angle"]:
        raise RuntimeError(
            f'Unknown coordPlane: {coordPlane}.  Must be one of "Focal Plane" or "Field Angle".'
        )

    if doMilli:  # Multiply by 1e3 for a pseudo "milli" metric.
        metricValues = {key: value * 1.0e3 for key, value in metricValues.items()}

    metricMin = np.nanmin(list(metricValues.values()))
    metricMax = np.nanmax(list(metricValues.values()))

    cmap = copy.copy(plt.get_cmap("viridis"))
    labelKeyColor = "darkcyan"

    extendStr = None
    if vMin is None and vMax is not None:
        extendStr = "max"
        if doOverColor:
            cmap.set_over(overColor)
    if vMin is not None and vMax is None:
        extendStr = "min"
        if doUnderColor:
            cmap.set_under(underColor)
    if vMin is not None and vMax is not None:
        extendStr = "both"
        if doUnderColor and doOverColor:
            cmap.set_over(overColor)
            cmap.set_under(underColor)
        elif doUnderColor and not doOverColor:
            cmap.set_over(underColor)
        elif not doUnderColor and doOverColor:
            cmap.set_over(overColor)

    if vMax is not None:
        if metricMin > vMax:
            # The metric is fully above the "bad" range, so switch to an all
            # red colormap to get some dynamic range while still noting these
            # as bad.
            cmap = copy.copy(plt.get_cmap("Reds"))
            # Bump down the min to avoid a whiteout.
            vMin = metricMin - 0.1 * (metricMax - metricMin)
            vMax = metricMax
            labelKeyColor = "tab:red"

    cmap.set_bad("grey")  # Set the color for NaNs to grey

    # If vMin or vMax is None, use the min and max of the metric values.
    if vMin is None:
        vMin = metricMin
    if vMax is None:
        vMax = metricMax
    norm = Normalize(vmin=vMin, vmax=vMax)

    xvals, yvals = [], []
    colors, patches = [], []

    mmToDeg = 100 * 0.2 / 3600  # Roughly
    # Try to compute mmToDeg from camera transforms
    for detectorId, metricValue in metricValues.items():
        detector = camera.get(detectorId)
        try:
            mmToDeg = np.rad2deg(detector.getCenter(FIELD_ANGLE)[0]) / detector.getCenter(FOCAL_PLANE)[0]
            break
        except Exception:
            continue

    for detectorId, metricValue in metricValues.items():
        detector = camera.get(detectorId)
        if coordPlane == "Focal Plane":
            corners = [(c.getX(), c.getY()) for c in detector.getCorners(FOCAL_PLANE)]
        elif coordPlane == "Field Angle":
            corners = [(np.rad2deg(c.getX()), np.rad2deg(c.getY())) for c in detector.getCorners(FIELD_ANGLE)]
        else:
            raise RuntimeError(
                f'Unknown coordPlane: {coordPlane}.  Must be one of "Focal Plane" or "Field Angle".'
            )
        for corner in corners:
            xvals.append(corner[0])
            yvals.append(corner[1])

        colors.append(metricValue)
        patches.append(Polygon(corners, closed=True))
        if coordPlane == "Focal Plane":
            center = detector.getCenter(FOCAL_PLANE)
            x0 = center.getX()
            y0 = center.getY()
        else:
            center = detector.getCenter(FIELD_ANGLE)
            x0 = np.rad2deg(center.getX())
            y0 = np.rad2deg(center.getY())
        yDelta = np.abs(y0 - corners[0][1])

        ax.text(
            x0,
            y0 - 0.45 * yDelta,
            str(detectorId),
            ha="center",
            va="center",
            size=baseFontSize,
            fontweight="semibold",
            color="black",
        )
        ax.text(
            x0,
            y0 + 0.35 * yDelta,
            "{:.2f}".format(metricValue),
            ha="center",
            va="center",
            size=baseFontSize,
            fontweight="semibold",
            color="white",
        )

    p = PatchCollection(patches, alpha=0.8, cmap=cmap, norm=norm)
    p.set_array(colors)
    ax.add_collection(p)

    partlyVigRadius = 317.0
    fullyVigRadius = 350.0
    partlyVigStr = "{:.1f} (mm)".format(partlyVigRadius)
    fullyVigStr = "{:.1f} (mm)".format(fullyVigRadius)
    if coordPlane == "Field Angle":
        partlyVigRadius *= mmToDeg
        fullyVigRadius *= mmToDeg
        partlyVigStr = "{:.2f} (deg)".format(partlyVigRadius)
        fullyVigStr = "{:.2f} (deg)".format(fullyVigRadius)

    partlyVignettedCircle = Circle(
        (0.0, 0.0),
        partlyVigRadius,
        edgecolor="black",
        alpha=0.4,
        fill=False,
        linewidth=1.1,
        linestyle=":",
        zorder=2,
        label="partly vig\n{}".format(partlyVigStr),
    )
    fullyVignettedCircle = Circle(
        (0.0, 0.0),
        fullyVigRadius,
        edgecolor="black",
        alpha=0.55,
        fill=False,
        linewidth=1.1,
        linestyle="--",
        zorder=2,
        label="fully vig\n{}".format(fullyVigStr),
    )
    ax.add_patch(partlyVignettedCircle)
    ax.add_patch(fullyVignettedCircle)

    fig.colorbar(p, ax=ax, extend=extendStr, label="{}".format(metricLabel))

    ax.add_collection(p)
    ax.set_xlim(min(xvals) - abs(0.1 * min(xvals)), max(xvals) + abs(0.1 * max(xvals)))
    ax.set_ylim(min(yvals) - abs(0.1 * min(yvals)), max(yvals) + abs(0.1 * max(yvals)))
    xMin, xMax = ax.get_xlim()
    yMin, yMax = ax.get_ylim()

    # Calculate statistics
    metricValuesList = list(metricValues.values())
    metricMean = np.nanmean(metricValuesList)
    metricMedian = np.nanmedian(metricValuesList)
    metricStd = np.nanstd(metricValuesList)

    statsText = f"Mean: {metricMean:.2f}\nMedian: {metricMedian:.2f}\nStd: {metricStd:.2f}"
    ax.text(
        0.98,
        0.98,
        statsText,
        transform=ax.transAxes,
        fontsize=baseFontSize + 2,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.text(
        0.5,
        0.995,
        f"{metricLabel}",
        transform=ax.transAxes,
        fontsize=baseFontSize + 2,
        va="top",
        ha="center",
        color="white",
        zorder=30,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="dimgrey", alpha=1.0),
    )

    # Define a square patch (lower-left corner, width, height)
    square = Rectangle(
        (0.01, 0.01),
        width=0.07,
        height=0.07,
        transform=ax.transAxes,
        facecolor=labelKeyColor,
        edgecolor="grey",
        alpha=0.8,
        zorder=0,
    )
    ax.add_patch(square)

    legendKeyText = "value"
    ax.text(
        0.045,
        0.058,
        legendKeyText,
        transform=ax.transAxes,
        fontsize=baseFontSize + 1,
        fontweight="semibold",
        va="center",
        ha="center",
        color="white",
    )
    legendKeyText = "det#"
    ax.text(
        0.045,
        0.03,
        legendKeyText,
        transform=ax.transAxes,
        fontsize=baseFontSize + 1,
        fontweight="semibold",
        va="center",
        ha="center",
        color="black",
        zorder=3,
    )

    if coordPlane == "Focal Plane":
        ax.set_xlabel("Focal Plane X [mm]", fontsize=baseFontSize + 5)
        ax.set_ylabel("Focal Plane Y [mm]", fontsize=baseFontSize + 5)
    elif coordPlane == "Field Angle":
        ax.set_xlabel("Field Angle X [deg]", fontsize=baseFontSize + 5)
        ax.set_ylabel("Field Angle Y [deg]", fontsize=baseFontSize + 5)
    ax.set_aspect("equal")

    ax.legend(
        handles=[partlyVignettedCircle, fullyVignettedCircle],
        loc="upper left",
        fontsize=baseFontSize + 1,
        handlelength=1.2,
        handleheight=0.4,
    )

    if title:
        fig.suptitle(title, fontsize=min(14, (baseFontSize + 10)))

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)
    return fig
