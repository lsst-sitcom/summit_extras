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
    "makeFocalPlaneFWHMPlot",
    "getFwhmValues",
]


from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from lsst.afw.cameraGeom import FIELD_ANGLE, Camera
from lsst.afw.table import ExposureCatalog
from lsst.obs.lsst import LsstCam

if TYPE_CHECKING:
    import numpy.typing as npt


def getFwhmValues(visitSummary: ExposureCatalog) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get FWHM values and detector IDs from a visit summary.

    Parameters
    ----------
    visitSummary : `lsst.afw.table.ExposureCatalog`
        The visit summary table containing FWHM values.

    Returns
    -------
    fwhm_values : `numpy.ndarray`
        The FWHM values from the visit summary.
    detector_ids : `numpy.ndarray`
        The detector IDs corresponding to the FWHM values.
    """
    camera = LsstCam().getCamera()
    detectors = [det.getId() for det in camera]

    fwhmValues = []
    detectorIds = []
    for detectorId in detectors:
        row = visitSummary[visitSummary["id"] == detectorId]

        if len(row) > 0:
            psfSigma = row["psfSigma"][0]
            fwhm = psfSigma * 2.355 * 0.2  # Convert to microns (0.2"/pixel)
            fwhmValues.append(fwhm)
            detectorIds.append(detectorId)

    return np.array(fwhmValues), np.array(detectorIds)


def makeFocalPlaneFWHMPlot(
    fig: plt.Figure,
    ax: plt.Axes,
    fwhmValues: npt.NDArray[np.float64],
    detectorIds: npt.NDArray[np.float64],
    camera: Camera,
    vmin: float | None = None,
    vmax: float | None = None,
    saveAs: str = "",
    title: str = "",
):
    """Plot the FWHM across the Focal Plane, from the fwhm_values
    and detector_ids. The FWHM values are plotted per detector on
    a focal plane plot, with the color indicating the FWHM value.
    The color map is normalized between the minimum and maximum
    FWHM values.

    If ``saveAs`` is provided, the figure will be saved at the
    specified file path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    fwhm_values : `numpy.ndarray`
        The FWHM values to plot.
    detector_ids : `numpy.ndarray`
        The IDs of the detectors corresponding to the FWHM values.
    camera : `list`
        The list of camera detector objects.
    vmin : `float`, optional
        The minimum value for the color map
    vmax : `float`, optional
        The maximum value for the color map
    saveAs : `str`, optional
        The file path to save the figure.
    title : `str`, optional
        The title of the plot. If not provided, no title is set.
    """
    # If vmin and vmax are None, use the min and max of the FWHM values
    if vmin is None:
        vmin = np.nanmin(fwhmValues)
    if vmax is None:
        vmax = np.nanmax(fwhmValues)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    for i, detectorId in enumerate(detectorIds):
        detector = camera.get(detectorId)
        corners = detector.getCorners(FIELD_ANGLE)
        cornersDeg = np.rad2deg(corners)

        x = cornersDeg[:, 0]
        y = cornersDeg[:, 1]

        x = np.append(x, x[0])
        y = np.append(y, y[0])

        color = cmap(norm(fwhmValues[i]))
        ax.fill(x, y, color=color, edgecolor="gray", linewidth=0.5)

        # Compute center of detector for label
        xCenter = np.mean(cornersDeg[:, 0])
        yCenter = np.mean(cornersDeg[:, 1])

        ax.text(
            xCenter, yCenter, f"{fwhmValues[i]:.2f}", color="white", fontsize=10, ha="center", va="center"
        )

    # Calculate statistics
    meanFwhm = np.nanmean(fwhmValues)
    medianFwhm = np.nanmedian(fwhmValues)
    stdFwhm = np.nanstd(fwhmValues)

    statsText = f"Mean: {meanFwhm:.2f}''\nMedian: {medianFwhm:.2f}''\nStd: {stdFwhm:.2f}''"
    ax.text(
        0.98,
        0.98,
        statsText,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("FWHM (arcsec)")

    plt.xlabel("Field Angle Y [deg]")
    plt.ylabel("Field Angle X [deg]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=18)

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)
