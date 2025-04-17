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
]


from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from lsst.afw.cameraGeom import FIELD_ANGLE, Camera

if TYPE_CHECKING:
    import numpy.typing as npt


def makeFocalPlaneFWHMPlot(
    fig: plt.Figure,
    ax: plt.Axes,
    fwhm_values: npt.NDArray[np.float64],
    detector_ids: npt.NDArray[np.str_],
    camera: Camera,
    saveAs: str = "",
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
    saveAs : `str`, optional
        The file path to save the figure.
    """
    norm = Normalize(vmin=min(fwhm_values), vmax=max(fwhm_values))
    cmap = plt.cm.viridis

    for i, name in enumerate(detector_ids):
        detector = camera.get(name)
        corners = detector.getCorners(FIELD_ANGLE)
        corners_deg = np.rad2deg(corners)

        x = corners_deg[:, 0]
        y = corners_deg[:, 1]

        x = np.append(x, x[0])
        y = np.append(y, y[0])

        color = cmap(norm(fwhm_values[i]))
        ax.fill(x, y, color=color, edgecolor="gray", linewidth=0.5)

        # Compute center of detector for label
        x_center = np.mean(corners_deg[:, 0])
        y_center = np.mean(corners_deg[:, 1])

        # Add FWHM text in the center
        ax.text(
            x_center, y_center, f"{fwhm_values[i]:.2f}", color="white", fontsize=10, ha="center", va="center"
        )

    # Calculate statistics
    mean_fwhm = np.nanmean(fwhm_values)
    median_fwhm = np.nanmedian(fwhm_values)
    std_fwhm = np.nanstd(fwhm_values)

    stats_text = f"Mean: {mean_fwhm:.2f}''\n" f"Median: {median_fwhm:.2f}''\n" f"Std: {std_fwhm:.2f}''"
    ax.text(
        0.98,
        0.98,
        stats_text,
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

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)
