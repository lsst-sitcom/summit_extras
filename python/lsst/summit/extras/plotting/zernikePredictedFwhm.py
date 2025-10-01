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
    "makeZernikePredictedFWHMPlot",
]


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


def makeZernikePredictedFWHMPlot(
    table: Table,
    wavefrontData: dict,
    saveAs: str = "",
):
    """Make a focal plane plot of predicted FWHM based on Zernike coefficients.

    Left panel: Shows the measured zernikes in the corner
    and the interpolated values across the focal plane.

    Right top panel: shows the measured FWHM values
    across the focal plane.

    Right bottom panel: shows interpolated FWHM values based
    on the interpolated zernikes across the focal plane.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table containing the measured FWHM data to be plotted.
    wavefrontData : `dict`
        Dictionary containing wavefront measured and interpolated data.
    saveAs : `str`, optional
        If provided, the plot will be saved to this file.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.75], figure=fig)

    # --- Left: Zernikes grid ---
    gs_left = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gs[0])

    zernikesMeasured = wavefrontData["zksMeasured"]
    for i, zk_id in enumerate(np.arange(4, zernikesMeasured.shape[1])):
        ax = fig.add_subplot(gs_left[i])
        valsMeasured = zernikesMeasured[:, zk_id]
        valsInterp = wavefrontData["zksInterpolated"][:, zk_id]

        vmax = np.max(np.abs(np.concatenate([valsMeasured, valsInterp])))
        vmin = -vmax
        sc = ax.scatter(
            wavefrontData["rotatedPositions"][:, 0],
            -wavefrontData["rotatedPositions"][:, 1],
            c=valsInterp,
            s=25,
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(
            wavefrontData["fieldAngles"][:, 0],
            -wavefrontData["fieldAngles"][:, 1],
            c=valsMeasured,
            s=85,
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )

        circle = plt.Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
        ax.add_patch(circle)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cbar.ax.tick_params(labelsize=12)

        ax.set_aspect("equal", "box")
        ax.axis("off")
        ax.set_title(f"$Z_{{{zk_id}}}$", fontsize=15)

    # --- Right: stacked FWHM ---
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])
    ax = fig.add_subplot(gs_right[0])
    circle = plt.Circle((0, 0), 1.75, color="gray", fill=False, linestyle="--")
    ax.add_patch(circle)
    sc = ax.scatter(table["aa_x"], table["aa_y"], c=table["FWHM"], s=9)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title("Measured FWHM 500nm", fontsize=15)

    # AOS FWHM
    ax = fig.add_subplot(gs_right[1])
    circle = plt.Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")

    ax.scatter(
        wavefrontData["fieldAngles"][:, 0],
        -wavefrontData["fieldAngles"][:, 1],
        c=wavefrontData["fwhmMeasured"],
        s=50,
    )
    sc = ax.scatter(table["aa_x"], table["aa_y"], c=wavefrontData["fwhmInterpolated"], s=8)
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title("Zernike-predicted AOS FWHM", fontsize=15)

    visitId = table.meta["LSST BUTLER DATAID VISIT"]
    dayObs = visitId // 100000
    seqNum = visitId % 100000
    fig.suptitle(
        f"dayObs={dayObs} seqNum={seqNum}",
        fontsize=12,
        y=1.01,
    )

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)
