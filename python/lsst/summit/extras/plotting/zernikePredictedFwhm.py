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
    "makeDofPredictedFWHMPlot",
]


from itertools import zip_longest
from textwrap import fill
from typing import TYPE_CHECKING, Any

import matplotlib.gridspec as gridspec
import numpy as np
from astropy.table import Table
from matplotlib import colormaps
from matplotlib.patches import Circle

from lsst.utils.plotting.figures import make_figure

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


def formatGroup(
    name: str,
    idx: list[int],
    dofs: np.ndarray,
    ncols: int = 1,
) -> list[str]:
    """Format a group of DOFs into a list of strings for display.

    Parameters
    ----------
    name : `str`
        The name of the group.
    idx : `list[int]`
        The indices of the DOFs in this group.
    dofs : `np.ndarray`
        The array of DOF values.
    ncols : `int`, optional
        The number of columns to arrange the output in.

    Returns
    -------
    formatted : `list[str]`
        A list of formatted strings representing the DOFs in the group.
    """
    labels = ["M2 dz", "M2 dx", "M2 dy", "M2 rx", "M2 ry", "Cam dz", "Cam dx", "Cam dy", "Cam rx", "Cam ry"]
    labels += [f"B{i}" for i in range(1, 21)]
    labels += [f"B{i}" for i in range(1, 21)]

    # Start defining the formatted lines
    lines = [f"• {name}"]
    lines.append("\u200a")  # add some small spacing
    if "bending" in name:  # arrange in multiple columns for bending
        chunkSize = int(np.ceil(len(idx) / ncols))
        chunks = [idx[i : i + chunkSize] for i in range(0, len(idx), chunkSize)]
        for row in zip_longest(*chunks):
            rowStrs = []
            for i in row:
                if i is not None:
                    rowStrs.append(f"{labels[i]:<3}: {dofs[i]:7.3f}")
            lines.append("   " + "   ".join(rowStrs))
    else:  # simple vertical listing for tilts/decenters
        for i in idx:
            lines.append(f"   {labels[i]:<8}: {dofs[i]:7.3f}")
    # No final spacing for the two bottom groups
    if "M1M3" in name.upper() or "decenterings" in name.lower():
        lines.append(" ")
    return lines


def makeDofPredictedFWHMPlot(
    table: Table,
    wavefrontData: dict[str, Any],
    donutBlur: float,
    dofState: np.ndarray,
    nollIndices: np.ndarray,
    saveAs: str = "",
    zMin: int = 4,
    vmaxEllipticities: float = 0.2,
    fwhmRange: float = 0.2,
):
    """Make a focal plane plot of predicted FWHM based on estimated DOFs.

    Top center: Shows the measured zernikes and predicted zernikes from degrees
    of freedom at the four corners.

    Top right: panel with all the predicted degreed of freedom across the focal
    plane.

    Left bottom panel: shows interpolated zernike values based on the predicted
    degree of freedom state. It also shows measured values at the corners.

    Right bottom panel: shows the predicted AOS FWHM values across the focal
    plane based on the degrees of freedom, the predicted AOS FWHM + donutBlur,
    the measured FWHM, and the subtraction of measured - AOS FWHM - donut blur,
    in quadrature, of course.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table containing the measured FWHM data to be plotted.
    wavefrontData : `dict`
        Dictionary containing wavefront measured and interpolated data.
    donutBlur : `float`
        The donut blur value to be added in quadrature to the AOS FWHM.
    dofState : `np.ndarray`
        The state of the degrees of freedom predicted.
    nollIndices : `list[int]`
        List of Noll indices that were used in the wavefront sensing.
    saveAs : `str`, optional
        If provided, the plot will be saved to this file.
    zMin : `int`, optional
        The minimum Noll index used in the wavefront sensing.
    vmaxEllipticities : `float`, optional
        The maximum value for ellipticity color scaling.
    fwhmRange : `float`, optional
        The range of FWHM values for color scaling.
    """
    fig = make_figure(figsize=(40, 25))

    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 5.5], figure=fig, wspace=0.075)

    # --- Left: Zernikes grid ---
    gsLeft = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        height_ratios=[1, 3.5],
        subplot_spec=gs[0],
        hspace=0.15,
    )

    gsTopSplit = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gsLeft[0], width_ratios=[1.2, 2.0], wspace=0.05  # adjust ratio left vs right
    )

    # ----- Summary plot text panel -----
    # -----------------------------------
    axText = fig.add_subplot(gsTopSplit[0])
    axText.axis("off")

    visitId = table.meta["LSST BUTLER DATAID VISIT"]
    dayObs = visitId // 100000
    seqNum = visitId % 100000

    axText.text(
        0.43,
        1.05,
        r"$\bf{AOS\ Corner\ Predicted\ FWHM}$",
        transform=axText.transAxes,
        fontsize=19,
        va="top",
        ha="center",
    )

    bodyStr = (
        rf"day_obs = $\bf{{{dayObs}}}$"
        "\n"
        rf"seq_num = $\bf{{{seqNum}}}$"
        "\n\n"
        "Degrees of freedom: 0-9,10-16,30-34\n"
        "Number of v-modes: 12\n"
        f"Zernikes: {nollIndices.tolist()}\n\n"
        "Plots:\n"
        "(i) Zernikes predicted at corners\n"
        "(ii) Zernikes predicted across FOV\n"
        "(iii) DOFs predicted\n"
        "(iv) Predicted FWHM across FOV"
    )
    wrapped = "\n".join([fill(line, width=36) for line in bodyStr.split("\n")])
    axText.text(
        0.02,
        0.88,
        wrapped,
        transform=axText.transAxes,
        fontsize=15,
        family="monospace",
        va="top",
        ha="left",
        multialignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=0.5),  # outline
    )

    # ----- Zernike comparison at corners -----
    # -----------------------------------------
    gsTop = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gsTopSplit[1],
        wspace=0.05,
        hspace=0.1,
    )

    axes = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:
                axes[i, j] = fig.add_subplot(gsTop[i, j])
            else:
                axes[i, j] = fig.add_subplot(gsTop[i, j], sharex=axes[0, 0], sharey=axes[0, 0])

    zernikesMeasured = wavefrontData["zksMeasured"]
    zernikesEstimated = wavefrontData["zksEstimated"][:, : zernikesMeasured.shape[1]]
    zkIds = np.arange(zMin, zernikesMeasured.shape[1])
    x = np.arange(len(zkIds))
    barWidth = 0.35

    bwrMap: Colormap = colormaps["bwr"]

    for sensor in range(zernikesMeasured.shape[0]):
        ax = axes.flat[sensor]
        ax.bar(
            x - barWidth / 2, zernikesMeasured[sensor, zMin:], barWidth, label="Measured", color=bwrMap(0.0)
        )
        ax.bar(
            x + barWidth / 2,
            zernikesEstimated[sensor, zMin:],
            barWidth,
            label="Constrained",
            color=bwrMap(1.0),
        )

        ax.text(
            0.98,
            0.93,
            f"{wavefrontData['detector'][sensor]}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

        # xticks only at Z5, Z10, Z15, Z20, Z25
        tickIds = [5, 10, 15, 20, 25]
        tickPos = [i - zMin for i in tickIds if i in zkIds]
        tickIdsPresent = [i for i in tickIds if i in zkIds]
        ax.set_xticks(tickPos)
        if sensor >= 2:
            ax.set_xlabel("Noll index", fontsize=11)
            ax.set_xticklabels(tickIdsPresent)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if sensor % 2 == 0:  # left column
            ax.set_ylabel("(um)", fontsize=11)
        else:  # right column
            ax.tick_params(axis="y", labelleft=False)

        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

        for z in np.arange(zMin, zernikesMeasured.shape[1]):
            ax.axvline(z - zMin, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

        if sensor == 1:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.58, 1.03),  # centered above this axis
                borderaxespad=0,
                fontsize=15,
                frameon=False,
                ncol=2,  # two columns (one row)
            )

    # ----- Zernike grid across FOV -----
    # -----------------------------------
    gsBottom = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gsLeft[1], hspace=0.15, wspace=0.05)
    for i, zkId in enumerate(np.arange(zMin, zernikesMeasured.shape[1])):
        ax = fig.add_subplot(gsBottom[i])
        valsMeasured = zernikesMeasured[:, zkId]
        valsInterp = wavefrontData["zksInterpolated"][:, zkId]

        vmax = np.nanmax(np.abs(np.concatenate([valsMeasured, valsInterp])))
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
            s=95,
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )

        circle = Circle((0, 0), 1.75, color="gray", fill=False, linestyle="--")
        ax.add_patch(circle)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cbar.ax.tick_params(labelsize=12)

        ax.set_aspect("equal", "box")
        ax.axis("off")
        ax.set_title(f"$Z_{{{zkId}}}$", fontsize=15)

        if i == 0:  # only add once at top-left panel
            handles = [
                ax.scatter([], [], s=95, facecolors="none", edgecolors="gray", label="Measured"),
                ax.scatter([], [], s=25, facecolors="none", edgecolors="gray", label="Predicted"),
            ]
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.8, 1.36),  # move above the subplot
                fontsize=15,
                frameon=False,
                ncol=2,  # two columns = one row
                handletextpad=1.5,
            )

    # --- Right: stacked FWHM + DOF ---
    gsRight = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 6], subplot_spec=gs[1])

    # ----- Predicted DOFs -----
    # --------------------------
    axText = fig.add_subplot(gsRight[0])
    axText.axis("off")

    groups: dict[str, list[int]] = {
        "Decenterings (M2 + Cam)": [0, 1, 2, 5, 6, 7],  # 0-9
        "Tilts (M2 + Cam)": [3, 4, 8, 9],  # 10-16
        "M1M3 bending modes": list(np.arange(10, 30)),  # 17-29 (13 modes)
        "M2 bending modes": list(np.arange(30, 50)),  # 30-49 (20 modes)
    }

    # Build left/right columns
    leftCol = formatGroup("Decenterings (um)", groups["Decenterings (M2 + Cam)"], dofState)
    leftCol += formatGroup("Tilts (deg)", groups["Tilts (M2 + Cam)"], dofState)

    rightCol = formatGroup("M1M3 bending modes (um)", groups["M1M3 bending modes"], dofState, ncols=4)
    rightCol += formatGroup("M2 bending modes (um)", groups["M2 bending modes"], dofState, ncols=4)

    # Merge into side-by-side lines
    mergedLines = []
    for left, right in zip_longest(leftCol, rightCol, fillvalue=""):
        mergedLines.append(f"{left:<25}   {right}")

    textBlock = "\n".join(mergedLines)

    # Draw textbox
    axText.text(
        0.055,
        1.07,
        "Inferred DOFs",
        transform=axText.transAxes,
        fontsize=15,
        va="top",
        ha="center",
    )
    axText.text(
        0.015,
        0.9,
        textBlock,
        transform=axText.transAxes,
        fontsize=13,
        family="monospace",
        va="top",
        ha="left",
        multialignment="left",
        bbox=dict(boxstyle="round,pad=0.9", facecolor="white", edgecolor="black", linewidth=0.5),
    )

    sqrtFwhm9505 = np.sqrt(np.percentile(table["FWHM"], 95) ** 2 - np.percentile(table["FWHM"], 5) ** 2)
    fwhmMetric = np.nanmedian((table["FWHM"] ** 2 - wavefrontData["fwhmInterpolated"] ** 2 - donutBlur**2))
    bodyStr = (
        f"FWHM p25 = {np.percentile(table['FWHM'], 25):.2f} arcsec\n"
        f"FWHM p50 = {np.percentile(table['FWHM'], 50):.2f} arcsec\n"
        f"FWHM p75 = {np.percentile(table['FWHM'], 75):.2f} arcsec\n\n"
        f"e1 p50 = {np.percentile(np.abs(table['e1']), 50):.3f}\n"
        f"e2 p50 = {np.percentile(np.abs(table['e2']), 50):.3f}\n\n"
        f"Donut blur = {donutBlur:.2f} arcsec\n"
        f"Median AOS FWHM = {np.median(wavefrontData['fwhmMeasured']):.2f} arcsec\n\n"
        f"sqrt(fwhm_95 - fwhm_05) = {sqrtFwhm9505:.2f} arcsec\n\n"
        f"⟨FWHM^2_meas - FWHM^2_AOS - blur^2⟩ = {fwhmMetric:.2f} arcsec^2"
    )
    wrapped = "\n".join([fill(line, width=40) for line in bodyStr.split("\n")])
    axText.text(
        0.72,
        1.07,
        "Useful Metrics",
        transform=axText.transAxes,
        fontsize=15,
        va="top",
        ha="center",
    )
    axText.text(
        0.68,
        0.89,
        wrapped,
        transform=axText.transAxes,
        fontsize=15,
        family="monospace",
        va="top",
        ha="left",
        multialignment="left",
        bbox=dict(boxstyle="round,pad=0.9", facecolor="white", edgecolor="black", linewidth=0.5),  # outline
    )

    # ----- Grid of predicted FWHMs -----
    # -----------------------------------
    gsRightBottom = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gsRight[1], wspace=0.05, hspace=0.05)
    # AOS FWHM + Donut blur
    ax = fig.add_subplot(gsRightBottom[0])
    fwhmWithAtm = np.sqrt(wavefrontData["fwhmInterpolated"] ** 2 + donutBlur**2)
    cornersFwhmWithAtm = np.sqrt(wavefrontData["fwhmMeasured"] ** 2 + donutBlur**2)

    vals = np.concatenate(
        [
            fwhmWithAtm - np.median(fwhmWithAtm),
            cornersFwhmWithAtm - np.median(cornersFwhmWithAtm),
            table["FWHM"] - np.median(table["FWHM"]),
        ]
    )
    vmin, vmax = np.percentile(vals, [5, 95])

    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=fwhmWithAtm,
        s=9,
        vmin=vmin + np.median(fwhmWithAtm),
        vmax=vmax + np.median(fwhmWithAtm),
    )
    ax.scatter(
        wavefrontData["fieldAngles"][:, 0],
        -wavefrontData["fieldAngles"][:, 1],
        c=cornersFwhmWithAtm,
        s=50,
        vmin=vmin + np.median(cornersFwhmWithAtm),
        vmax=vmax + np.median(cornersFwhmWithAtm),
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("(arcsec)", fontsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(r"Predicted $\sqrt{ \mathrm{FWHM}_{\mathrm{AOS}}^2 + \mathrm{donut\_blur}^2 }$", fontsize=15)

    # Measured FWHM
    ax = fig.add_subplot(gsRightBottom[1])
    ax.text(
        0.5,
        1.12,
        rf"dayobs = $\bf{{{dayObs}}}$,   seq_num = $\bf{{{seqNum}}}$",
        transform=ax.transAxes,
        fontsize=14,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.4),
    )

    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=table["FWHM"],
        s=9,
        vmin=vmin + np.median(table["FWHM"]),
        vmax=vmax + np.median(table["FWHM"]),
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("(arcsec)", fontsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(r"Measured $\mathrm{FWHM}$", fontsize=15)

    # Measured - AOS FWHM + Donut blur
    ax = fig.add_subplot(gsRightBottom[2])
    vals = table["FWHM"] ** 2 - fwhmWithAtm**2
    vmax = np.nanmax(np.abs(vals))
    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=vals,
        s=9,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("(arcsec^2)", fontsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(
        r"$\mathrm{FWHM}_{\mathrm{measured}}^2 - \mathrm{FWHM}_{\mathrm{AOS}}^2 - \mathrm{donut\_blur}^2$",  # noqa: E501
        fontsize=15,
    )

    # e1 ellipticities
    # predicted e1
    ax = fig.add_subplot(gsRightBottom[3])
    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=wavefrontData["e1Interpolated"],
        cmap="seismic",
        s=9,
        vmin=-vmaxEllipticities,
        vmax=vmaxEllipticities,
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(r"Predicted $e_1$", fontsize=15)

    # Measured e1
    ax = fig.add_subplot(gsRightBottom[4])
    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=table["e1"],
        cmap="seismic",
        s=9,
        vmin=-vmaxEllipticities,
        vmax=vmaxEllipticities,
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(r"Measured $e_1$", fontsize=15)

    # Measured e1 - predicted e1
    ax = fig.add_subplot(gsRightBottom[5])
    vals = table["e1"] - wavefrontData["e1Interpolated"]
    vmax = np.nanmax(np.abs(vals))
    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=vals,
        s=9,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(
        r"Measured $e_1$ - Predicted $e_1$",
        fontsize=15,
    )

    # e2 ellipticities
    # predicted e2
    ax = fig.add_subplot(gsRightBottom[6])
    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=wavefrontData["e2Interpolated"],
        cmap="seismic",
        s=9,
        vmin=-vmaxEllipticities,
        vmax=vmaxEllipticities,
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(r"Predicted $e_2$", fontsize=15)

    # Measured e2
    ax = fig.add_subplot(gsRightBottom[7])
    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=table["e2"],
        cmap="seismic",
        s=9,
        vmin=-vmaxEllipticities,
        vmax=vmaxEllipticities,
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(r"Measured $e_2$", fontsize=15)

    # Measured e2 - predicted e2
    ax = fig.add_subplot(gsRightBottom[8])
    vals = table["e2"] - wavefrontData["e2Interpolated"]
    vmax = np.nanmax(np.abs(vals))
    sc = ax.scatter(
        table["aa_x"],
        table["aa_y"],
        c=vals,
        s=9,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(
        r"Measured $e_2$ - Predicted $e_2$",
        fontsize=15,
    )

    if saveAs:
        fig.savefig(saveAs, bbox_inches="tight", pad_inches=0)


def makeZernikePredictedFWHMPlot(
    table: Table,
    wavefrontData: dict[str, Any],
    saveAs: str = "",
):
    """Make a focal plane plot of predicted FWHM based on Zernike coefficients.

    Left panel: Shows the measured zernikes in the corner and the interpolated
    values across the focal plane.

    Right top panel: shows the measured FWHM values across the focal plane.

    Right bottom panel: shows interpolated FWHM values based on the
    interpolated zernikes across the focal plane.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table containing the measured FWHM data to be plotted.
    wavefrontData : `dict`
        Dictionary containing wavefront measured and interpolated data.
    saveAs : `str`, optional
        If provided, the plot will be saved to this file.
    """
    fig = make_figure(figsize=(20, 12))

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.75], figure=fig)

    # --- Left: Zernikes grid ---
    gsLeft = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gs[0])

    zernikesMeasured = wavefrontData["zksMeasured"]
    for i, zkId in enumerate(np.arange(4, zernikesMeasured.shape[1])):
        ax = fig.add_subplot(gsLeft[i])
        valsMeasured = zernikesMeasured[:, zkId]
        valsInterp = wavefrontData["zksInterpolated"][:, zkId]

        vmax = np.nanmax(np.abs(np.concatenate([valsMeasured, valsInterp])))
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

        circle = Circle((0, 0), 1.75, color="gray", fill=False, linestyle="--")
        ax.add_patch(circle)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cbar.ax.tick_params(labelsize=12)

        ax.set_aspect("equal", "box")
        ax.axis("off")
        ax.set_title(f"$Z_{{{zkId}}}$", fontsize=15)

    # --- Right: stacked FWHM ---
    gsRight = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])
    ax = fig.add_subplot(gsRight[0])
    sc = ax.scatter(table["aa_x"], table["aa_y"], c=table["FWHM"], s=9)
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=14)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title("Measured FWHM", fontsize=15)

    # AOS FWHM
    ax = fig.add_subplot(gsRight[1])
    ax.scatter(
        wavefrontData["fieldAngles"][:, 0],
        -wavefrontData["fieldAngles"][:, 1],
        c=wavefrontData["fwhmMeasured"],
        s=50,
    )
    sc = ax.scatter(table["aa_x"], table["aa_y"], c=wavefrontData["fwhmInterpolated"], s=8)
    circle = Circle((0, 0), 1.75, color="red", fill=False, linestyle="--")
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
        fig.savefig(saveAs, bbox_inches="tight", pad_inches=0)
