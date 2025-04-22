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
    "addColorbarToAxes",
    "makeTableFromSourceCatalogs",
    "makeFigureAndAxes",
    "extendTable",
    "makeFocalPlanePlot",
    "makeEquatorialPlot",
    "makeAzElPlot",
]


from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack
from astropy.table import vstack
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE
from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import LinearTransform, radians
from lsst.utils.plotting.figures import make_figure

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt
    from matplotlib.colorbar import Colorbar

    from lsst.afw.cameraGeom import Camera
    from lsst.afw.image import VisitInfo
    from lsst.afw.table import SourceCatalog


def randomRows(table: Table, maxRows: int) -> Table:
    """Select a random subset of rows from the given table.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table containing the data to be plotted.
    maxRows : `int`
        The maximum number of rows to select.

    Returns
    -------
    table : `astropy.table.Table`
        The table containing the randomly selected subset of rows.
    """
    n = len(table)
    if n > maxRows:
        rng = np.random.default_rng()
        indices = rng.choice(n, maxRows, replace=False)
        table = table[indices]
    return table


def randomRowsPerDetector(table: Table, maxRowsPerDetector: int) -> Table:
    """Select a random subset of rows for each detector from the given table.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table containing the data to be plotted.
    maxRowsPerDetector : `int`
        The maximum number of rows to select per detector.

    Returns
    -------
    table : `astropy.table.Table`
        The table containing the randomly selected subset of rows.
    """
    keep = np.full(len(table), False)
    for det in np.unique(table["detector"]):
        detrows = np.nonzero(table["detector"] == det)[0]
        if len(detrows) > maxRowsPerDetector:
            rng = np.random.default_rng()
            detrows = rng.choice(detrows, maxRowsPerDetector, replace=False)
        keep[detrows] = True
    return table[keep]


def addColorbarToAxes(mappable: plt.Axes) -> Colorbar:
    """Add a colorbar to the given axes.

    Parameters
    ----------
    mappable : `matplotlib.cm.ScalarMappable`
        The mappable object to which the colorbar will be added.

    Returns
    -------
    cbar : `matplotlib.colorbar.Colorbar`
        The colorbar object that was added to the axes.
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    return cbar


def makeTableFromSourceCatalogs(icSrcs: dict[int, SourceCatalog], visitInfo: VisitInfo) -> Table:
    """Extract the shapes from the source catalogs into an astropy table.

    The shapes of the PSF candidates are extracted from the source catalogs and
    transformed into the required coordinate systems for plotting either focal
    plane coordinates, az/el coordinates, or equatorial coordinates.

    Parameters
    ----------
    icSrcs : `dict` [`int`, `lsst.afw.table.SourceCatalog`]
        A dictionary of source catalogs, keyed by the detector numbers.
    visitInfo : `lsst.afw.image.VisitInfo`
        The visit information for a representative visit.

    Returns
    -------
    table : `astropy.table.Table`
        The table containing the data from the source catalogs.
    """
    if len(icSrcs) == 0:
        return Table()
    tables = []

    for detectorNum, icSrc in icSrcs.items():
        icSrc = icSrc.asAstropy()
        icSrc = icSrc[icSrc["calib_psf_candidate"]]
        icSrc["detector"] = detectorNum
        tables.append(icSrc)

    table = vstack(tables, metadata_conflicts="silent")

    # Add shape columns
    table["Ixx"] = table["slot_Shape_xx"] * (0.2) ** 2
    table["Ixy"] = table["slot_Shape_xy"] * (0.2) ** 2
    table["Iyy"] = table["slot_Shape_yy"] * (0.2) ** 2
    table["T"] = table["Ixx"] + table["Iyy"]
    table["FWHM"] = np.sqrt(table["T"] / 2 * np.log(256))
    table["e1"] = (table["Ixx"] - table["Iyy"]) / table["T"]
    table["e2"] = 2 * table["Ixy"] / table["T"]
    table["e"] = np.hypot(table["e1"], table["e2"])
    table["x"] = table["base_FPPosition_x"]
    table["y"] = table["base_FPPosition_y"]

    table.meta["rotTelPos"] = (
        visitInfo.boresightParAngle - visitInfo.boresightRotAngle - (np.pi / 2 * radians)
    ).asRadians()
    table.meta["rotSkyPos"] = visitInfo.boresightRotAngle.asRadians()

    rtp = table.meta["rotTelPos"]
    srtp, crtp = np.sin(rtp), np.cos(rtp)
    aaRot = np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    table = extendTable(table, aaRot, "aa")
    table.meta["aaRot"] = aaRot

    rsp = table.meta["rotSkyPos"]
    srsp, crsp = np.sin(rsp), np.cos(rsp)
    nwRot = np.array([[crsp, -srsp], [srsp, crsp]])
    table = extendTable(table, nwRot, "nw")
    table.meta["nwRot"] = nwRot

    return table


def extendTable(table: Table, rot: npt.NDArray[np.float64], prefix: str) -> Table:
    """Extend the given table with additional columns for the rotated shapes.

    Parameters
    ----------
    table : `astropy.table.Table`
        The input table containing the original shapes.
    rot : `numpy.ndarray`
        The rotation matrix used to rotate the shapes.
    prefix : `str`
        The prefix to be added to the column names of the rotated shapes.

    Returns
    -------
    table : `astropy.table.Table`
        The extended table with additional columns representing the rotated
        shapes.
    """
    transform = LinearTransform(rot)
    rot_shapes = []
    for row in table:
        shape = Quadrupole(row["Ixx"], row["Iyy"], row["Ixy"])
        rot_shape = shape.transform(transform)
        rot_shapes.append(rot_shape)
    table[prefix + "_Ixx"] = [sh.getIxx() for sh in rot_shapes]
    table[prefix + "_Iyy"] = [sh.getIyy() for sh in rot_shapes]
    table[prefix + "_Ixy"] = [sh.getIxy() for sh in rot_shapes]
    table[prefix + "_e1"] = (table[prefix + "_Ixx"] - table[prefix + "_Iyy"]) / table["T"]
    table[prefix + "_e2"] = 2 * table[prefix + "_Ixy"] / table["T"]
    table[prefix + "_x"] = rot[0, 0] * table["x"] + rot[0, 1] * table["y"]
    table[prefix + "_y"] = rot[1, 0] * table["x"] + rot[1, 1] * table["y"]
    return table


def makeFigureAndAxes() -> tuple[plt.Figure, Any]:
    """Create a figure and axes for plotting.

    Returns
    -------
    fig : `matplotlib.figure.Figure`:
        The created figure.
    axes : `numpy.ndarray`
        The created axes.
    """
    fig = make_figure(figsize=(10, 6))
    gs = GridSpec(2, 3, figure=fig)

    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = fig.add_subplot(gs[0, 2])
    axes[1, 2] = fig.add_subplot(gs[1, 2])
    return fig, axes


def makeFocalPlanePlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
) -> None:
    """Plot the PSFs in focal plane (detector) coordinates i.e. the raw shapes.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPoints : `int`, optional
        The maximum number of points to plot. If the number of points in the
        table is greater than this value, a random subset of points will be
        plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    if len(table) == 0:
        return
    table = randomRows(table, maxPoints)

    cbar = addColorbarToAxes(axes[0, 0].scatter(table["x"], table["y"], c=table["T"], s=5))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(table["x"], table["y"], c=table["e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5)
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(table["x"], table["y"], c=table["e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5)
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        table["x"],
        table["y"],
        table["e"] * np.cos(0.5 * np.arctan2(table["e2"], table["e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["e2"], table["e1"])),
        headlength=0,
        headaxislength=0,
        scale=1,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.05, label="0.05", labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("Focal Plane x [mm]")
        ax.set_ylabel("Focal Plane y [mm]")
        ax.set_aspect("equal")

    # Plot camera detector outlines
    for det in camera:
        xs = []
        ys = []
        for corner in det.getCorners(FOCAL_PLANE):
            xs.append(corner.x)
            ys.append(corner.y)
        xs.append(xs[0])
        ys.append(ys[0])
        xs = np.array(xs)
        ys = np.array(ys)
        for ax in axes.ravel():
            ax.plot(xs, ys, c="k", lw=1, alpha=0.3)

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)


def makeEquatorialPlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
) -> None:
    """Plot the PSFs on the focal plane, rotated to equatorial coordinates.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPoints : `int`, optional
        The maximum number of points to plot. If the number of points in the
        table is greater than this value, a random subset of points will be
        plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    if len(table) == 0:
        return
    table = randomRows(table, maxPoints)

    cbar = addColorbarToAxes(axes[0, 0].scatter(table["nw_x"], table["nw_y"], c=table["T"], s=5))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(
            table["nw_x"], table["nw_y"], c=table["nw_e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(
            table["nw_x"], table["nw_y"], c=table["nw_e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        table["nw_x"],
        table["nw_y"],
        table["e"] * np.cos(0.5 * np.arctan2(table["nw_e2"], table["nw_e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["nw_e2"], table["nw_e1"])),
        headlength=0,
        headaxislength=0,
        scale=1,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.05, label="0.05", labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("West")
        ax.set_ylabel("North")
        ax.set_aspect("equal")
        ax.set_xlim(-90, 90)
        ax.set_ylim(-90, 90)

    # Plot camera detector outlines
    nwRot = table.meta["nwRot"]
    for det in camera:
        xs = []
        ys = []
        for corner in det.getCorners(FOCAL_PLANE):
            xs.append(corner.x)
            ys.append(corner.y)
        xs.append(xs[0])
        ys.append(ys[0])
        xs = np.array(xs)
        ys = np.array(ys)
        rxs = nwRot[0, 0] * xs + nwRot[0, 1] * ys
        rys = nwRot[1, 0] * xs + nwRot[1, 1] * ys
        for ax in axes.ravel():
            ax.plot(rxs, rys, c="k", lw=1, alpha=0.3)

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)


def makeAzElPlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPointsPerDetector: int = 5,
    saveAs: str = "",
) -> None:
    """Plot the PSFs on the focal plane, rotated to az/el coordinates.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPointsPerDetector : `int`, optional
        The maximum number of points per detector to plot. If the number of
        points in the table is greater than this value, a random subset of
        points will be plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    mm_to_deg = 100 * 0.2 / 3600
    if len(table) == 0:
        return
    oneRaftOnly = camera.getName() in ["LSSTComCam", "LSSTComCamSim", "TS8"]
    # I think this is roughly right for plotting - diameter is 5x but we need
    # less border, and 4.5 looks about right by eye.
    fullCameraFactor = 4.5
    plotLimit = 90 * mm_to_deg if oneRaftOnly else 90 * fullCameraFactor * mm_to_deg
    quiverScale = 5

    table = randomRowsPerDetector(table, maxPointsPerDetector)

    cbar = addColorbarToAxes(
        axes[0, 1].scatter(table["aa_x"] * mm_to_deg, table["aa_y"] * mm_to_deg, c=table["FWHM"], s=5)
    )
    cbar.set_label("FWHM [arcsec]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    axes[1, 0].scatter(
        table["aa_x"] * mm_to_deg,
        table["aa_y"] * mm_to_deg,
        c=table["aa_e1"],
        vmin=-emax,
        vmax=emax,
        cmap="bwr",
        s=5,
    )
    axes[1, 0].text(0.05, 0.92, "e1", transform=axes[1, 0].transAxes, fontsize=10)

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(
            table["aa_x"] * mm_to_deg,
            table["aa_y"] * mm_to_deg,
            c=table["aa_e2"],
            vmin=-emax,
            vmax=emax,
            cmap="bwr",
            s=5,
        )
    )
    cbar.set_label("e")
    axes[1, 1].text(0.05, 0.92, "e2", transform=axes[1, 1].transAxes, fontsize=10)

    Q = axes[0, 0].quiver(
        table["aa_x"] * mm_to_deg,
        table["aa_y"] * mm_to_deg,
        table["e"] * np.cos(0.5 * np.arctan2(table["aa_e2"], table["aa_e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["aa_e2"], table["aa_e1"])),
        headlength=0,
        headaxislength=0,
        scale=quiverScale,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.2, label="0.2", labelpos="S")

    for ax in axes[:2, :2].ravel():
        ax.set_aspect("equal")
        ax.set_xlim(-plotLimit, plotLimit)
        ax.set_ylim(-plotLimit, plotLimit)
    for ax in axes[1, :2]:
        ax.set_xlabel("$\\Delta$ Azimuth [deg]")
    for ax in axes[:2, 0]:
        ax.set_ylabel("$\\Delta$ Elevation [deg]")

    # Plot camera detector outlines for single-raft cameras
    if oneRaftOnly:
        aaRot = table.meta["aaRot"]
        for det in camera:
            xs = []
            ys = []
            for corner in det.getCorners(FOCAL_PLANE):
                xs.append(corner.x)
                ys.append(corner.y)
            xs.append(xs[0])
            ys.append(ys[0])
            xs = np.array(xs)
            ys = np.array(ys)
            rxs = aaRot[0, 0] * xs + aaRot[0, 1] * ys
            rys = aaRot[1, 0] * xs + aaRot[1, 1] * ys
            # Place detector label
            x = min([c.x for c in det.getCorners(FOCAL_PLANE)])
            y = max([c.y for c in det.getCorners(FOCAL_PLANE)])
            rx = aaRot[0, 0] * x + aaRot[0, 1] * y
            ry = aaRot[1, 0] * x + aaRot[1, 1] * y
            rtp = table.meta["rotTelPos"]
            for ax in axes[:2, :2].ravel():
                ax.plot(rxs, rys, c="k", lw=1, alpha=0.3)
                ax.text(
                    rx,
                    ry,
                    det.getName(),
                    rotation_mode="anchor",
                    rotation=np.rad2deg(-rtp) - 90,
                    horizontalalignment="left",
                    verticalalignment="top",
                    color="k",
                    fontsize=6,
                    zorder=20,
                )
    else:
        # For multi-raft, shade outlines of rafts
        aaRot = table.meta["aaRot"]
        for i in range(5):
            for j in range(5):
                if i in (0, 4) and j in (0, 4):  # No corners
                    continue
                raft = f"R{i}{j}"
                detector_type = camera[raft + "_S00"].getPhysicalType()
                c0 = camera[raft + "_S00"].getCorners(FIELD_ANGLE)[0]
                c1 = camera[raft + "_S02"].getCorners(FIELD_ANGLE)[1]
                c2 = camera[raft + "_S22"].getCorners(FIELD_ANGLE)[2]
                c3 = camera[raft + "_S20"].getCorners(FIELD_ANGLE)[3]
                xs = np.rad2deg([c0.x, c1.x, c2.x, c3.x, c0.x])
                ys = np.rad2deg([c0.y, c1.y, c2.y, c3.y, c0.y])
                xs = np.array(xs)
                ys = np.array(ys)
                rxs = aaRot[0, 0] * xs + aaRot[0, 1] * ys
                rys = aaRot[1, 0] * xs + aaRot[1, 1] * ys
                for ax in axes[:2, :2].ravel():
                    c = "#999999" if detector_type == "E2V" else "#DDDDDD"
                    polygon = Polygon(
                        list(zip(rxs, rys)), closed=True, fill=True, edgecolor="none", facecolor=c, alpha=0.2
                    )
                    ax.add_patch(polygon)

    # Add histograms
    fwhm_percentile = np.nanpercentile(table["FWHM"].data, [25, 50, 75])
    e_percentile = np.nanpercentile(table["e"].data, [25, 50, 75])
    axes[0, 2].hist(table["FWHM"], bins=int(np.sqrt(len(table))), color="C0")
    axes[1, 2].hist(table["e"], bins=int(np.sqrt(len(table))), color="C1")
    text_kwargs = {
        "transform": axes[0, 2].transAxes,
        "fontsize": 10,
        "horizontalalignment": "right",
        "verticalalignment": "top",
    }
    axes[0, 2].text(0.95, 0.95, "FWHM", **text_kwargs)
    axes[0, 2].text(0.95, 0.89, "[arcsec]", **text_kwargs)
    axes[0, 2].text(0.95, 0.83, f"median: {fwhm_percentile[1]:.2f}", **text_kwargs)
    axes[0, 2].text(0.95, 0.77, f"IQR: {fwhm_percentile[2] - fwhm_percentile[0]:.2f}", **text_kwargs)
    axes[0, 2].axvline(fwhm_percentile[1], color="k", lw=2)
    axes[0, 2].axvline(fwhm_percentile[0], color="grey", lw=1)
    axes[0, 2].axvline(fwhm_percentile[2], color="grey", lw=1)

    axes[1, 2].text(
        0.95,
        0.95,
        "e",
        transform=axes[1, 2].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="top",
    )
    axes[1, 2].axvline(e_percentile[1], color="k", lw=2)
    axes[1, 2].axvline(e_percentile[0], color="grey", lw=1)
    axes[1, 2].axvline(e_percentile[2], color="grey", lw=1)
    text_kwargs["transform"] = axes[1, 2].transAxes
    axes[1, 2].text(0.95, 0.89, f"median: {e_percentile[1]:.2f}", **text_kwargs)
    axes[1, 2].text(0.95, 0.83, f"IQR: {e_percentile[2] - e_percentile[0]:.2f}", **text_kwargs)

    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    if saveAs:
        fig.savefig(saveAs)
