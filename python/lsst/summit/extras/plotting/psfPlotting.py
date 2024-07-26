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
from astropy.table import vstack
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.afw.cameraGeom import FOCAL_PLANE
from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import LinearTransform, radians

if TYPE_CHECKING:
    import numpy.typing as npt
    from astropy.table import Table
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
    tables = []

    for detectorNum, icSrc in icSrcs.items():
        icSrc = icSrc.asAstropy()
        icSrc = icSrc[icSrc["calib_psf_candidate"]]
        icSrc["detector"] = detectorNum
        tables.append(icSrc)

    table = vstack(tables)
    # Add shape columns
    table["Ixx"] = table["slot_Shape_xx"] * (0.2) ** 2
    table["Ixy"] = table["slot_Shape_xy"] * (0.2) ** 2
    table["Iyy"] = table["slot_Shape_yy"] * (0.2) ** 2
    table["T"] = table["Ixx"] + table["Iyy"]
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


def extendTable(table: Table, rot: npt.NDArray[np.float_], prefix: str) -> Table:
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


def makeFigureAndAxes() -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Create a figure and axes for plotting.

    Returns
    -------
    fig : `matplotlib.figure.Figure`:
        The created figure.
    axes : `numpy.ndarray`
        The created axes.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    return fig, axes


def makeFocalPlanePlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
):
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
):
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
    maxPoints: int = 1000,
    saveAs: str = "",
):
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
    maxPoints : `int`, optional
        The maximum number of points to plot. If the number of points in the
        table is greater than this value, a random subset of points will be
        plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    table = randomRows(table, maxPoints)

    cbar = addColorbarToAxes(axes[0, 0].scatter(table["aa_x"], table["aa_y"], c=table["T"], s=5))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(
            table["aa_x"], table["aa_y"], c=table["aa_e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(
            table["aa_x"], table["aa_y"], c=table["aa_e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        table["aa_x"],
        table["aa_y"],
        table["e"] * np.cos(0.5 * np.arctan2(table["aa_e2"], table["aa_e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["aa_e2"], table["aa_e1"])),
        headlength=0,
        headaxislength=0,
        scale=1,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.05, label="0.05", labelpos="S")
    for ax in axes.ravel():
        ax.set_xlabel("Az")
        ax.set_ylabel("Alt")
        ax.set_aspect("equal")
        ax.set_xlim(-90, 90)
        ax.set_ylim(-90, 90)

    # Plot camera detector outlines
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
        for ax in axes.ravel():
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

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)
