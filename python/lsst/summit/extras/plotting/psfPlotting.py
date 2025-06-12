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


import sys
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table, vstack
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, DetectorType
from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import LinearTransform, radians
from lsst.utils.plotting.figures import make_figure

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt
    from matplotlib.colorbar import Colorbar
    from matplotlib.pyplot import Axes, Figure

    from lsst.afw.cameraGeom import Camera
    from lsst.afw.image import VisitInfo
    from lsst.afw.table import SourceCatalog

# I think this is roughly right for plotting - diameter is 5x raft but we need
# less border, and 4.5 looks about right by eye.
FULL_CAMERA_FACTOR = 4.5
QUIVER_SCALE = 5.0
MM_TO_DEG = 100 * 0.2 / 3600  # Close enough


class CapturePrintToAxis:
    """Capture print statements and write them to a matplotlib axis.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The axes to which the printed text will be added.
    x : `float`, optional
        The x-coordinate in axes coordinates where the text will be placed.
    y : `float`, optional
        The y-coordinate in axes coordinates where the text will be placed.
    **kwargs : `dict`, optional
        Additional keyword arguments to be passed to the `ax.text` method,
        such as `fontsize`, `ha`, `va`, etc.
    """

    def __init__(self, ax: Axes, x: float = 0.0, y: float = 1.0, **kwargs) -> None:
        self.ax = ax
        self.x = x
        self.y = y
        kwargs.setdefault("font", "monospace")
        self.kwargs = kwargs
        self.buf: list[str] = []

    def write(self, text: str) -> None:
        self.buf.append(text)

    def flush(self) -> None:
        pass

    def __enter__(self) -> CapturePrintToAxis:
        self.old_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        sys.stdout = self.old_stdout

        text_str = "".join(self.buf)

        self.ax.text(self.x, self.y, text_str, transform=self.ax.transAxes, **self.kwargs)


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


def addColorbarToAxes(mappable: Axes) -> Colorbar:
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

    # For az/el and equitorial, represent the field in degrees instead of mm.
    rtp = table.meta["rotTelPos"]
    srtp, crtp = np.sin(rtp), np.cos(rtp)
    aa_rot = (
        np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    )
    table = extendTable(table, aa_rot, "aa", MM_TO_DEG)
    table.meta["aa_rot"] = aa_rot

    rsp = table.meta["rotSkyPos"]
    srsp, crsp = np.sin(rsp), np.cos(rsp)
    nw_rot = np.array([[crsp, -srsp], [srsp, crsp]])
    table = extendTable(table, nw_rot, "nw", MM_TO_DEG)
    table.meta["nw_rot"] = nw_rot

    return table


def extendTable(
    table: Table,
    rot: npt.NDArray[np.float64],
    prefix: str,
    xy_factor: float = 1.0,
) -> Table:
    """Extend the given table with additional columns for the rotated shapes.

    Parameters
    ----------
    table : `astropy.table.Table`
        The input table containing the original shapes.
    rot : `numpy.ndarray`
        The rotation matrix used to rotate the shapes.
    prefix : `str`
        The prefix to be added to the column names of the rotated shapes.
    xy_factor: float, optional
        A factor to scale the x and y coordinates. Default is 1.0, which means no
        scaling.

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
    table[prefix + "_x"] = xy_factor * (rot[0, 0] * table["x"] + rot[0, 1] * table["y"])
    table[prefix + "_y"] = xy_factor * (rot[1, 0] * table["x"] + rot[1, 1] * table["y"])
    return table


def makeFigureAndAxes() -> tuple[Figure, Any]:
    """Create a figure and axes for plotting.

    Returns
    -------
    fig : `matplotlib.figure.Figure`:
        The created figure.
    axs : `numpy.ndarray`
        The created axes.
    """
    # Note, tuning params here manually.  Be careful if adjusting.
    fig = make_figure(figsize=(10, 6))

    scatter_spec = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        left=0.05,
        right=0.56,
        bottom=0.05,
        top=0.9,
        wspace=0,
        hspace=0,
        width_ratios=[1, 1.07],  # Room for colorbar on right side
    )
    hist_spec = GridSpec(
        nrows=2,
        ncols=1,
        figure=fig,
        left=0.65,
        right=0.95,
        bottom=0.05,
        top=0.9,
        wspace=0,
        hspace=0.15,
    )

    axs = np.empty((2, 3), dtype=object)
    axs[0, 0] = fig.add_subplot(scatter_spec[0, 0])
    axs[0, 1] = fig.add_subplot(scatter_spec[0, 1])
    axs[1, 0] = fig.add_subplot(scatter_spec[1, 0])
    axs[1, 1] = fig.add_subplot(scatter_spec[1, 1])
    axs[0, 2] = fig.add_subplot(hist_spec[0, 0])
    axs[1, 2] = fig.add_subplot(hist_spec[1, 0])

    for ax in axs[0, :2].ravel():
        ax.set_xticks([])
    for ax in axs[:, 1].ravel():
        ax.set_yticks([])
    for ax in axs[:, 2].ravel():
        ax.set_yticks([])

    return fig, axs


def plotData(
    fig: Figure,
    axs: np.ndarray[Axes],
    table: Table,
    camera: Camera,
    prefix: str = "",
) -> None:
    """Plot the data from the table on the provided figure and axes.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axs : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `astropy.table.Table`
        The table containing the data to be plotted.
    camera : `lsst.afw.cameraGeom.Camera`
        The camera object containing the detector information.
    prefix : `str`, optional
        The prefix to be added to the column names of the rotated shapes.
    """
    x = table[prefix + "x"]
    y = table[prefix + "y"]
    e1 = table[prefix + "e1"]
    e2 = table[prefix + "e2"]
    e = table["e"]
    fwhm = table["FWHM"]

    # Quiver plot
    Q = axs[0, 0].quiver(
        x,
        y,
        e * np.cos(0.5 * np.arctan2(e2, e1)),
        e * np.sin(0.5 * np.arctan2(e2, e1)),
        headlength=0,
        headaxislength=0,
        scale=QUIVER_SCALE,
        pivot="middle",
    )
    axs[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.2, label="0.2", labelpos="S")

    # FWHM plot
    cbar = addColorbarToAxes(axs[0, 1].scatter(x, y, c=fwhm, s=5))
    cbar.set_label("FWHM [arcsec]")

    # Ellipticity plots
    emax = np.quantile(np.abs(np.concatenate([e1, e2])), 0.98)
    axs[1, 0].scatter(x, y, c=e1, vmin=-emax, vmax=emax, cmap="bwr", s=5)
    axs[1, 0].text(0.05, 0.92, "e1", transform=axs[1, 0].transAxes, fontsize=10)

    cbar = addColorbarToAxes(axs[1, 1].scatter(x, y, c=e2, vmin=-emax, vmax=emax, cmap="bwr", s=5))
    cbar.set_label("e")
    axs[1, 1].text(0.05, 0.92, "e2", transform=axs[1, 1].transAxes, fontsize=10)

    # FWHM hist
    axs[0, 2].hist(fwhm, bins=int(np.sqrt(len(table))), color="C0")
    fwhm_percentile = np.nanpercentile(fwhm, [25, 50, 75])
    with CapturePrintToAxis(axs[0, 2], x=0.95, y=0.95, ha="right", va="top", fontsize=9, font="monospace"):
        print("FWHM [arcsec]")
        print(f"25%: {fwhm_percentile[0]:.3f}")
        print(f"50%: {fwhm_percentile[1]:.3f}")
        print(f"75%: {fwhm_percentile[2]:.3f}")
    axs[0, 2].axvline(fwhm_percentile[0], color="grey", lw=1)
    axs[0, 2].axvline(fwhm_percentile[1], color="k", lw=2)
    axs[0, 2].axvline(fwhm_percentile[2], color="grey", lw=1)

    axs[1, 2].hist(e, bins=int(np.sqrt(len(table))), color="C1")
    e_percentile = np.nanpercentile(e, [25, 50, 75])
    with CapturePrintToAxis(axs[1, 2], x=0.95, y=0.95, ha="right", va="top", fontsize=9, font="monospace"):
        print("e  ")
        print(f"25%: {e_percentile[0]:.3f}")
        print(f"50%: {e_percentile[1]:.3f}")
        print(f"75%: {e_percentile[2]:.3f}")
    axs[1, 2].axvline(e_percentile[0], color="grey", lw=1)
    axs[1, 2].axvline(e_percentile[1], color="k", lw=2)
    axs[1, 2].axvline(e_percentile[2], color="grey", lw=1)


def outlineDetectors(
    fig: Figure,
    axs: np.ndarray[Axes],
    camera: Camera,
    rot: np.ndarray[float],
    rotAngle: float,
    xy_factor: float = 1.0,
):
    """Plot the outlines of the detectors.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axs : `numpy.ndarray`
        The array of axes objects to plot on.
    camera : `lsst.afw.cameraGeom.Camera`
        The camera object containing the detector information.
    rot : `numpy.ndarray`
        The rotation matrix used to rotate the detector outlines.
    rotAngle : `float`, optional
        The rotation angle in radians to apply to the detector labels.
    xy_factor : `float`, optional
        A factor to scale the x and y coordinates. Default is 1.0, which means no
        scaling.
    """
    for det in camera:
        if det.getType() != DetectorType.SCIENCE:
            continue
        xs = []
        ys = []
        for corner in det.getCorners(FOCAL_PLANE):
            xs.append(corner.x)
            ys.append(corner.y)
        xs.append(xs[0])
        ys.append(ys[0])
        xs = np.array(xs)
        ys = np.array(ys)
        rxs = xy_factor * (rot[0, 0] * xs + rot[0, 1] * ys)
        rys = xy_factor * (rot[1, 0] * xs + rot[1, 1] * ys)
        # Place detector label
        x = min([c.x for c in det.getCorners(FOCAL_PLANE)])
        y = max([c.y for c in det.getCorners(FOCAL_PLANE)])
        rx = xy_factor * (rot[0, 0] * x + rot[0, 1] * y)
        ry = xy_factor * (rot[1, 0] * x + rot[1, 1] * y)
        for ax in axs.ravel():
            ax.plot(rxs, rys, c="k", lw=1, alpha=0.3)
            ax.text(
                rx,
                ry,
                det.getName(),
                rotation_mode="anchor",
                rotation=np.rad2deg(-rotAngle) - 90,
                horizontalalignment="left",
                verticalalignment="top",
                color="k",
                fontsize=6,
                zorder=20,
            )


def shadeRafts(
    fig: Figure,
    axs: np.ndarray[Axes],
    camera: Camera,
    rot: np.ndarray[float],
    xy_factor: float = 1.0,
):
    """Shade the rafts in the focal plane plot.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axs : `numpy.ndarray`
        The array of axes objects to plot on.
    camera : `lsst.afw.cameraGeom.Camera`
        The camera object containing the detector information.
    rot : `numpy.ndarray`
        The rotation matrix used to rotate the raft outlines.
    xy_factor : `float`, optional
        A factor to scale the x and y coordinates. Default is 1.0, which means no
        scaling.
    """
    for i in range(5):
        for j in range(5):
            if i in (0, 4) and j in (0, 4):  # No corners
                continue
            raft = f"R{i}{j}"
            detector_type = camera[raft + "_S00"].getPhysicalType()
            c0 = camera[raft + "_S00"].getCorners(FOCAL_PLANE)[0]
            c1 = camera[raft + "_S02"].getCorners(FOCAL_PLANE)[1]
            c2 = camera[raft + "_S22"].getCorners(FOCAL_PLANE)[2]
            c3 = camera[raft + "_S20"].getCorners(FOCAL_PLANE)[3]
            xs = np.array([c0.x, c1.x, c2.x, c3.x, c0.x])
            ys = np.array([c0.y, c1.y, c2.y, c3.y, c0.y])
            rxs = xy_factor * (rot[0, 0] * xs + rot[0, 1] * ys)
            rys = xy_factor * (rot[1, 0] * xs + rot[1, 1] * ys)
            for ax in axs.ravel():
                c = "#999999" if detector_type == "E2V" else "#DDDDDD"
                polygon = Polygon(
                    list(zip(rxs, rys)), closed=True, fill=True, edgecolor="none", facecolor=c, alpha=0.2
                )
                ax.add_patch(polygon)


def makeFocalPlanePlot(
    fig: Figure,
    axs: np.ndarray[Axes],
    table: Table,
    camera: Camera,
    maxPointsPerDetector: int = 5,
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
    ``axs`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axs : `numpy.ndarray`
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
    if len(table) == 0:
        return
    table = randomRowsPerDetector(table, maxPointsPerDetector)

    oneRaftOnly = camera.getName() in ["LSSTComCam", "LSSTComCamSim", "TS8"]
    plotLimit = 90 if oneRaftOnly else 90 * FULL_CAMERA_FACTOR

    plotData(fig, axs, table, camera)

    for ax in axs[:2, :2].ravel():
        ax.set_xlim(-plotLimit, plotLimit)
        ax.set_ylim(-plotLimit, plotLimit)
    for ax in axs[1, :2]:
        ax.set_xlabel("Focal Plane x [mm]")
    for ax in axs[:2, 0]:
        ax.set_ylabel("Focal Plane y [mm]")

    visitId = table.meta["LSST BUTLER DATAID VISIT"]
    dayObs = visitId // 100000
    seqNum = visitId % 100000
    fig.suptitle(
        f"dayObs={dayObs} seqNum={seqNum}",
        fontsize=12,
        y=0.95,
    )

    rot = np.eye(2)
    if oneRaftOnly:
        rotAngle = -np.pi / 2
        outlineDetectors(
            fig,
            axs[:2, :2].ravel(),
            camera,
            rot,
            rotAngle,
        )
    else:
        shadeRafts(
            fig,
            axs[:2, :2].ravel(),
            camera,
            rot,
        )

    if saveAs:
        fig.savefig(saveAs)


def makeEquatorialPlot(
    fig: Figure,
    axs: np.ndarray[Axes],
    table: Table,
    camera: Camera,
    maxPointsPerDetector: int = 5,
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
    ``axs`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axs : `numpy.ndarray`
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
    if len(table) == 0:
        return
    table = randomRowsPerDetector(table, maxPointsPerDetector)

    oneRaftOnly = camera.getName() in ["LSSTComCam", "LSSTComCamSim", "TS8"]
    plotLimit = 90 * MM_TO_DEG if oneRaftOnly else 90 * MM_TO_DEG * FULL_CAMERA_FACTOR

    plotData(fig, axs, table, camera, prefix="nw_")

    for ax in axs[:2, :2].ravel():
        ax.set_xlim(-plotLimit, plotLimit)
        ax.set_ylim(-plotLimit, plotLimit)
    for ax in axs[1, :2]:
        ax.set_xlabel(r"$\Delta$ West [deg]")
    for ax in axs[:2, 0]:
        ax.set_ylabel(r"$\Delta$ North [deg]")

    visitId = table.meta["LSST BUTLER DATAID VISIT"]
    dayObs = visitId // 100000
    seqNum = visitId % 100000
    fig.suptitle(
        f"dayObs={dayObs} seqNum={seqNum}",
        fontsize=12,
        y=0.95,
    )

    rot = table.meta["nw_rot"]
    if oneRaftOnly:
        rotAngle = -table.meta["rotSkyPos"] - np.pi / 2
        outlineDetectors(
            fig,
            axs[:2, :2].ravel(),
            camera,
            rot,
            rotAngle,
            xy_factor=MM_TO_DEG,
        )
    else:
        shadeRafts(
            fig,
            axs[:2, :2].ravel(),
            camera,
            rot,
            xy_factor=MM_TO_DEG,
        )

    if saveAs:
        fig.savefig(saveAs)


def makeAzElPlot(
    fig: Figure,
    axs: np.ndarray[Axes],
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
    ``axs`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axs : `numpy.ndarray`
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
    if len(table) == 0:
        return
    table = randomRowsPerDetector(table, maxPointsPerDetector)

    oneRaftOnly = camera.getName() in ["LSSTComCam", "LSSTComCamSim", "TS8"]
    plotLimit = 90 * MM_TO_DEG if oneRaftOnly else 90 * MM_TO_DEG * FULL_CAMERA_FACTOR

    plotData(fig, axs, table, camera, prefix="aa_")

    for ax in axs[:2, :2].ravel():
        ax.set_xlim(-plotLimit, plotLimit)
        ax.set_ylim(-plotLimit, plotLimit)
    for ax in axs[1, :2]:
        ax.set_xlabel("$\\Delta$ Azimuth [deg]")
    for ax in axs[:2, 0]:
        ax.set_ylabel("$\\Delta$ Elevation [deg]")

    visitId = table.meta["LSST BUTLER DATAID VISIT"]
    dayObs = visitId // 100000
    seqNum = visitId % 100000
    fig.suptitle(
        f"dayObs={dayObs} seqNum={seqNum}",
        fontsize=12,
        y=0.95,
    )

    rot = table.meta["aa_rot"]
    if oneRaftOnly:
        rotAngle = table.meta["rotTelPos"]
        outlineDetectors(
            fig,
            axs[:2, :2].ravel(),
            camera,
            rot,
            rotAngle,
            xy_factor=MM_TO_DEG,
        )
    else:
        shadeRafts(
            fig,
            axs[:2, :2].ravel(),
            camera,
            rot,
            xy_factor=MM_TO_DEG,
        )

    if saveAs:
        fig.savefig(saveAs)
