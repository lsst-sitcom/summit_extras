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

__all__ = [
    "addColorbarToAxes",
    "extendTable",
    "makeFocalPlanePlot",
    "makeAzElPlot",
    "makeEquatorialPlot",
    "makeFigureAndAxes",
]


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.afw.cameraGeom import FOCAL_PLANE
from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import LinearTransform


def addColorbarToAxes(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    return cbar


def extendTable(table, rot, prefix):
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


def makeFigureAndAxes():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    return fig, axes


def makeFocalPlanePlot(fig, axes, table, camera, saveAs=""):
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


def makeEquatorialPlot(fig, axes, table, camera, saveAs=""):
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


def makeAzElPlot(fig, axes, table, camera, saveAs=""):
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
