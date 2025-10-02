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

import itertools
import logging
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from astropy.time import TimeDelta
from lsst_efd_client import EfdClient
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import lsst.daf.butler as dafButler
import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.summit.utils.efdUtils import getCommands, getEfdData
from lsst.summit.utils.simonyi.mountData import getAzElRotHexDataForPeriod
from lsst.summit.utils.utils import dayObsIntToString
from lsst.utils.plotting.figures import make_figure

if TYPE_CHECKING:
    from astropy.time import Time
    from matplotlib.figure import Figure


__all__ = ["plotExposureTiming"]

READOUT_TIME = TimeDelta(2.3, format="sec")

COMMANDS_TO_QUERY = [
    # MTPtg
    "lsst.sal.MTPtg.command_azElTarget",
    "lsst.sal.MTPtg.command_disable",
    "lsst.sal.MTPtg.command_enable",
    "lsst.sal.MTPtg.command_exitControl",
    "lsst.sal.MTPtg.command_offsetAzEl",
    "lsst.sal.MTPtg.command_poriginOffset",
    "lsst.sal.MTPtg.command_raDecTarget",
    "lsst.sal.MTPtg.command_rotOffset",
    "lsst.sal.MTPtg.command_standby",
    "lsst.sal.MTPtg.command_start",
    "lsst.sal.MTPtg.command_startTracking",
    "lsst.sal.MTPtg.command_stopTracking",
    # MTMount
    "lsst.sal.MTMount.command_applySettingsSet",
    "lsst.sal.MTMount.command_closeMirrorCovers",
    "lsst.sal.MTMount.command_disable",
    "lsst.sal.MTMount.command_enable",
    "lsst.sal.MTMount.command_enableCameraCableWrapFollowing",
    "lsst.sal.MTMount.command_exitControl",
    "lsst.sal.MTMount.command_homeBothAxes",
    "lsst.sal.MTMount.command_moveToTarget",
    "lsst.sal.MTMount.command_openMirrorCovers",
    "lsst.sal.MTMount.command_park",
    "lsst.sal.MTMount.command_restoreDefaultSettings",
    "lsst.sal.MTMount.command_setLogLevel",
    "lsst.sal.MTMount.command_standby",
    "lsst.sal.MTMount.command_start",
    "lsst.sal.MTMount.command_startTracking",
    "lsst.sal.MTMount.command_stop",
    "lsst.sal.MTMount.command_stopTracking",
    # 'lsst.sal.MTMount.command_trackTarget',  # exclude the 20Hz data
    # M1M3
    "lsst.sal.MTM1M3.command_clearSlewFlag",
    "lsst.sal.MTM1M3.command_setSlewControllerSettings",
    "lsst.sal.MTM1M3.command_setSlewFlag",
    "lsst.sal.MTM1M3.logevent_slewControllerSettings"
    # MTCamera
    "lsst.sal.MTCamera.logevent_startIntegration",
    "lsst.sal.MTCamera.logevent_startLoadFilter",
    "lsst.sal.MTCamera.logevent_startReadout",
    "lsst.sal.MTCamera.logevent_startRotateCarousel",
    "lsst.sal.MTCamera.logevent_startSetFilter",
    "lsst.sal.MTCamera.logevent_startShutterClose",
    "lsst.sal.MTCamera.logevent_startShutterOpen",
    "lsst.sal.MTCamera.logevent_startUnloadFilter",
    "lsst.sal.MTCamera.logevent_endLoadFilter",
    "lsst.sal.MTCamera.logevent_endOfImageTelemetry",
    "lsst.sal.MTCamera.logevent_endReadout",
    "lsst.sal.MTCamera.logevent_endRotateCarousel",
    "lsst.sal.MTCamera.logevent_endSetFilter",
    "lsst.sal.MTCamera.logevent_endShutterClose",
    "lsst.sal.MTCamera.logevent_endShutterOpen",
    "lsst.sal.MTCamera.logevent_endUnloadFilter",
    # MTAos
    # 'lsst.sal.MTAOS.logevent_cameraHexapodCorrection',
    "lsst.sal.MTAOS.logevent_configurationApplied",
    # 'lsst.sal.MTAOS.logevent_configurationsAvailable',
    # 'lsst.sal.MTAOS.logevent_degreeOfFreedom',
    # 'lsst.sal.MTAOS.logevent_errorCode',
    "lsst.sal.MTAOS.logevent_m1m3Correction",
    "lsst.sal.MTAOS.logevent_m2Correction",
    "lsst.sal.MTAOS.logevent_m2HexapodCorrection",
    # 'lsst.sal.MTAOS.logevent_mirrorStresses',
    # 'lsst.sal.MTAOS.logevent_ofcDuration',
    "lsst.sal.MTAOS.logevent_rejectedCameraHexapodCorrection",
    "lsst.sal.MTAOS.logevent_rejectedDegreeOfFreedom",
    "lsst.sal.MTAOS.logevent_rejectedM1M3Correction",
    "lsst.sal.MTAOS.logevent_rejectedM2Correction",
    "lsst.sal.MTAOS.logevent_rejectedM2HexapodCorrection",
    # 'lsst.sal.MTAOS.logevent_summaryState',
    # 'lsst.sal.MTAOS.logevent_wavefrontError',
    # 'lsst.sal.MTAOS.logevent_wepDuration'
    # Brian says to find + add the settle event
    # MTDome
    # "lsst.sal.MTDome.azimuth"
]

HEXAPOD_TOPICS = [
    "lsst.sal.MTHexapod.ackcmd",
    # 'lsst.sal.MTHexapod.actuators',
    # 'lsst.sal.MTHexapod.application',
    "lsst.sal.MTHexapod.command_disable",
    "lsst.sal.MTHexapod.command_enable",
    # 'lsst.sal.MTHexapod.command_exitControl',
    "lsst.sal.MTHexapod.command_move",
    "lsst.sal.MTHexapod.command_offset",
    "lsst.sal.MTHexapod.command_setCompensationMode",
    # 'lsst.sal.MTHexapod.command_setLogLevel',
    "lsst.sal.MTHexapod.command_standby",
    "lsst.sal.MTHexapod.command_start",
    # 'lsst.sal.MTHexapod.electrical',
    # 'lsst.sal.MTHexapod.logevent_commandableByDDS',
    # 'lsst.sal.MTHexapod.logevent_compensatedPosition',
    "lsst.sal.MTHexapod.logevent_compensationMode",
    # 'lsst.sal.MTHexapod.logevent_compensationOffset',
    # 'lsst.sal.MTHexapod.logevent_configuration',
    "lsst.sal.MTHexapod.logevent_configurationApplied",
    # 'lsst.sal.MTHexapod.logevent_configurationsAvailable',
    # 'lsst.sal.MTHexapod.logevent_connected',
    # 'lsst.sal.MTHexapod.logevent_controllerState',
    "lsst.sal.MTHexapod.logevent_errorCode",
    # 'lsst.sal.MTHexapod.logevent_heartbeat',
    "lsst.sal.MTHexapod.logevent_inPosition",
    # 'lsst.sal.MTHexapod.logevent_interlock',
    # 'lsst.sal.MTHexapod.logevent_logLevel',
    # 'lsst.sal.MTHexapod.logevent_logMessage',
    # 'lsst.sal.MTHexapod.logevent_simulationMode',
    # 'lsst.sal.MTHexapod.logevent_softwareVersions',
    # 'lsst.sal.MTHexapod.logevent_summaryState',
    # 'lsst.sal.MTHexapod.logevent_uncompensatedPosition'
]

inPositionTopics = {
    "Hexapod": "lsst.sal.MTHexapod.logevent_inPosition",
    "M2": "lsst.sal.MTM2.logevent_m2AssemblyInPosition",
    "Azimuth": "lsst.sal.MTMount.logevent_azimuthInPosition",
    "Camera cable wrap": "lsst.sal.MTMount.logevent_cameraCableWrapInPosition",
    "Elevation": "lsst.sal.MTMount.logevent_elevationInPosition",
    "Rotator": "lsst.sal.MTRotator.logevent_inPosition",
    "Dome": "lsst.sal.MTDome.logevent_azMotion",
}


def getAxisName(topic):
    # Note the order here matters, e.g. cameraCableWrap is a substring of
    # MTMount so it should be checked first, likewise axes are special cases
    # of the MTMount so should be checked first.
    if "MTDome.logevent_azMotion" in topic:
        return "dome"

    if "MTMount.logevent_elevationInPosition" in topic:
        return "el"

    if "MTMount.logevent_azimuthInPosition" in topic:
        return "az"

    if "MTRotator.logevent_inPosition" in topic:
        return "rot"

    if any(x in topic for x in ["MTCamera", "MTRotator", "cameraCableWrap"]):
        return "camera"

    if any(x in topic for x in ["MTPtg", "MTMount", "MTM1M3", "MTM2"]):
        return "mount"

    if any(x in topic for x in ["MTAOS", "MTHexapod", "MTM1M3", "MTM2"]):
        return "aos"


def getDomeData(
    client: EfdClient, begin: Time, end: Time, prePadding: float, postPadding: float, threshold: float = 2.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get dome data and when dome is within threshold of being in position.

    Parameters
    ----------
    client : `EfdClient`
        The client object used to retrieve EFD data.
    begin : `astropy.time.Time`
        The begin time for the data retrieval.
    end : `astropy.time.Time`
        The end time for the data retrieval.
    prePadding : `float`
        The amount of time in seconds to pad before the begin time.
    postPadding : `float`
        The amount of time in seconds to pad after the end time.
    threshold : `float`, optional
        The threshold in degrees for considering the dome to be in position.

    Returns
    -------
    domeData : `pd.DataFrame`
        The dome data with actual and commanded positions.
    domeBelowThreshold : `pd.DataFrame`
        A dataframe with a single entry indicating the time when the dome
        position error drops below the threshold.
    """
    domeData = getEfdData(
        client,
        "lsst.sal.MTDome.azimuth",
        columns=["positionActual", "positionCommanded"],
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )
    # find the time when the dome position error drops below threshold
    domeData["diff"] = (domeData["positionActual"] - domeData["positionCommanded"]).abs()
    # Boolean mask where condition holds
    mask = domeData["diff"] < threshold
    # Rising edge: True when mask is True
    # but previous sample was False (or NaN at start)
    rising = mask & (~mask.shift(1, fill_value=False))
    if rising.any():
        # The last rising edge
        # (latest time where we enter the < threshold region)
        event_time = rising[rising].index.max()

    # Make a new dataframe with the domeBelowThreshold
    domeBelowThreshold = pd.DataFrame(data={"inPosition": [True]}, index=[event_time])
    return domeData, domeBelowThreshold


def plotExposureTiming(
    client: EfdClient,
    expRecords: list[dafButler.DimensionRecord],
    prePadding: float = 1,
    postPadding: float = 3,
    narrowHeightRatio: float = 0.4,
) -> Figure | None:
    """Plot the mount command timings for a set of exposures.

    This function plots the mount position data for the entire time range of
    the exposures, regardless of whether the exposures are contiguous or not.
    The exposures are shaded in the plot to indicate the time range for each
    integration its readout, and any commands issued during the time range are
    plotted as vertical lines.

    Parameters
    ----------
    client : `EfdClient`
        The client object used to retrieve EFD data.
    expRecords : `list` of `lsst.daf.butler.DimensionRecord`
        A list of exposure records to plot. The timings will be plotted from
        the start of the first exposure to the end of the last exposure,
        regardless of whether intermediate exposures are included.
    prePadding : `float`, optional
        The amount of time to pad before the start of the first exposure.
    postPadding : `float`, optional
        The amount of time to pad after the end of the last exposure.
    narrowHeightRatio : `float`, optional
        Height ratio for narrow panels (mount, dome, camera, aos) relative
        to wide ones.
    Returns
    -------
    fig : `matplotlib.figure.Figure` or `None`
        The figure containing the plot, or `None` if no data is found.
    """
    log = logging.getLogger(__name__)

    inPositionAlpha = 0.5
    commandAlpha = 0.5
    integrationColor = "grey"
    readoutColor = "blue"

    expRecords.sort(key=lambda x: (x.day_obs, x.seq_num))  # ensure we're sorted

    startSeqNum = expRecords[0].seq_num
    endSeqNum = expRecords[-1].seq_num
    dayObs = expRecords[0].day_obs
    if expRecords[-1].day_obs != dayObs:
        raise ValueError("All exposures must be from the same day_obs")
    title = f"Mount command timings for {dayObsIntToString(dayObs)} seqNums {startSeqNum} - {endSeqNum}"

    begin = expRecords[0].timespan.begin
    end = expRecords[-1].timespan.end

    mountData = getAzElRotHexDataForPeriod(client, begin, end, prePadding, postPadding)
    if mountData.empty:
        log.warning(f"No mount data found for dayObs {dayObs} seqNums {startSeqNum}-{endSeqNum}")
        return

    az = mountData.azimuthData
    el = mountData.elevationData
    rot = mountData.rotationData

    domeData, domeBelowThreshold = getDomeData(client, begin, end, prePadding, postPadding)

    # Calculate relative heights for the gridspec
    narrowHeight = narrowHeightRatio
    wideHeight = 1.0
    totalHeight = 3 * narrowHeight + 4 * wideHeight
    heights = [
        narrowHeight / totalHeight,  # mount
        wideHeight / totalHeight,  # dome
        wideHeight / totalHeight,  # azimuth
        wideHeight / totalHeight,  # elevation
        wideHeight / totalHeight,  # rotation
        narrowHeight / totalHeight,  # aos
        narrowHeight / totalHeight,  # camera
    ]

    # Create figure with adjusted gridspec
    fig = make_figure(figsize=(18, 8))
    gs = fig.add_gridspec(8, 2, height_ratios=[*heights, 0.15], width_ratios=[0.8, 0.2], hspace=0)

    # Create axes with shared x-axis
    mountAx = fig.add_subplot(gs[0, 0])
    domeAx = fig.add_subplot(gs[1, 0])
    azimuthAx = fig.add_subplot(gs[2, 0], sharex=mountAx)
    elevationAx = fig.add_subplot(gs[3, 0], sharex=mountAx)
    rotationAx = fig.add_subplot(gs[4, 0], sharex=mountAx)
    aosAx = fig.add_subplot(gs[5, 0], sharex=mountAx)
    cameraAx = fig.add_subplot(gs[6, 0], sharex=mountAx)
    bottomLegendAx = fig.add_subplot(gs[7, :])

    # Create legend axes
    mountLegendAx = fig.add_subplot(gs[0, 1])
    domeLegendAx = fig.add_subplot(gs[1, 1])
    azLegendAx = fig.add_subplot(gs[2, 1])
    elLegendAx = fig.add_subplot(gs[3, 1])
    rotLegendAx = fig.add_subplot(gs[4, 1])
    aosLegendAx = fig.add_subplot(gs[5, 1])
    cameraLegendAx = fig.add_subplot(gs[6, 1])

    axes = {
        "dome": domeAx,
        "az": azimuthAx,
        "el": elevationAx,
        "rot": rotationAx,
        "mount": mountAx,
        "camera": cameraAx,
        "aos": aosAx,
    }

    legendAxes = {
        "dome": domeLegendAx,
        "az": azLegendAx,
        "el": elLegendAx,
        "rot": rotLegendAx,
        "mount": mountLegendAx,
        "camera": cameraLegendAx,
        "aos": aosLegendAx,
        "bottom": bottomLegendAx,
    }

    # Remove frames and ticks from legend axes
    for ax in legendAxes.values():
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot telemetry
    axes["az"].plot(az["actualPosition"])
    axes["el"].plot(el["actualPosition"])
    axes["rot"].plot(rot["actualPosition"])
    axes["dome"].plot(domeData["positionActual"], label="Actual")
    axes["dome"].plot(domeData["positionCommanded"], label="Commanded")
    axes["dome"].legend(loc="lower left", frameon=False)
    # Remove y-ticks for mount, aos, and camera axes
    for ax_name in ["mount", "aos", "camera"]:
        axes[ax_name].set_yticks([])

    # Hide the last x-tick label for all axes because although they're shared
    # the last values sticks out to the right
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for ax in axes.values():
            ax.set_xticklabels(ax.get_xticklabels()[:-1])

    # Shade exposure regions and add annotations
    for record in expRecords:
        startExposing = record.timespan.begin.utc.datetime
        endExposing = record.timespan.end.utc.datetime
        readoutEnd = (record.timespan.end + READOUT_TIME).utc.datetime
        seqNum = record.seq_num

        for ax in axes.values():
            ax.axvspan(startExposing, endExposing, color=integrationColor, alpha=0.3)
            ax.axvspan(endExposing, readoutEnd, color=readoutColor, alpha=0.1)

        # Add expRecord details inside the camera section of the plot
        midpoint = startExposing + (endExposing - startExposing) / 2
        label = f"seqNum = {seqNum}\nFilter={record.physical_filter}"
        axes["camera"].annotate(
            label,
            xy=(midpoint, 0.2),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    # Create separate legend entries for each axis type
    legendEntries = {ax_name: [] for ax_name in axes.keys()}
    # Handle in-position transitions
    for label, topic in inPositionTopics.items():
        axisName = getAxisName(topic)

        inPositionTransitions = getEfdData(
            client,
            topic,
            begin=begin,
            end=end,
            prePadding=prePadding,
            postPadding=postPadding,
            warn=False,
        )
        for time, data in inPositionTransitions.iterrows():
            inPosition = data["inPosition"]
            if inPosition:
                axes[axisName].axvline(time, color="green", linestyle="--", alpha=inPositionAlpha)
            else:
                axes[axisName].axvline(time, color="red", linestyle="-", alpha=inPositionAlpha)

        legendEntries[axisName].extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="green",
                    linestyle="--",
                    label=f"{label} in position=True",
                    alpha=inPositionAlpha,
                ),
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linestyle="-",
                    label=f"{label} in position=False",
                    alpha=inPositionAlpha,
                ),
            ]
        )
        # Add special domeBelowThreshold axvline
        if label == "Dome":
            inPositionTransitions = domeBelowThreshold
            for time, data in inPositionTransitions.iterrows():
                inPosition = data["inPosition"]
                if inPosition:
                    axes[axisName].axvline(time, color="magenta", linestyle="--", alpha=inPositionAlpha)

            legendEntries[axisName].extend(
                [
                    Line2D(
                        [0],
                        [0],
                        color="magenta",
                        linestyle="-",
                        label=f"{label} below threshold=True",
                        alpha=inPositionAlpha,
                    ),
                ]
            )

    # Handle commands
    commandTimes = getCommands(
        client, COMMANDS_TO_QUERY, begin, end, prePadding, postPadding, timeFormat="python"
    )

    for topic in HEXAPOD_TOPICS:
        try:
            hexData = getEfdData(
                client,
                topic,
                begin=begin,
                end=end,
                prePadding=prePadding,
                postPadding=postPadding,
                warn=False,
            )
            commandTimes.update({time: topic for time, _ in hexData.iterrows()})
        except ValueError:
            log.warning(f"Failed to get data for {topic}")

    # Create color maps for each axis
    color_maps = {ax_name: {} for ax_name in axes.keys()}
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    color_iterators = {ax_name: itertools.cycle(colors) for ax_name in axes.keys()}

    # Group commands by axis and assign colors
    for time, command in commandTimes.items():
        axisName = getAxisName(command)
        if command not in color_maps[axisName]:
            color_maps[axisName][command] = next(color_iterators[axisName])
        color = color_maps[axisName][command]
        axes[axisName].axvline(time, linestyle="-.", alpha=commandAlpha, color=color)

        # Add to legend entries if not already there
        shortCommand = command.replace("lsst.sal.", "")
        if shortCommand not in [entry.get_label() for entry in legendEntries[axisName]]:
            entry = Line2D([0], [0], color=color, linestyle="-.", label=shortCommand, alpha=commandAlpha)
            legendEntries[axisName].append(entry)

    # Create separate legends, using 2 columns if more than 5 items
    for axisName, entries in legendEntries.items():
        if entries:
            ncols = 2 if len(entries) > 5 else 1
            legendAxes[axisName].legend(
                handles=entries, loc="center left", bbox_to_anchor=(-0.5, 0.5), ncol=ncols
            )

    # Create bottom legend for shading explanation
    shadingLegendHandles = [
        Patch(facecolor=integrationColor, alpha=0.3, label="Shutter open period"),
        Patch(facecolor=readoutColor, alpha=0.1, label="Readout period"),
    ]
    bottomLegendAx.legend(handles=shadingLegendHandles, loc="center", bbox_to_anchor=(0.4, 0.5), ncol=2)

    # Set labels with horizontal orientation
    for axisName, ax in axes.items():
        ax.set_ylabel(
            (
                f"{axisName.title() if axisName != 'aos' else 'AOS'} commands"
                if axisName in ["mount", "camera", "aos"]
                else f"{axisName.title()} (deg)"
            ),
            rotation=0,
            ha="right",
            va="center",
        )

    axes["rot"].set_xlabel("Time (UTC)")

    # Add title centered on main plot area only
    axes["mount"].set_title(title)

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # example usage
    import lsst.summit.utils.butlerUtils as butlerUtils  # noqa: F811
    from lsst.summit.extras.slewTimingSimonyi import plotExposureTiming  # noqa: F811
    from lsst.summit.utils.efdUtils import makeEfdClient

    client = makeEfdClient()
    butler = butlerUtils.makeDefaultButler("LSSTComCam")
    dayObs = 20241116
    where = f"exposure.day_obs={dayObs} AND instrument='LSSTComCam'"
    records = list(butler.registry.queryDimensionRecords("exposure", where=where))
    records = sorted(records, key=lambda x: (x.day_obs, x.seq_num))
    print(f"Found {len(records)} records from {len(set(r.day_obs for r in records))} days")
    if len(set(r.day_obs for r in records)) == 1:
        rd = {r.seq_num: r for r in records if r.seq_num >= 1}
        print(f"{len(rd)} items in the dict")

    toPlot = [records[98], records[99]]

    az = plotExposureTiming(client, toPlot)
