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

import itertools

import matplotlib
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
from lsst_efd_client import EfdClient
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import lsst.daf.butler as dafButler
import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.summit.utils.efdUtils import getCommands, getEfdData
from lsst.summit.utils.simonyi.mountData import getAzElRotDataForPeriod
from lsst.summit.utils.utils import dayObsIntToString

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
}


def plotExposureTiming(
    client: EfdClient,
    expRecords: list[dafButler.DimensionRecord],
    plotHexapod: bool = False,
    prePadding: float = 1,
    postPadding: float = 3,
) -> matplotlib.figure.Figure:
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
    plotHexapod : `bool`, optional
        Plot the ATAOS.logevent_hexapodCorrectionStarted and
        ATAOS.logevent_hexapodCorrectionCompleted transitions?
    prePadding : `float`, optional
        The amount of time to pad before the start of the first exposure.
    postPadding : `float`, optional
        The amount of time to pad after the end of the last exposure.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The figure containing the plot.
    """
    inPositionAlpha = 0.5
    commandAlpha = 0.5
    integrationColor = "grey"
    readoutColor = "blue"

    legendHandles = []

    expRecords = sorted(expRecords, key=lambda x: (x.day_obs, x.seq_num))  # ensure we're sorted

    startSeqNum = expRecords[0].seq_num
    endSeqNum = expRecords[-1].seq_num
    dayObs = expRecords[0].day_obs
    title = f"Mount command timings for {dayObsIntToString(dayObs)} seqNums {startSeqNum} - {endSeqNum}"

    begin = expRecords[0].timespan.begin
    end = expRecords[-1].timespan.end

    az, el, rot, _ = getAzElRotDataForPeriod(client, begin, end, prePadding, postPadding)

    # Create a figure with a grid specification and have axes share x
    # and have no room between each
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 1, hspace=0)
    azimuth_ax = fig.add_subplot(gs[0, 0])
    elevation_ax = fig.add_subplot(gs[1, 0], sharex=azimuth_ax)
    rotation_ax = fig.add_subplot(gs[2, 0], sharex=azimuth_ax)
    axes = {"az": azimuth_ax, "el": elevation_ax, "rot": rotation_ax}

    # plot the telemetry
    axes["az"].plot(az["actualPosition"])
    axes["el"].plot(el["actualPosition"])
    axes["rot"].plot(rot["actualPosition"])

    # shade the expRecords' regions including the readout time
    for i, record in enumerate(expRecords):
        # these need to be in UTC because matplotlib magic turns all the axis
        # timings into UTC when plotting from a dataframe.
        startExposing = record.timespan.begin.utc.datetime
        endExposing = record.timespan.end.utc.datetime

        readoutEnd = (record.timespan.end + READOUT_TIME).utc.to_value("isot")
        seqNum = record.seq_num
        for axName, ax in axes.items():
            ax.axvspan(startExposing, endExposing, color=integrationColor, alpha=0.3)
            ax.axvspan(endExposing, readoutEnd, color=readoutColor, alpha=0.1)
            if axName == "el":  # only add seqNum annotation to bottom axis
                label = f"seqNum = {seqNum}"
                midpoint = startExposing + (endExposing - startExposing) / 2
                ax.annotate(
                    label,
                    xy=(midpoint, 0.5),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    # place vertical lines at the times when axes transition in/out of position
    for label, topic in inPositionTopics.items():
        # TODO: need to iterate over colours in this loop
        inPostionTransitions = getEfdData(
            client, topic, begin=begin, end=end, prePadding=prePadding, postPadding=postPadding
        )
        for time, data in inPostionTransitions.iterrows():
            inPosition = data["inPosition"]
            if inPosition:
                axes["az"].axvline(time, color="green", linestyle="--", alpha=inPositionAlpha)
                axes["el"].axvline(time, color="green", linestyle="--", alpha=inPositionAlpha)
                axes["rot"].axvline(time, color="green", linestyle="--", alpha=inPositionAlpha)
            else:
                axes["az"].axvline(time, color="red", linestyle="-", alpha=inPositionAlpha)
                axes["el"].axvline(time, color="red", linestyle="-", alpha=inPositionAlpha)
                axes["rot"].axvline(time, color="red", linestyle="-", alpha=inPositionAlpha)

        handle = Line2D(
            [0], [0], color="green", linestyle="--", label=f"{label} in position=True", alpha=inPositionAlpha
        )
        legendHandles.append(handle)
        handle = Line2D(
            [0], [0], color="red", linestyle="-", label=f"{label} in position=False", alpha=inPositionAlpha
        )
        legendHandles.append(handle)

    # place vertical lines at the times when commands were issued
    commandTimes = getCommands(
        client, COMMANDS_TO_QUERY, begin, end, prePadding, postPadding, timeFormat="python"
    )
    if plotHexapod:
        for topic in HEXAPOD_TOPICS:
            hexData = getEfdData(
                client, topic, begin=begin, end=end, prePadding=prePadding, postPadding=postPadding
            )
            newCommands = {}
            for time, data in hexData.iterrows():
                newCommands[time] = topic
            commandTimes.update(newCommands)

    uniqueCommands = list(set(commandTimes.values()))
    colorCycle = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
    commandColors = {command: next(colorCycle) for command in uniqueCommands}
    for time, command in commandTimes.items():
        color = commandColors[command]
        axes["az"].axvline(time, linestyle="-.", alpha=commandAlpha, color=color)
        axes["el"].axvline(time, linestyle="-.", alpha=commandAlpha, color=color)
        axes["rot"].axvline(time, linestyle="-.", alpha=commandAlpha, color=color)

    # manually build the legend to avoid duplicating the labels due to multiple
    # commands of the same name
    handles = [
        Line2D([0], [0], color=color, linestyle="-.", label=label, alpha=commandAlpha)
        for label, color in commandColors.items()
    ]
    legendHandles.extend(handles)

    axes["az"].set_ylabel("Azimuth (deg)")
    axes["el"].set_ylabel("Elevation (deg)")
    axes["rot"].set_ylabel("Rotation (deg)")
    axes["rot"].set_xlabel("Time (UTC)")  # this is UTC because of the magic matplotlib does on time indices
    fig.suptitle(title)

    shaded_handle = Patch(facecolor=integrationColor, alpha=0.3, label="Shutter open period")
    legendHandles.append(shaded_handle)
    shaded_handle = Patch(facecolor=readoutColor, alpha=0.1, label="Readout period")
    legendHandles.append(shaded_handle)
    # put the legend under the plot itself
    axes["rot"].legend(handles=legendHandles, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)

    fig.tight_layout()
    plt.show()
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

    az = plotExposureTiming(client, toPlot, plotHexapod=True)
