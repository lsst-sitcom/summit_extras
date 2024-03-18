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
from typing import Tuple

import astropy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import TimeDelta
from lsst_efd_client import EfdClient
from lsst_efd_client import merge_packed_time_series as mpts
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import lsst.daf.butler as dafButler
import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.summit.utils.efdUtils import getCommands, getEfdData

__all__ = ["plotExposureTiming"]

READOUT_TIME = TimeDelta(2.3, format="sec")

COMMANDS_TO_QUERY = [
    # at the time of writing this was all the commands that existed for ATPtg
    # and ATMCS. We explicitly exclude the 20Hz ATMCS.command_trackTarget
    # command, and include all others. Perhaps this should be done dynamically
    # by using the findTopics function and removing command_trackTarget from
    # the list?
    "lsst.sal.ATPtg.command_azElTarget",
    "lsst.sal.ATPtg.command_disable",
    "lsst.sal.ATPtg.command_enable",
    "lsst.sal.ATPtg.command_exitControl",
    "lsst.sal.ATPtg.command_offsetAbsorb",
    "lsst.sal.ATPtg.command_offsetAzEl",
    "lsst.sal.ATPtg.command_offsetClear",
    "lsst.sal.ATPtg.command_offsetPA",
    "lsst.sal.ATPtg.command_offsetRADec",
    "lsst.sal.ATPtg.command_pointAddData",
    "lsst.sal.ATPtg.command_pointLoadModel",
    "lsst.sal.ATPtg.command_pointNewFile",
    "lsst.sal.ATPtg.command_poriginAbsorb",
    "lsst.sal.ATPtg.command_poriginClear",
    "lsst.sal.ATPtg.command_poriginOffset",
    "lsst.sal.ATPtg.command_poriginXY",
    "lsst.sal.ATPtg.command_raDecTarget",
    "lsst.sal.ATPtg.command_rotOffset",
    "lsst.sal.ATPtg.command_standby",
    "lsst.sal.ATPtg.command_start",
    "lsst.sal.ATPtg.command_startTracking",
    "lsst.sal.ATPtg.command_stopTracking",
    "lsst.sal.ATMCS.command_disable",
    "lsst.sal.ATMCS.command_enable",
    "lsst.sal.ATMCS.command_exitControl",
    "lsst.sal.ATMCS.command_setInstrumentPort",
    "lsst.sal.ATMCS.command_standby",
    "lsst.sal.ATMCS.command_start",
    "lsst.sal.ATMCS.command_startTracking",
    "lsst.sal.ATMCS.command_stopTracking",
    # 'lsst.sal.ATMCS.command_trackTarget',  # exclude the 20Hz data
]


def getMountPositionData(
    client: EfdClient,
    begin: astropy.time.Time,
    end: astropy.time.Time,
    prePadding: int = 0,
    postPadding: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrieve the mount position data from the EFD.

    Parameters
    ----------
    client : `EfdClient`
        The EFD client used to retrieve the data.
    begin : `astropy.time.Time`
        The start time of the data retrieval window.
    end : `astropy.time.Time`
        The end time of the data retrieval window.
    prePadding : `float`, optional
        The amount of time to pad before the begin time, in seconds.
    postPadding : `float`, optional
        The amount of time to pad after the end time, in seconds.

    Returns
    -------
    alt, ax, rot : `tuple` of `pd.DataFrame`
        A tuple containing the azimuth, elevation, and rotation data as
        dataframes.
    """
    mountPosition = getEfdData(
        client,
        "lsst.sal.ATMCS.mount_AzEl_Encoders",
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )
    nasmythPosition = getEfdData(
        client,
        "lsst.sal.ATMCS.mount_Nasmyth_Encoders",
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )

    az = mpts(mountPosition, "azimuthCalculatedAngle", stride=1)
    el = mpts(mountPosition, "elevationCalculatedAngle", stride=1)
    rot = mpts(nasmythPosition, "nasmyth2CalculatedAngle", stride=1)
    return az, el, rot


def getAxesInPosition(
    client: EfdClient,
    begin: astropy.time.Time,
    end: astropy.time.Time,
    prePadding: int = 0,
    postPadding: int = 0,
) -> pd.DataFrame:
    return getEfdData(
        client,
        "lsst.sal.ATMCS.logevent_allAxesInPosition",
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )


def plotExposureTiming(
    client: EfdClient, expRecords: dafButler.DimensionRecord, prePadding: int = 1, postPadding: int = 3
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
    title = f"Mount command timings for seqNums {startSeqNum} - {endSeqNum}"

    begin = expRecords[0].timespan.begin
    end = expRecords[-1].timespan.end

    az, el, rot = getMountPositionData(client, begin, end, prePadding=prePadding, postPadding=postPadding)

    # Create a figure with a grid specification and have axes share x
    # and have no room between each
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 1, hspace=0)
    azimuth_ax = fig.add_subplot(gs[0, 0])
    elevation_ax = fig.add_subplot(gs[1, 0], sharex=azimuth_ax)
    rotation_ax = fig.add_subplot(gs[2, 0], sharex=azimuth_ax)
    axes = {"az": azimuth_ax, "el": elevation_ax, "rot": rotation_ax}

    # plot the telemetry
    axes["az"].plot(az["azimuthCalculatedAngle"])
    axes["el"].plot(el["elevationCalculatedAngle"])
    axes["rot"].plot(rot["nasmyth2CalculatedAngle"])

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
    inPostionTransitions = getAxesInPosition(client, begin, end, prePadding, postPadding)
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
        [0], [0], color="green", linestyle="--", label="allAxesInPosition=True", alpha=inPositionAlpha
    )
    legendHandles.append(handle)
    handle = Line2D(
        [0], [0], color="red", linestyle="-", label="allAxesInPosition=False", alpha=inPositionAlpha
    )
    legendHandles.append(handle)

    # place vertical lines at the times when commands were issued
    commandTimes = getCommands(
        client, COMMANDS_TO_QUERY, begin, end, prePadding, postPadding, timeFormat="python"
    )
    if plotHexapod:
        hexMoveStarts = getEfdData(
            client,
            "lsst.sal.ATAOS.logevent_hexapodCorrectionStarted",
            expRecord=record,
            prePadding=prePadding,
            postPadding=postPadding,
        )
        hexMoveEnds = getEfdData(
            client,
            "lsst.sal.ATAOS.logevent_hexapodCorrectionCompleted",
            expRecord=record,
            prePadding=prePadding,
            postPadding=postPadding,
        )
        newCommands = {}
        for time, data in hexMoveStarts.iterrows():
            newCommands[time] = "lsst.sal.ATAOS.logevent_hexapodCorrectionStarted"
        for time, data in hexMoveEnds.iterrows():
            newCommands[time] = "lsst.sal.ATAOS.logevent_hexapodCorrectionCompleted"
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
    from lsst.summit.extras.slewTiming import plotExposureTiming  # noqa: F811
    from lsst.summit.utils.efdUtils import makeEfdClient

    client = makeEfdClient()
    butler = butlerUtils.makeDefaultLatissButler(embargo=True)

    where = "exposure.day_obs=20240215"
    records = list(butler.registry.queryDimensionRecords("exposure", where=where))
    records = sorted(records, key=lambda x: (x.day_obs, x.seq_num))
    print(f"Found {len(records)} records from {len(set(r.day_obs for r in records))} days")

    expRecords = [records[61], records[62]]
    az = plotExposureTiming(client, expRecords)
