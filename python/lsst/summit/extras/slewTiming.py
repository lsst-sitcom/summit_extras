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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime
from astropy.time import TimeDelta

import lsst.summit.utils.butlerUtils as butlerUtils
from lsst_efd_client import merge_packed_time_series as mpts
from lsst.summit.utils.efdUtils import getEfdData

__all__ = ['plotExposureTiming']

READOUT_TIME = TimeDelta(2.3, format='sec')

COMMANDS_TO_QUERY = [
    # at the time of writing this was all the commands that existed for ATPtg
    # and ATMCS. We explicitly exclude the 20Hz ATMCS.command_trackTarget
    # command, and include all others. Perhaps this should be done dynamically
    # by using the findTopics function and removing command_trackTarget from
    # the list?
    'lsst.sal.ATPtg.command_azElTarget',
    'lsst.sal.ATPtg.command_disable',
    'lsst.sal.ATPtg.command_enable',
    'lsst.sal.ATPtg.command_exitControl',
    'lsst.sal.ATPtg.command_offsetAbsorb',
    'lsst.sal.ATPtg.command_offsetAzEl',
    'lsst.sal.ATPtg.command_offsetClear',
    'lsst.sal.ATPtg.command_offsetPA',
    'lsst.sal.ATPtg.command_offsetRADec',
    'lsst.sal.ATPtg.command_pointAddData',
    'lsst.sal.ATPtg.command_pointLoadModel',
    'lsst.sal.ATPtg.command_pointNewFile',
    'lsst.sal.ATPtg.command_poriginAbsorb',
    'lsst.sal.ATPtg.command_poriginClear',
    'lsst.sal.ATPtg.command_poriginOffset',
    'lsst.sal.ATPtg.command_poriginXY',
    'lsst.sal.ATPtg.command_raDecTarget',
    'lsst.sal.ATPtg.command_rotOffset',
    'lsst.sal.ATPtg.command_standby',
    'lsst.sal.ATPtg.command_start',
    'lsst.sal.ATPtg.command_startTracking',
    'lsst.sal.ATPtg.command_stopTracking',
    'lsst.sal.ATMCS.command_disable',
    'lsst.sal.ATMCS.command_enable',
    'lsst.sal.ATMCS.command_exitControl',
    'lsst.sal.ATMCS.command_setInstrumentPort',
    'lsst.sal.ATMCS.command_standby',
    'lsst.sal.ATMCS.command_start',
    'lsst.sal.ATMCS.command_startTracking',
    'lsst.sal.ATMCS.command_stopTracking',
    # 'lsst.sal.ATMCS.command_trackTarget',  # exclude the 20Hz data
]


def getMountPositionData(client, tBegin, tEnd, prePadding=0, postPadding=0):
    mountPosition = getEfdData(
        client,
        "lsst.sal.ATMCS.mount_AzEl_Encoders",
        begin=tBegin,
        end=tEnd,
        prePadding=prePadding,
        postPadding=postPadding
    )
    nasmythPosition = getEfdData(
        client, "lsst.sal.ATMCS.mount_Nasmyth_Encoders",
        begin=tBegin,
        end=tEnd,
        prePadding=prePadding,
        postPadding=postPadding
    )

    az = mpts(mountPosition, 'azimuthCalculatedAngle', stride=1)
    el = mpts(mountPosition, 'elevationCalculatedAngle', stride=1)
    rot = mpts(nasmythPosition, 'nasmyth2CalculatedAngle', stride=1)
    return az, el, rot


def getCommandsIssued(client, commands, tBegin, tEnd, prePadding, postPadding):
    commandTimes = {}
    for command in commands:
        data = getEfdData(
            client,
            command,
            begin=tBegin,
            end=tEnd,
            prePadding=prePadding,
            postPadding=postPadding,
            warn=False  # most commands will not be issue so we expect many empty queries
        )
        if not data.empty:
            for time, _ in data.iterrows():
                # this is much the most simple data structure, and the chance
                # of commands being *exactly* simultaneous is minimal so try
                # it like this, and just raise if we get collisions for now. So
                # far in testing this seems to be just fine.
                if time in commandTimes:
                    raise ValueError(f"There is already a command at {time=} - make a better data structure!")
                commandTimes[time] = command
    return commandTimes


def getAxesInPosition(client, tBegin, tEnd, prePadding, postPadding):
    return getEfdData(
        client,
        'lsst.sal.ATMCS.logevent_allAxesInPosition',
        begin=tBegin,
        end=tEnd,
        prePadding=prePadding,
        postPadding=postPadding
    )


def plotExposureTiming(client, expRecords, prePadding=1, postPadding=3):
    inPositionAlpha = 0.5
    commandAlpha = 0.5
    integrationColor = 'grey'
    readoutColor = 'blue'

    legendHandles = []

    expRecords = sorted(expRecords, key=lambda x: (x.day_obs, x.seq_num))  # ensure we're sorted

    startSeqNum = expRecords[0].seq_num
    endSeqNum = expRecords[-1].seq_num
    title = f"Mount command timings for seqNums {startSeqNum} - {endSeqNum}"

    # redo all the data getting functions to take tBegin and end
    tBegin = expRecords[0].timespan.begin
    tEnd = expRecords[-1].timespan.end

    az, el, rot = getMountPositionData(client, tBegin, tEnd, prePadding=prePadding, postPadding=postPadding)

    # Create a figure with a grid specification
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 1, hspace=0)

    # Create axes with shared x-axis
    azimuth_ax = fig.add_subplot(gs[0, 0])
    elevation_ax = fig.add_subplot(gs[1, 0], sharex=azimuth_ax)
    rotation_ax = fig.add_subplot(gs[2, 0], sharex=azimuth_ax)

    # Store axes in a dictionary
    axes = {'az': azimuth_ax, 'el': elevation_ax, 'rot': rotation_ax}

    axes['az'].plot(az['azimuthCalculatedAngle'])
    axes['el'].plot(el['elevationCalculatedAngle'])
    axes['rot'].plot(rot['nasmyth2CalculatedAngle'])

    # shade the expRecords' regions
    for i, record in enumerate(expRecords):
        # these need to be in UTC because matplotlib magic turns all the axis
        # timings into UTC when plotting from a dataframe.
        start = record.timespan.begin.utc.to_value("isot")
        stop = record.timespan.end.utc.to_value("isot")
        readoutEnd = (record.timespan.end + READOUT_TIME).utc.to_value("isot")
        seqNum = record.seq_num
        for axName, ax in axes.items():
            ax.axvspan(start, stop, color=integrationColor, alpha=0.3)
            ax.axvspan(stop, readoutEnd, color=readoutColor, alpha=0.1)
            if axName == 'el':  # only add seqNum annotation to bottom axis
                label = f'seqNum = {seqNum}'
                start_date = datetime.fromisoformat(start)
                stop_date = datetime.fromisoformat(stop)
                midpoint = start_date + (stop_date - start_date) / 2
                ax.annotate(label, xy=(midpoint, 0.5), xycoords=('data', 'axes fraction'),
                            ha='center', va='center', fontsize=10, color='black')

    inPostionTransitions = getAxesInPosition(client, expRecords, prePadding, postPadding)
    for time, data in inPostionTransitions.iterrows():
        inPosition = data['inPosition']
        if inPosition:
            axes['az'].axvline(time, color="green", linestyle="--", alpha=inPositionAlpha)
            axes['el'].axvline(time, color="green", linestyle="--", alpha=inPositionAlpha)
            axes['rot'].axvline(time, color="green", linestyle="--", alpha=inPositionAlpha)
        else:
            axes['az'].axvline(time, color="red", linestyle="-", alpha=inPositionAlpha)
            axes['el'].axvline(time, color="red", linestyle="-", alpha=inPositionAlpha)
            axes['rot'].axvline(time, color="red", linestyle="-", alpha=inPositionAlpha)
    handle = Line2D(
        [0],
        [0],
        color='green',
        linestyle='--',
        label='allAxesInPosition=True',
        alpha=inPositionAlpha
    )
    legendHandles.append(handle)
    handle = Line2D(
        [0],
        [0],
        color='red',
        linestyle='-',
        label='allAxesInPosition=False',
        alpha=inPositionAlpha
    )
    legendHandles.append(handle)

    commandTimes = getCommandsIssued(client, COMMANDS_TO_QUERY, tBegin, tEnd, prePadding, postPadding)
    uniqueCommands = list(set(commandTimes.values()))
    colorCycle = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    commandColors = {command: next(colorCycle) for command in uniqueCommands}
    for time, command in commandTimes.items():
        color = commandColors[command]
        axes['az'].axvline(time, linestyle="-.", alpha=commandAlpha, color=color)
        axes['el'].axvline(time, linestyle="-.", alpha=commandAlpha, color=color)
        axes['rot'].axvline(time, linestyle="-.", alpha=commandAlpha, color=color)

    # manually build the legend to avoid duplicating the labels due to multiple
    # commands of the same name
    handles = [Line2D([0], [0], color=color, linestyle='-.', label=label, alpha=commandAlpha)
               for label, color in commandColors.items()]
    legendHandles.extend(handles)

    # Hide x-axis labels for all but the bottom plot
    for ax in [azimuth_ax, elevation_ax]:
        # can this be done with calling plt.?
        plt.setp(ax.get_xticklabels(), visible=False)

    axes['az'].set_ylabel('Azimuth (deg)')
    axes['el'].set_ylabel('Elevation (deg)')
    axes['rot'].set_ylabel('Rotation (deg)')

    axes['rot'].set_xlabel('Time (UTC, really)')
    # XXX set the title here

    shaded_handle = Patch(facecolor=integrationColor, alpha=0.3, label='Shutter open period')
    legendHandles.append(shaded_handle)
    shaded_handle = Patch(facecolor=readoutColor, alpha=0.1, label='Readout period')
    legendHandles.append(shaded_handle)
    # put the legend under the plot itself
    axes['rot'].legend(handles=legendHandles, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

    # can this be fig.tight_layout?
    plt.tight_layout()
    plt.show()

    return az


if __name__ == "__main__":
    from lsst.summit.utils.efdUtils import makeEfdClient
    client = makeEfdClient()
    butler = butlerUtils.makeDefaultLatissButler(embargo=True)

    where = 'exposure.day_obs=20240215'
    records = list(butler.registry.queryDimensionRecords('exposure', where=where))
    records = sorted(records, key=lambda x: (x.day_obs, x.seq_num))
    print(f'Found {len(records)} records from {len(set(r.day_obs for r in records))} days')

    expRecords = [records[61], records[62]]
    az = plotExposureTiming(client, expRecords)
