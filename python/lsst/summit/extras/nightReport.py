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

from dataclasses import dataclass
import logging
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER
from astro_metadata_translator import ObservationInfo
from lsst.summit.utils.butlerUtils import makeDefaultLatissButler, getSeqNumsForDayObs, sanitize_day_obs

__all__ = ['NightReporter', 'saveReport', 'loadReport']

CALIB_VALUES = ['FlatField position', 'Park position', 'azel_target']
N_STARS_PER_SYMBOL = 6
MARKER_SEQUENCE = ['*', 'o', "D", 'P', 'v', "^", 's', '.', ',', 'o', 'v', '^',
                   '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h',
                   'H', '+', 'x', 'X', 'D', 'd', '|', '_']
SOUTHPOLESTAR = 'HD 185975'

PICKLE_TEMPLATE = "%s.pickle"

KEY_MAPPER = {'OBJECT': 'object',
              'EXPTIME': 'exposure_time',
              'IMGTYPE': 'observation_type',
              'MJD-BEG': 'datetime_begin',
              }

# TODO: DM-34250 rewrite (and document) this whole file.


def getValue(key, header, stripUnits=True):
    """Get a header value the Right Way.

    If it is available from the ObservationInfo, use that, either directly or
    via the KEY_MAPPER dict.
    If not, try to get it from the header.
    If it is not in the header, return None.
    """
    if key in KEY_MAPPER:
        key = KEY_MAPPER[key]

    if hasattr(header['ObservationInfo'], key):
        val = getattr(header['ObservationInfo'], key)
        if hasattr(val, 'value') and stripUnits:
            return val.value
        else:
            return val

    return header.get(key, None)


# wanted these to be on the class but it doesn't pickle itself nicely
def saveReport(reporter, savePath):
    # the reporter.butler seems to pickle OK but perhaps it should be
    # removed for saving?
    filename = os.path.join(savePath, PICKLE_TEMPLATE % reporter.dayObs)
    with open(filename, "wb") as dumpFile:
        pickle.dump(reporter, dumpFile)


def loadReport(loadPath, dayObs):
    filename = os.path.join(loadPath, PICKLE_TEMPLATE % dayObs)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found")
    with open(filename, "rb") as input_file:
        reporter = pickle.load(input_file)
    return reporter


@dataclass
class ColorAndMarker:
    '''Class for holding colors and marker symbols'''
    color: list
    marker: str = '*'


class NightReporter():

    def __init__(self, location, dayObs, deferLoadingData=False):
        self._supressAstroMetadataTranslatorWarnings()  # call early

        self.butler = makeDefaultLatissButler(location)
        if isinstance(dayObs, str):
            dayObs = sanitize_day_obs(dayObs)
            print('Converted string-format dayObs to integer for Gen3')
        self.dayObs = dayObs
        self.data = {}
        self.stars = None
        self.cMap = None
        if not deferLoadingData:
            self.rebuild()

    def _supressAstroMetadataTranslatorWarnings(self):
        """NB: must be called early"""
        logging.basicConfig()
        _astroLogger = logging.getLogger("lsst.obs.lsst.translators.latiss")
        _astroLogger.setLevel(logging.ERROR)

    def rebuild(self, dayObs=None):
        """Reload new observations, or load a different night"""
        dayToUse = self.dayObs
        if dayObs:
            # new day, so blow away old data
            # as scraping skips seqNums we've loaded!
            if dayObs != self.dayObs:
                self.data = {}
                self.dayObs = dayObs
            dayToUse = dayObs
        self._scrapeData(dayToUse)
        self.stars = self.getObservedObjects()
        self.cMap = self.makeStarColorAndMarkerMap(self.stars)

    def _scrapeData(self, dayObs):
        """Load data into self.data skipping as necessary. Don't call directly!

        Don't call directly as the rebuild() function zeros out data for when
        it's a new dayObs."""
        seqNums = getSeqNumsForDayObs(self.butler, dayObs)
        for seqNum in sorted(seqNums):
            if seqNum in self.data.keys():
                continue
            dataId = {'day_obs': dayObs, 'seq_num': seqNum, 'detector': 0}
            md = self.butler.get('raw.metadata', dataId)
            self.data[seqNum] = md.toDict()
            self.data[seqNum]['ObservationInfo'] = ObservationInfo(md)
        print(f"Loaded data for seqNums {sorted(seqNums)[0]} to {sorted(seqNums)[-1]}")

    def getUniqueValuesForKey(self, key, ignoreCalibs=True):
        values = []
        for seqNum in self.data.keys():
            v = getValue(key, self.data[seqNum])
            if ignoreCalibs is True and v in CALIB_VALUES:
                continue
            values.append(v)
        return list(set(values))

    def _makePolarPlot(self, azimuthsInDegrees, zenithAngles, marker="*-",
                       title=None, makeFig=True, color=None, objName=None):
        if makeFig:
            _ = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        ax.plot([a*np.pi/180 for a in azimuthsInDegrees], zenithAngles, marker, c=color, label=objName)
        if title:
            ax.set_title(title, va='bottom')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        return ax

    def makePolarPlotForObjects(self, objects=None, withLines=True):
        if not objects:
            objects = self.stars
        objects = self._safeListArg(objects)

        _ = plt.figure(figsize=(10, 10))

        for i, obj in enumerate(objects):
            azs = self.getAllValuesForKVPair('AZSTART', ("OBJECT", obj))
            els = self.getAllValuesForKVPair('ELSTART', ("OBJECT", obj))
            assert(len(azs) == len(els))
            if len(azs) == 0:
                print(f"WARNING: found no alt/az data for {obj}")
            zens = [90 - el for el in els]
            color = self.cMap[obj].color
            marker = self.cMap[obj].marker
            if withLines:
                marker += '-'

            ax = self._makePolarPlot(azs, zens, marker=marker, title=None, makeFig=False,
                                     color=color, objName=obj)
        lgnd = ax.legend(bbox_to_anchor=(1.05, 1), prop={'size': 15}, loc='upper left')
        for h in lgnd.legendHandles:
            size = 14
            if '-' in marker:
                size += 5
            h.set_markersize(size)

    def getAllValuesForKVPair(self, keyToGet, keyValPairAsTuple, uniqueOnly=False):
        """e.g. all the RA values for OBJECT=='HD 123'"""
        ret = []
        for seqNum in self.data.keys():
            if getValue(keyValPairAsTuple[0], self.data[seqNum]) == keyValPairAsTuple[1]:
                ret.append(getValue(keyToGet, self.data[seqNum]))
        if uniqueOnly:
            return list(set(ret))
        return ret

    @staticmethod
    def makeStarColorAndMarkerMap(stars):
        markerMap = {}
        colors = cm.rainbow(np.linspace(0, 1, N_STARS_PER_SYMBOL))
        for i, star in enumerate(stars):
            markerIndex = i//(N_STARS_PER_SYMBOL)
            colorIndex = i%(N_STARS_PER_SYMBOL)
            markerMap[star] = ColorAndMarker(colors[colorIndex], MARKER_SEQUENCE[markerIndex])
        return markerMap

    def getObjectValues(self, key, objName):
        return self.getAllValuesForKVPair(key, ('OBJECT', objName), uniqueOnly=False)

    def getAllHeaderKeys(self):
        return list(list(self.data.items())[0][1].keys())

    @staticmethod  # designed for use in place of user-provided filter callbacks so gets self via call
    def isDispersed(self, seqNum):
        filt = self.data[seqNum]['ObservationInfo'].physical_filter
        grating = filt.split(FILTER_DELIMITER)[1]
        if "EMPTY" not in grating.upper():
            return True
        return False

    def _calcObjectAirmasses(self, objects, filterFunc=None):
        if filterFunc is None:
            def noopFilter(*args, **kwargs):
                return True
            filterFunc = noopFilter
        airMasses = {}
        for star in objects:
            seqNums = self.getObjectValues('SEQNUM', star)
            airMasses[star] = [(self.data[seqNum]['ObservationInfo'].boresight_airmass,
                                getValue('MJD-BEG', self.data[seqNum]))
                               for seqNum in sorted(seqNums) if filterFunc(self, seqNum)]
        return airMasses

    def getSeqNums(self, filterFunc, *args, **kwargs):
        """Get seqNums for a corresponding filtering function.

        filterFunc is called with (self, seqNum) and must return a bool."""
        seqNums = []
        for seqNum in self.data.keys():
            if filterFunc(self, seqNum, *args, **kwargs):
                seqNums.append(seqNum)
        return seqNums

    def getObservedObjects(self):
        return self.getUniqueValuesForKey('OBJECT')

    def plotPerObjectAirMass(self, objects=None, airmassOneAtTop=True, filterFunc=None):
        """filterFunc is self as the first argument and seqNum as second."""
        if not objects:
            objects = self.stars

        objects = self._safeListArg(objects)

        # lazy to always recalculate but it's not *that* slow
        # and optionally passing around can be messy
        # TODO: keep some of this in class state
        airMasses = self._calcObjectAirmasses(objects, filterFunc=filterFunc)

        _ = plt.figure(figsize=(10, 6))
        for star in objects:
            if airMasses[star]:  # skip stars fully filtered out by callbacks
                ams, times = np.asarray(airMasses[star])[:, 0], np.asarray(airMasses[star])[:, 1]
            else:
                continue
            color = self.cMap[star].color
            marker = self.cMap[star].marker
            plt.plot(times, ams, color=color, marker=marker, label=star, ms=10, ls='')

        plt.ylabel('Airmass', fontsize=20)
        if airmassOneAtTop:
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
        _ = plt.legend(bbox_to_anchor=(1, 1.025), prop={'size': 15}, loc='upper left')

    def printObsTable(self, imageType=None, tailNumber=0):
        """Print a table of the days observations.

        Parameters
        ----------
        imageType : str
            Only consider images with this image type
        tailNumber : int
            Only print out the last n entries in the night
        """
        lines = []
        if not imageType:
            seqNums = self.data.keys()
        else:
            seqNums = [s for s in self.data.keys()
                       if self.data[s]['ObservationInfo'].observation_type == imageType]

        seqNums = sorted(seqNums)
        for i, seqNum in enumerate(seqNums):
            try:
                expTime = self.data[seqNum]['ObservationInfo'].exposure_time.value
                filt = self.data[seqNum]['ObservationInfo'].physical_filter
                imageType = self.data[seqNum]['ObservationInfo'].observation_type
                d1 = self.data[seqNum]['ObservationInfo'].datetime_begin
                obj = self.data[seqNum]['ObservationInfo'].object
                if i == 0:
                    d0 = d1
                dt = (d1-d0)
                d0 = d1
                timeOfDay = d1.isot.split('T')[1]
                msg = f'{seqNum:4} {imageType:9} {obj:10} {timeOfDay} {filt:25} {dt.sec:6.1f}  {expTime:2.2f}'
            except KeyError:
                msg = f'{seqNum:4} - error parsing headers/observation info! Check the file'
            lines.append(msg)

        print(r"{seqNum} {imageType} {obj} {timeOfDay} {filt} {timeSinceLastExp} {expTime}")
        for line in lines[-tailNumber:]:
            print(line)

    def calcShutterOpenEfficiency(self, seqMin=0, seqMax=0):
        if seqMin == 0:
            seqMin = min(self.data.keys())
        if seqMax == 0:
            seqMax = max(self.data.keys())
        assert seqMax > seqMin
        assert (seqMin in self.data.keys())
        assert (seqMax in self.data.keys())

        timeStart = self.data[seqMin]['ObservationInfo'].datetime_begin
        timeEnd = self.data[seqMax]['ObservationInfo'].datetime_end
        expTimeSum = 0
        for seqNum in range(seqMin, seqMax+1):
            if seqNum not in self.data.keys():
                print(f"Warning! No data found for seqNum {seqNum}")
                continue
            expTimeSum += self.data[seqNum]['ObservationInfo'].exposure_time.value

        timeOnSky = (timeEnd - timeStart).sec
        efficiency = expTimeSum/timeOnSky
        print(f"{100*efficiency:.2f}% shutter open in seqNum range {seqMin} and {seqMax}")
        print(f"Total integration time = {expTimeSum:.1f}s")
        return efficiency

    @staticmethod
    def _safeListArg(arg):
        if type(arg) == str:
            return [arg]
        assert(type(arg) == list), f"Expect list, got {arg}"
        return arg
