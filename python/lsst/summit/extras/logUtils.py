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
# This file is part of summit_utils.
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

import logging
import math
import lsst.summit.utils.butlerUtils as butlerUtils

__all__ = ["LogBrowser",
           ]

_LOG = logging.getLogger(__name__)


class LogBrowser():
    # add the end of the common part of any similar errors you want collapsed here
    # this part of the string won't be lost, but is used to classify things
    # as the same animal
    SPECIAL_ZOO_CASES = ['with gufunc signature (n?,k),(k,m?)->(n?,m?)',
                         ]

    def __init__(self, taskName, collection, doCache=True):
        self.taskName = taskName
        self.collection = collection
        self.doCache = doCache

        self.log = _LOG.getChild("logBrowser")
        self.butler = butlerUtils.makeDefaultLatissButler(extraCollections=[collection])

        self.expRecords = []
        self.logs = []
        if self.doCache:
            self.expRecords = self._getExpRecords()
            self.logs = self._loadLogs(self.expRecords)

    def _getExpRecords(self):
        expRecords = self.butler.registry.queryDimensionRecords("exposure",
                                                                collections=self.collection,
                                                                datasets='raw')
        expRecords = list(set(expRecords))
        self.log.info(f"Found {len(expRecords)} exposure records in collection")
        return expRecords

    @staticmethod
    def recordToDataId(record):
        return {'day_obs': record.day_obs, 'seq_num': record.seq_num}

    def dataIdToRecord(self, dataId):
        records = []
        for record in self.expRecords:
            match = True
            for k in dataId.keys():
                if not getattr(record, k) == dataId[k]:
                    match = False
                    break

            if not match:  # pass the break through
                continue

            records.append(record)
        if len(records) != 1:
            raise ValueError(f"Found {len(records)} records for {dataId}, expected exactly 1")
        return records[0]

    def _loadLogs(self, expRecords):
        logs = {}
        logName = f"{self.taskName}_log"
        for i, record in enumerate(expRecords):
            if (i+1) % 100 == 0:
                self.log.info(f"Loaded {i+1} logs...")
            dataId = {'day_obs': record.day_obs, 'seq_num': record.seq_num}
            log = self.butler.get(logName, dataId=dataId, detector=0)
            logs[record] = log
        return logs

    def getPasses(self):
        fails = self.getFailRecords()
        passes = [r for r in self.expRecords if r not in fails]
        return passes

    def printPasses(self):
        passes = self.getPasses()
        for record in passes:
            print(self.recordToDataId(record))
        # return sortRecordsByDayObsThenSeqNum(passes)

    def countFails(self):
        print(f"{len(self.getFailRecords())} fail cases found")

    def getFailRecords(self):
        fails = []
        for record, log in self.logs.items():
            if log[-1].message.find('failed') != -1:
                fails.append(record)
        return fails

    def printFailLogs(self, full=False):
        fails = self.getFailRecords()
        for fail in fails:
            dataId = self.recordToDataId(fail)
            print(f'\n{dataId}:')
            log = self.logs[fail]
            if full:
                for line in log:
                    print(line.message)
            else:
                msg = log[-1].message
                parts = msg.split('Exception ')
                if len(parts) == 2:
                    print(parts[1])
                else:
                    print(msg)

    def doFailZoology(self, giveExampleId=False):
        zoo = {}
        examples = {}
        fails = self.getFailRecords()
        for fail in fails:
            dataId = self.recordToDataId(fail)
            log = self.logs[fail]
            msg = log[-1].message
            parts = msg.split('Exception ')
            if len(parts) != 2:  # pretty sure all fails contain one and only one 'Exception' but be safe
                self.log.warning(f'Surprise parsing log for {dataId}')
                continue
            else:
                error = parts[1]
                for error_string in self.SPECIAL_ZOO_CASES:
                    if error.find(error_string) != -1:
                        error = error.split(error_string)[0] + error_string
                if error not in zoo:
                    zoo[error] = 1
                    if giveExampleId:
                        examples[error] = dataId
                else:
                    zoo[error] += 1

        pad = 0  # don't pad when giving examples, it looks weird
        if not giveExampleId:
            maxCount = max([v for v in zoo.values()])
            pad = math.ceil(math.log10(maxCount))  # number of digits in the largest count

        for error in sorted(zoo.keys()):
            count = zoo[error]
            print(f"{count:{pad}} instance{'s' if count > 1 else ' '} of {error}")
            if giveExampleId:
                print(f"example dataId: {examples[error]}\n")
