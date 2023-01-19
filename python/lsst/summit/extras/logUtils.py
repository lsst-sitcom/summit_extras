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

import logging
import math

__all__ = ["LogBrowser"]

_LOG = logging.getLogger(__name__)


class LogBrowser():
    """A convenience class for helping identify different failure modes within
    a processing collection.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler. Must contain the collection to be examined.
    taskName : `str`
        The name of the task, e.g. ``isr``, ``characterizeImage``, etc.
    collection : `str`
        The processing collection to use.

    Notes
    -----
    Many tasks throw errors with values in them, meaning the ``doFailZoology``
    function doesn't collapse them down to a single failure case as one would
    like. If this is the case, take the first part of the message that is
    common among the ones you would like to be classed together, and add it to
    the class property ``SPECIAL_ZOO_CASES`` to declare a new type of error
    animal.

    example usage:
    logBrowser = LogBrowser(buter, taskName, collection)
    fail = 'TaskError: Fatal astrometry failure detected: mean on-sky distance'
    logBrowser.SPECIAL_ZOO_CASES.append(fail)
    logBrowser.doFailZoology()
    """
    IGNORE_LOGS_FROM = [
        # butler.datastores is verbose by default and not interesting to most
        'lsst.daf.butler.datastores',
    ]
    SPECIAL_ZOO_CASES = ['with gufunc signature (n?,k),(k,m?)->(n?,m?)',
                         ]

    def __init__(self, butler, taskName, collection):
        self.taskName = taskName
        self.collection = collection

        self.log = _LOG.getChild("logBrowser")
        self.butler = butler

        self.dataRefs = self._getDataRefs()
        self.logs = self._loadLogs(self.dataRefs)

    def _getDataRefs(self):
        """Get the dataRefs for the specified task and collection.

        Returns
        -------
        dataRefs : `list` [`lsst.daf.butler.core.datasets.ref.DatasetRef`]
        """
        results = self.butler.registry.queryDatasets(f'{self.taskName}_log',
                                                     collections=self.collection,
                                                     findFirst=True)
        results = list(set(results))
        self.log.info(f"Found {len(results)} datasets in collection for task {self.taskName}")
        return sorted(results)

    def _loadLogs(self, dataRefs):
        """Load all the logs for the dataRefs.

        Returns
        -------
        logs : `dict` {`lsst.daf.butler.core.datasets.ref.DatasetRef`:
                       `lsst.daf.butler.core.logging.ButlerLogRecords`}
            A dict of all the logs, keyed by their dataRef.
        """
        logs = {}
        for i, dataRef in enumerate(dataRefs):
            if (i+1) % 100 == 0:
                self.log.info(f"Loaded {i+1} logs...")
            log = self.butler.getDirect(dataRef)
            logs[dataRef] = log
        return logs

    def getPassingDataIds(self):
        """Get the dataIds for all passes within the collection for the task.

        Returns
        -------
        dataIds : `list` [`lsst.daf.butler.dimensions.DataCoordinate`]
        """
        fails = self._getFailDataRefs()
        passes = [r.dataId for r in self.dataRefs if r not in fails]
        return passes

    def getFailingDataIds(self):
        """Get the dataIds for all fails within the collection for the task.

        Returns
        -------
        dataIds : `list` [`lsst.daf.butler.dimensions.DataCoordinate`]
        """
        fails = self._getFailDataRefs()
        return [r.dataId for r in fails]

    def printPasses(self):
        """Print out all the passing dataIds.
        """
        passes = self.getPassingDataIds()
        for dataId in passes:
            print(dataId)

    def printFails(self):
        """Print out all the failing dataIds.
        """
        fails = self.getFailingDataIds()
        for dataId in fails:
            print(dataId)

    def countFails(self):
        """Print a count of all the failing dataIds.
        """
        print(f"{len(self._getFailDataRefs())} failing cases found")

    def countPasses(self):
        """Print a count of all the passing dataIds.
        """
        print(f"{len(self.getPassingDataIds())} passing cases found")

    def _getFailDataRefs(self):
        """Get a list of all the failing dataRefs.

        Note that these are dataset references to the logs, and as such are
        not fails themselves, but logs containing the fail messages, and as
        such the item of interest for the failures are their dataIds. This is
        why ``_getFailDataRefs()`` is a private method, but getFailingDataIds
        is the public API.

        Returns
        -------
        logs : `list` [`lsst.daf.butler.core.datasets.ref.DatasetRef`]
            A list of all the failing dataRefs.
        """
        fails = []
        for dataRef, log in self.logs.items():
            # dereferencing a log with [] gives the individual lines in it,
            # each containing a level, message, etc.
            # the final task failure message always comes in the last line
            # of the log and contains the string 'failed' as this is the
            # pipeline executor reporting on success/fail and the time and id.
            if log[-1].message.find('failed') != -1:
                fails.append(dataRef)
        return fails

    def _printLineIf(self, logLine):
        """Print the line if the name of the logger isn't in IGNORE_LOGS_FROM.

        Parameters
        ----------
        logLine : `lsst.daf.butler.logging.ButlerLogRecord`
            The log line to print the message from.
        """
        skip = False
        for skipTask in self.IGNORE_LOGS_FROM:
            if logLine.name.find(skipTask) != -1:
                skip = True
                break
        if not skip:
            self._printFormattedLine(logLine)

    @staticmethod
    def _printFormattedLine(logLine):
        """Print the line, formatted as it would be for a normal task.

        Parameters
        ----------
        logLine : `lsst.daf.butler.logging.ButlerLogRecord`
            The log line to print the message from.
        """
        print(f"{logLine.levelname} {logLine.name}: {logLine.message}")

    def printFailLogs(self, full=False):
        """Print the logs of all failing task instances.

        Parameters
        ----------
        full : `bool`, optional
            Prints the full log if true, otherwise just prints the last line
            containing the exception message. This defaults to False because
            logs can be very long when printed in full, and printing all in
            full can be many many thousands of lines.
        """
        fails = self._getFailDataRefs()
        for dataRef in fails:
            print(f'\n{dataRef.dataId}:')
            log = self.logs[dataRef]
            if full:  # print the whole thing
                for line in log:
                    self._printLineIf.print(line)
            else:
                # print the last line from the Exception onwards if found,
                # failing over to printing the whole thing just in case.
                msg = log[-1].message
                parts = msg.split('Exception ')
                if len(parts) == 2:
                    print(parts[1])
                else:
                    print(msg)

    def doFailZoology(self, giveExampleId=False):
        """Print all the different types of error, with a count for how many of
        each type occurred.

        Parameters
        ----------
        giveExampleId : `bool`, optional
            If true, for each type of error seen, print an example dataId. This
            can be useful if you want to rerun a single image from the command
            line to debug a particular type of failure mode.
        """
        zoo = {}
        examples = {}
        fails = self._getFailDataRefs()
        for dataRef in fails:
            log = self.logs[dataRef]
            msg = log[-1].message  # log[-1].message is the text of the last line of the log
            parts = msg.split('Exception ')
            if len(parts) != 2:  # pretty sure all fails contain one and only one 'Exception' but be safe
                self.log.warning(f'Surprise parsing log for {dataRef.dataId}')
                continue
            else:
                error = parts[1]
                for error_string in self.SPECIAL_ZOO_CASES:
                    if error.find(error_string) != -1:
                        error = error.split(error_string)[0] + error_string + '...'
                if error not in zoo:
                    zoo[error] = 1
                    if giveExampleId:
                        examples[error] = dataRef.dataId
                else:
                    zoo[error] += 1

        pad = 0  # don't pad when giving examples, it looks weird
        if not giveExampleId:
            if zoo.values():
                maxCount = max([v for v in zoo.values()])
                pad = math.ceil(math.log10(maxCount))  # number of digits in the largest count

        for error in sorted(zoo.keys()):
            count = zoo[error]
            print(f"{count:{pad}} instance{'s' if count > 1 else ' '} of {error}")
            if giveExampleId:
                print(f"example dataId: {examples[error]}\n")

    def printSingleLog(self, dataId, full=True):
        """Convenience function for printing a single log by its dataId.

        Useful because you are given example dataIds by `doFailZoology()` but
        printing all the logs and looking for that id is not practical.

        Parameters
        ----------
        dataId : `dict` or `lsst.daf.butler.dimensions.DataCoordinate`
            The dataId.
        full : `bool`, optional
            Print the log in full, or just the exception?
        """
        dRefs = [d for d in self.dataRefs if d.dataId == dataId]
        if len(dRefs) != 1:
            raise ValueError(f"Found {len(dRefs)} for {dataId}, expected exactly 1.")
        dataRef = dRefs[0]

        print(f'\n{dataRef.dataId}:')
        log = self.logs[dataRef]
        if full:
            for line in log:
                self._printLineIf(line)
        else:
            msg = log[-1].message  # log[-1].message is the text of the last line of the log
            parts = msg.split('Exception ')
            if len(parts) == 2:
                print(parts[1])
            else:
                print(msg)
