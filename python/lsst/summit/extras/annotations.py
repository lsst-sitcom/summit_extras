# This file is part of rapid_analysis.
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

from lsst.rapid.analysis.imageSorter import ImageSorter, TAGS


def _idTrans(dataIdDictOrTuple):
    """Take a dataId and turn it into the internal tuple format.
    """
    if type(dataIdDictOrTuple) == tuple:
        return dataIdDictOrTuple
    elif type(dataIdDictOrTuple) == dict:
        return (dataIdDictOrTuple['dayObs'], dataIdDictOrTuple['seqNum'])
    else:
        raise RuntimeError(f'Failed to parse dataId {dataIdDictOrTuple}')


class Annotations():
    """Class for interfacing with annotations, as written by the imageSorter.
    """

    def __init__(self, filename):
        self.filename = filename
        self.tags, self.notes = self._load(filename)

    def _load(self, filename):
        """Load tags and notes from specified file.
        """
        tags, notes = ImageSorter.loadAnnotations(filename)
        return tags, notes

    def getTags(self, dataId):
        """Get the tags for a specified dataId.

        Empty string means no tags, None means not examined"""
        return self.tags.get(_idTrans(dataId), None)

    def getNotes(self, dataId):
        """Get the notes for the specified dataId.
        """
        return self.notes.get(_idTrans(dataId), None)

    def hasTags(self, dataId, flags):
        """Check if a dataId has all the specificed tags"""
        tag = self.getTags(dataId)
        if tag is None:  # not just 'if tag' becuase '' is not the same as None but both as False-y
            return None
        return all(i in tag for i in flags.upper())

    def getListOfCheckedData(self):
        """Return a list of all dataIds which have been examined.
        """
        return sorted(list(self.tags.keys()))

    def getListOfDataWithNotes(self):
        """Return a list of all dataIds which have notes associated.
        """
        return sorted(list(self.notes.keys()))

    def isExamined(self, dataId):
        """Check if the dataId has been examined or not.
        """
        return _idTrans(dataId) in self.tags

    def printTags(self):
        """Display the list of tag definitions.
        """
        print(TAGS)

    def getIdsWithGivenTags(self, tags, exactMatches=False):
        if exactMatches:
            return [dId for (dId, tag) in self.tags.items() if (all(t in tag for t in tags.upper()) and
                                                                (len(tags) == len(tag)))]
        else:
            return [dId for (dId, tag) in self.tags.items() if all(t in tag for t in tags.upper())]
