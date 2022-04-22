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

import unittest
import tempfile
import pickle

import lsst.utils.tests
from lsst.summit.extras.annotations import Annotations, _idTrans


tagId1 = {'dayObs': '1970-01-01', 'seqNum': 1}  # A
tagId2 = {'dayObs': '1970-01-02', 'seqNum': 2}  # AB
tagId3 = {'dayObs': '1970-01-03', 'seqNum': 3}  # ABC
tagAndNotesId = {'dayObs': '1970-01-04', 'seqNum': 4}  # D tag with a note
notesOnlyId = {'dayObs': '1970-01-05', 'seqNum': 5}  # this is a note alone
checkedOnlyId = {'dayObs': '1970-01-06', 'seqNum': 6}  # ""
nonexistentId = {'dayObs': '3030-13-32', 'seqNum': 666}  # 6
testTuple = (tagId1['dayObs'], tagId1['seqNum'])
testTupleNonExistent = (nonexistentId['dayObs'], nonexistentId['seqNum'])

allGoodIdsInternalFormat = set([_idTrans(x) for x in [tagId1, tagId2, tagId3, notesOnlyId,
                                                      tagAndNotesId, checkedOnlyId]])


class AnnotationsTestCase(lsst.utils.tests.TestCase):
    """A test case for annotations."""

    def setUp(self):

        inputFile = tempfile.mktemp() + '.pickle'
        tags = {('1970-01-01', 1): 'a',
                ('1970-01-02', 2): 'aB',
                ('1970-01-03', 3): 'Abc',
                ('1970-01-04', 4): 'd tag with a note',
                ('1970-01-05', 5): ' this is a note alone',
                ('1970-01-06', 6): ''}
        with open(inputFile, "wb") as f:
            pickle.dump(tags, f)

        annotations = Annotations(inputFile)
        self.assertTrue(annotations is not None)
        self.annotations = annotations

    def test_isExamined(self):
        self.assertTrue(self.annotations.isExamined(tagId1))
        self.assertTrue(self.annotations.isExamined(tagId2))
        self.assertTrue(self.annotations.isExamined(tagId3))
        self.assertTrue(self.annotations.isExamined(tagAndNotesId))
        self.assertTrue(self.annotations.isExamined(notesOnlyId))
        self.assertTrue(self.annotations.isExamined(checkedOnlyId))

        self.assertTrue(self.annotations.isExamined(testTuple))

        self.assertFalse(self.annotations.isExamined(nonexistentId))
        self.assertFalse(self.annotations.isExamined(testTupleNonExistent))

    def test_getTags(self):
        self.assertTrue(self.annotations.getTags(tagId1) == 'A')
        self.assertTrue(self.annotations.getTags(tagId2) == 'AB')
        self.assertTrue(self.annotations.getTags(tagId3) == 'ABC')
        self.assertTrue(self.annotations.getTags(tagAndNotesId) == 'D')
        self.assertTrue(self.annotations.getTags(notesOnlyId) == '')  # examined but no tags so not None
        self.assertTrue(self.annotations.getTags(nonexistentId) is None)  # not examined so is None

    def test_getNotes(self):
        self.assertTrue(self.annotations.getNotes(tagId1) is None)
        self.assertTrue(self.annotations.getNotes(tagId2) is None)
        self.assertTrue(self.annotations.getNotes(tagId3) is None)
        self.assertTrue(self.annotations.getNotes(tagAndNotesId) == 'tag with a note')
        self.assertTrue(self.annotations.getNotes(notesOnlyId) == 'this is a note alone')

    def test_hasTags(self):
        self.assertTrue(self.annotations.hasTags(tagId1, 'a'))
        self.assertTrue(self.annotations.hasTags(tagId1, 'A'))  # case insensitive on tags
        self.assertTrue(self.annotations.hasTags(tagId2, 'AB'))  # multicharacter
        self.assertTrue(self.annotations.hasTags(tagId2, 'Ab'))  # mixed case
        self.assertFalse(self.annotations.hasTags(tagId1, 'B+'))  # false multichar

        self.assertTrue(self.annotations.hasTags(tagAndNotesId, 'd'))
        self.assertFalse(self.annotations.hasTags(notesOnlyId, 'a'))
        self.assertTrue(self.annotations.hasTags(notesOnlyId, ''))
        # XXX fix or remove after other tests

    def test_getListOfCheckedData(self):
        correctIds = set([_idTrans(x) for x in [tagId1, tagId2, tagId3, notesOnlyId,
                                                tagAndNotesId, checkedOnlyId]])
        ids = self.annotations.getListOfCheckedData()
        self.assertTrue(correctIds == set(ids))

    def test_getListOfDataWithNotes(self):
        correctIds = set([_idTrans(x) for x in [notesOnlyId, tagAndNotesId]])
        ids = self.annotations.getListOfDataWithNotes()
        self.assertTrue(correctIds == set(ids))

    def test_getIdsWithGivenTags(self):
        allIds = allGoodIdsInternalFormat
        _t = _idTrans

        ids = self.annotations.getIdsWithGivenTags('', exactMatches=False)
        self.assertTrue(allIds == set(ids))
        ids = self.annotations.getIdsWithGivenTags('', exactMatches=True)
        self.assertTrue(allIds != set(ids))

        ids = self.annotations.getIdsWithGivenTags('b', exactMatches=False)
        correct = set([_t(dId) for dId in [tagId2, tagId3]])
        self.assertTrue(set(ids) == correct)

        ids = self.annotations.getIdsWithGivenTags('b', exactMatches=True)  # nothing only 'b' alone
        self.assertTrue(ids == [])

        ids = self.annotations.getIdsWithGivenTags('Ba', exactMatches=False)  # reversed so not a substring
        correct = set([_t(dId) for dId in [tagId2, tagId3]])
        self.assertTrue(set(ids) == correct)

        # check exact matches for multiple tags
        ids = self.annotations.getIdsWithGivenTags('ab', exactMatches=True)
        correct = set([_t(dId) for dId in [tagId2]])
        self.assertTrue(set(ids) == correct)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
