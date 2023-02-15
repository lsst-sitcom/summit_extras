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
from typing import Iterable

import lsst.utils.tests

import matplotlib as mpl
mpl.use('Agg')

from lsst.summit.extras import SpectralFocusAnalyzer, NonSpectralFocusAnalyzer  # noqa: E402
import lsst.summit.utils.butlerUtils as butlerUtils  # noqa: E402


class FocusAnalysisTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.butler = butlerUtils.makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        cls.dayObs = 20220628
        cls.seqNums = range(280, 288+1)
        cls.focusAnalyzer = SpectralFocusAnalyzer()

        # default is 3 boxes, so setting four tests the generality of the code.
        # The values have been picked to land on the pretty small spectrum we
        # have in the test data.
        cls.focusAnalyzer.setSpectrumBoxOffsets([1600, 1700, 1800, 1900])

    def test_run(self):
        # we don't check the plots, but set doDisplay to True to check the
        # plots are generated without error
        self.focusAnalyzer.getFocusData(self.dayObs, self.seqNums, doDisplay=True)
        result = self.focusAnalyzer.fitDataAndPlot()
        self.assertIsInstance(result, Iterable)
        self.assertEqual(len(result), len(self.focusAnalyzer.getSpectrumBoxOffsets()))

        for number in result:
            # check they're all numbers, non-nan, and vaguely sensible
            self.assertGreater(number, -1.0)
            self.assertLess(number, 1.0)


class NonSpectralFocusAnalysisTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.butler = butlerUtils.makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        cls.dayObs = 20220405
        cls.seqNums = range(523, 531+1)
        cls.focusAnalyzer = NonSpectralFocusAnalyzer()

    def test_run(self):
        # we don't check the plots, but set doDisplay to True to check the
        # plots are generated without error
        self.focusAnalyzer.getFocusData(self.dayObs, self.seqNums, doDisplay=True)
        result = self.focusAnalyzer.fitDataAndPlot()
        self.assertIsInstance(result, dict)

        # result is a dict which looks like this:
        # {'fwhmFitMin': 0.029221417454391708,
        #  'ee90FitMin': 0.0754762884665358,
        #  'ee80FitMin': 0.07188778363981346,
        #  'ee50FitMin': 0.13998855716378267}
        self.assertEqual(len(result), 4)

        for k, v in result.items():
            # check they're all numbers, non-nan, and vaguely sensible
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, float)

            self.assertGreater(v, 0.0)
            self.assertLess(v, 1.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
