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
import os
import shutil

import lsst.utils.tests

from lsst.summit.extras.animation import Animator
from lsst.summit.utils.butlerUtils import makeDefaultLatissButler


class AnimationTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        # test for the existence of ffmpeg and skip test if not found
        if shutil.which('ffmpeg') is None:
            raise unittest.SkipTest("Skipping tests that require ffmpeg.")

        try:
            cls.butler = makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        cls.dataIds = [{'day_obs': 20200315, 'seq_num': 120, 'detector': 0},
                       {'day_obs': 20200315, 'seq_num': 121, 'detector': 0}]
        cls.outputDir = tempfile.mkdtemp()
        cls.outputFilename = os.path.join(cls.outputDir, 'testAnimation.mp4')

    def test_animation(self):
        """Test that the animator produces a large file without worrying about
        the contents?
        """
        animator = Animator(self.butler, self.dataIds, self.outputDir, self.outputFilename,
                            dataProductToPlot='raw',
                            remakePngs=True,
                            debug=False,
                            clobberVideoAndGif=True,
                            plotObjectCentroids=True,
                            useQfmForCentroids=True)
        writtenFilename = animator.run()

        self.assertTrue(writtenFilename == self.outputFilename)
        self.assertTrue(os.path.isfile(self.outputFilename))
        self.assertTrue(os.path.getsize(self.outputFilename) > 10000)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
