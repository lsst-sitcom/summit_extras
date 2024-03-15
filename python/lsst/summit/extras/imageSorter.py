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

import os
import pickle
import re
from os import system
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from PIL import Image

if TYPE_CHECKING:
    from typing import List, Tuple


TAGS = """
  - (Blank/no annotation) - nominally good, i.e. nothing notable in the image
Q - bad main star location (denoted by cross-hair on image sorter)
F - Obviously very poor focus (worse than just seeing, does NOT include donuts)
D - Donut image
O - Occlusion (dome or door)
V - No back bias suspected
P - Non-standard PSF (rotator/mount issues/tracking error, etc)
S - Satellite or plane crossing image
! - Something interesting/crazy - see notes on image
"""

INSTRUCTIONS = (
    TAGS
    + "\n"
    + """
                = - apply the same annotations as the previous image
                To enter no tags but some notes, just start with a space
                """
)


class ImageSorter:
    """Take a list on png files, as created by lsst.summit.extras.animator
    and tag each dataId with a number of attributes.

    Returns a dict of dataId dictionaries with values being the corresponding
    """

    def __init__(self, fileList: "List[str]", outputFilename: str):
        self.fileList = fileList
        self.outputFilename = outputFilename

    @staticmethod
    def _getDataIdFromFilename(filename: str) -> "Tuple[str, int]":
        # filename of the form 2021-02-18-705-quickLookExp.png
        filename = os.path.basename(filename)
        mat = re.match(r"^(\d{4}-\d{2}-\d{2})-(\d*)-.*$", filename)
        if not mat:
            raise RuntimeError(f"Failed to extract dayObs/seqNum from {filename}")
        dayObs = mat.group(1)  # type: str
        seqNum = int(mat.group(2))  # type: int
        return (dayObs, seqNum)

    def getPreviousAnnotation(self, info: "List[str]", imNum: int) -> str:
        if imNum == 0:
            raise RuntimeError("There is no previous annotation for the first image.")

        previousFilename = self.fileList[imNum - 1]
        previousDataId = self._getDataIdFromFilename(previousFilename)
        previousAnnotation = info[previousDataId]
        return previousAnnotation

    def addData(self, dataId, info, answer: str, mode: str, imNum: int) -> None:
        """Modes = O(verwrite), S(kip), A(ppend)"""
        if "=" in answer:
            answer = self.getPreviousAnnotation(info, imNum)

        if dataId not in info:
            info[dataId] = answer
            return

        if mode == "O":
            info[dataId] = answer
        elif mode in ["B", "A"]:
            oldAnswer = info[dataId]
            answer = "".join([oldAnswer, answer])
            info[dataId] = answer
        else:
            raise RuntimeError(f"Unrecognised mode {mode} - should be impossible")
        return

    @classmethod
    def loadAnnotations(cls, pickleFilename: str) -> "Tuple[dict, dict]":
        """Load back and split up annotations for easy use.

        Anything after a space is returned as a whole string,
        anything before it is lower-cased and returned as tags.

        from lsst.summit.extras import ImageSorter
        tags, notes = ImageSorter.loadAnnotations(pickleFilename)
        """
        loaded = cls._load(pickleFilename)

        tags, notes = {}, {}

        for dataId, answerFull in loaded.items():
            answer = answerFull.lower()
            if answerFull.startswith(" "):  # notes only case
                tags[dataId] = ""
                notes[dataId] = answerFull.strip()
                continue

            if " " in answer:
                answer = answerFull.split()[0]
                notes[dataId] = " ".join([_ for _ in answerFull.split()[1:]])
            tags[dataId] = answer.upper()

        return tags, notes

    @staticmethod
    def _load(filename: str):
        """Internal loading only.

        Not to be used by users for reading back annotations"""
        with open(filename, "rb") as pickleFile:
            info = pickle.load(pickleFile)
        return info

    @staticmethod
    def _save(info, filename: str) -> None:
        with open(filename, "wb") as dumpFile:
            pickle.dump(info, dumpFile)

    def sortImages(self) -> dict:
        mode = "A"
        info = {}
        if os.path.exists(self.outputFilename):
            info = self._load(self.outputFilename)

            print(f"Output file {self.outputFilename} exists with info on {len(info)} files:")
            print("Press A - view all images, appending info to existing entries")
            print("Press O - view all images, overwriting existing entries")
            print("Press S - skip all images with existing annotations, including blank annotations")
            print("Press B - skip all images with annotations that are not blank")
            print("Press D - just display existing data and exit")
            print("Press Q to quit")
            mode = input()
            mode = mode[0].upper()

            if mode == "Q":
                exit()
            elif mode == "D":
                for dataId, value in info.items():
                    print(f"{dataId[0]} - {dataId[1]}: {value}")
                exit()
            elif mode in "AOSB":
                pass
            else:
                print("Unrecognised response - try again")
                self.sortImages()
                return None  # don't run twice in this case!

        # need to write file first, even if empty, because _load and _save
        # are inside the loop to ensure that annotations aren't lost even on
        # full crash
        print(INSTRUCTIONS)
        self._save(info, self.outputFilename)

        plt.figure(figsize=(10, 10))
        for imNum, filename in enumerate(self.fileList):
            info = self._load(self.outputFilename)

            dataId = self._getDataIdFromFilename(filename)
            if dataId in info and mode in ["S", "B"]:  # always skip if found for S and if not blank for B
                if (mode == "S") or (mode == "B" and info[dataId] != ""):
                    continue

            with Image.open(filename) as pilImage:
                pilImage = Image.open(filename)
                width, height = pilImage.size
                cropLR, cropUD = 100 - 50, 180 - 50
                cropped = pilImage.crop((cropLR, cropUD, width - cropLR, height - cropUD))
                plt.clf()
                plt.imshow(cropped, interpolation="bicubic")
                plt.show(block=False)
                plt.draw()  # without this you get the same image each time
                plt.tight_layout()
                osascriptCall = """/usr/bin/osascript -e 'tell app "Finder" to """
                osascriptCall += """set frontmost of process "Terminal" to true' """
                system(osascriptCall)

            oldAnswer = None  # just so we can display existing info with the dataId
            if dataId in info:
                oldAnswer = info[dataId]
            inputStr = f"{dataId[0]} - {dataId[1]}: %s" % ("" if oldAnswer is None else oldAnswer)
            answer = input(inputStr)
            if "exit" in answer:
                break  # break don't exit so data is written!

            self.addData(dataId, info, answer, mode, imNum)
            self._save(info, self.outputFilename)

        print(f"Info written to {self.outputFilename}")

        return info


if __name__ == "__main__":
    # TODO: DM-34239 Remove this
    fileList = [
        "/Users/merlin/rsync/animatorOutput/pngs/dayObs-2020-02-17-seqNum-232-calexp.png",
        "/Users/merlin/rsync/animatorOutput/pngs/dayObs-2020-02-17-seqNum-233-calexp.png",
        "/Users/merlin/rsync/animatorOutput/pngs/dayObs-2020-02-17-seqNum-234-calexp.png",
        "/Users/merlin/rsync/animatorOutput/pngs/dayObs-2020-02-17-seqNum-235-calexp.png",
    ]

    sorter = ImageSorter(fileList, "/Users/merlin/scratchfile.txt")
    sorter.sortImages()
