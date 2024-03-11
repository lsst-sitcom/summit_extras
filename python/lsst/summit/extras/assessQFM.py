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

import argparse
import time
from multiprocessing import Pool
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask, QuickFrameMeasurementTaskConfig

if TYPE_CHECKING:
    import lsst.daf.butler as dafButler


class AssessQFM:
    """Test a new version of quickFrameMeasurementTask against the baseline
    results.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler repository with the relevant exposures.
    dataProduct : `str`, optional
        Data product on which to run quickFrameMeasurement.
    dataset : `str`, optional
        File holding a table of vetted quickFrameMeasurement results.
    successCut : `float`, optional
        Distance in pixels between the baseline and new measurement centroids
        for already successful fits in order to consider the new fit equally
        successful.
    nearSuccessCut : `float`, optional
        Distance in pixels between the baseline and new measurement centroids
        for fits that were close to correct in order to consider the new fit
        approximately as successful.
    donutCut : `float`, optional
        Distance in pixels between the baseline and new measurement centroids
        for fits of donut images to consider the new fit approximately as
        successful.
    logLevel : `int`, optional
        Level of QuickFrameMeasurementTask log messages. Setting to 50 means
        that only CRITICAL messages will be printed.
    """

    def __init__(
        self,
        butler: dafButler.Butler,
        dataProduct: str = "quickLookExp",
        dataset: str = "data/qfm_baseline_assessment.parq",
        successCut: int = 2,
        nearSuccessCut: int = 10,
        donutCut: int = 10,
        logLevel: int = 50,
    ):
        self.butler = butler

        qfmTaskConfig = QuickFrameMeasurementTaskConfig()
        self.qfmTask = QuickFrameMeasurementTask(config=qfmTaskConfig)
        self.qfmTask.log.setLevel(logLevel)

        self.testData = pd.read_parquet(dataset)
        self.dataProduct = dataProduct
        self.dataIds = [
            {"day_obs": row["day_obs"], "seq_num": row["sequence_number"], "detector": row["detector"]}
            for i, row in self.testData.iterrows()
        ]

        self.cuts = {"G": successCut, "QG": nearSuccessCut, "DG": donutCut}

        self.resultKey = {
            "G": "Success",  # Centroid is centered on the brightest star
            "QG": "Near success",  # Centroid is near the center of the brightest star
            "BI": "Bad image",  # A tracking issue, for example. Don't expect good fit
            "WF": "Wrong star",  # Centroid is not on the brightest star
            "OF": "Other failure",  # Other source of failure
            "FG": "Good failure",  # Calibration image, so failure is expected
            "FP": "False positive",  # No stars, so fit should have failed
            "DG": "Success (Donut)",  # Donut image, centroid is somewhere on donut
            "DF": "Failure (Donut)",  # Donut image, fit failed
            "SG": "Success (Giant donut)",  # Giant donut, centroid is somewhere on donut
            "SF": "Failure (Giant donut)",  # Giant donut, fit failed
            "U": "Ambiguous",  # Centroid is on a star, but unclear whether it is the brightest
        }

    def run(self, nSamples: int = None, nProcesses: int = 1, outputFile: str = None) -> None:
        """Run quickFrameMeasurement on a sample dataset, save the new results,
        and compare them with the baseline, vetted by-eye results.

        Parameters
        ----------
        nSamples : `int`
            Number of exposures to check. If nSamples is greater than the
            number of exposures in the vetted dataset, will check all.
        nProcesses : `int`
            Number of threads to use. If greater than one, multithreading will
            be used.
        outputFile : `str`
            Name of the output file.
        """

        if nSamples is not None:
            if nSamples > len(self.dataIds):
                nSamples = len(self.dataIds)
            samples = np.random.choice(range(len(self.dataIds)), size=nSamples, replace=False)
            testSubset = self.testData.iloc[samples]
        else:
            testSubset = self.testData

        if nProcesses > 1:
            with Pool(processes=nProcesses) as p:
                df_split = np.array_split(testSubset, nProcesses)
                pool_process = p.map(self._runQFM, df_split)
                qfmResults = pd.concat(pool_process)
        else:
            qfmResults = self._runQFM(testSubset)

        if outputFile:
            qfmResults.to_parquet(outputFile)

        self.compareToBaseline(qfmResults)

    def _runQFM(self, testset: pd.DataFrame) -> pd.DataFrame:
        """Run quickFrameMeasurement on a subset of the dataset.

        Parameters
        ----------
        testset : `pandas.DataFrame`
            Table of vetted exposures.

        Returns
        -------
        qfmResults : `pandas.DataFrame`
            Table of results from new quickFrameMeasurement run.
        """

        qfmResults = pd.DataFrame(index=testset.index, columns=self.testData.columns)
        for i, row in testset.iterrows():
            dataId = {
                "day_obs": row["day_obs"],
                "seq_num": row["sequence_number"],
                "detector": row["detector"],
            }

            exp = self.butler.get(self.dataProduct, dataId=dataId)
            qfmRes = qfmResults.loc[i]

            t1 = time.time()
            result = self.qfmTask.run(exp)
            t2 = time.time()
            qfmRes["runtime"] = t2 - t1

            if result.success:
                pixCoord = result.brightestObjCentroid
                qfmRes["centroid_x"] = pixCoord[0]
                qfmRes["centroid_y"] = pixCoord[1]
                qfmRes["finalTag"] = "P"

            else:
                qfmRes["finalTag"] = "F"
        return qfmResults

    def compareToBaseline(self, comparisonData: pd.DataFrame):
        """Compare a table of quickFrameMeasurement results with the
        baseline vetted data, and print output of the comparison.

        Parameters
        ----------
        comparisonData : `pandas.DataFrame`
            Table to compare with baseline results.
        """
        baselineData = self.testData.loc[comparisonData.index]

        # First the cases that succeeded in the baseline results:
        for key in ["G", "QG", "WF", "DG", "SG", "FP", "U"]:
            key_inds = baselineData["finalTag"] == key
            if key_inds.sum() == 0:
                continue
            origResults = baselineData[key_inds]
            newResults = comparisonData[key_inds]

            stillSucceeds = (newResults["finalTag"] == "P").sum()
            print(f"Results for '{self.resultKey[key]}' cases:")
            print(f"    {stillSucceeds} out of {len(origResults)} still succeed")

            centroid_distances = (
                (origResults["centroid_x"] - newResults["centroid_x"]) ** 2
                + (origResults["centroid_y"] - newResults["centroid_y"]) ** 2
            ) ** 0.5

            if key in ["G", "QG", "DG"]:
                inCut = centroid_distances < self.cuts[key]
                print(
                    f"    {inCut.sum()} out of {len(origResults)} centroids are within {self.cuts[key]} "
                    "pixels of the baseline centroid fit."
                )
            if key in ["U", "WF", "QG"]:
                print("    Individual exposures:")
                print(f"    {'day_obs':<10}{'sequence_number':<17}{'old centroid':<17}{'new centroid':<17}")
                for i, res in origResults.iterrows():
                    newRes = newResults.loc[i]
                    old_centroid = f"({res['centroid_x']:.1f}, {res['centroid_y']:.1f})"
                    new_centroid = f"({newRes['centroid_x']:.1f}, {newRes['centroid_y']:.1f})"
                    print(
                        f"    {res['day_obs']:<10}{res['sequence_number']:<17}{old_centroid:<17}"
                        f"{new_centroid:<17}"
                    )

        # Next the cases that failed in the past:
        for key in ["FG", "DF", "SF", "OF"]:
            key_inds = baselineData["finalTag"] == key
            if key_inds.sum() == 0:
                continue
            origResults = baselineData[key_inds]
            newResults = comparisonData[key_inds]

            stillFails = (newResults["finalTag"] == "F").sum()
            print(f"Results for '{self.resultKey[key]}' cases:")
            print(f"    {stillFails} out of {len(origResults)} still fail")

        print("Runtime comparison:")
        print(
            f"    Baseline: {np.mean(baselineData['runtime']):.2f}+/-"
            f"{np.std(baselineData['runtime']):.2f} seconds"
        )
        print(
            f"    Current: {np.mean(comparisonData['runtime']):.2f}+/-"
            f"{np.std(comparisonData['runtime']):.2f} seconds"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embargo",
        dest="embargo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use embargo butler",
    )
    parser.add_argument(
        "--nPool", dest="nPool", default=1, type=int, help="Number of threads to use in multiprocessing"
    )
    parser.add_argument(
        "--nSamples",
        dest="nSamples",
        default=None,
        type=int,
        help="Number of sample exposures to use in assessment (default is all)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="outputFile",
        default="newQFMresults.parq",
        help="Name of output file for new quickFrameMeasurement results",
    )
    args = parser.parse_args()

    butler = butlerUtils.makeDefaultLatissButler(embargo=args.embargo)
    assess = AssessQFM(butler)
    nSamples = args.nSamples

    t0 = time.time()
    assess.run(nSamples=nSamples, nProcesses=args.nPool, outputFile=args.outputFile)
    t1 = time.time()
    if nSamples is None:
        nSamples = assess.testData.shape[0]
    print(f"Total time for {nSamples} samples and {args.nPool} cores: {(t1 - t0):.2f} seconds")
