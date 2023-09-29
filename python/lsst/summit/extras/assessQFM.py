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

import pandas as pd
import time
import numpy as np

from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask, QuickFrameMeasurementTaskConfig
import lsst.summit.utils.butlerUtils as butlerUtils


class AssessQFM():
    """Test a new version of quickFrameMeasurementTask against the baseline
    results.

    Parameters
    ----------
    butler :
    qfmTask : 
        If None, then use the version that is set up.
    """

    def __init__(self, butler, qfmTask=None, dataProduct='quickLookExp',
                 dataset='~/stack_projects/quickFrameMeasurement/baseline_meas.parq',
                 successCut=2, nearSuccessCut=10, donutCut=10):

        self.butler = butler
        if qfmTask is not None:
            self.qfmTask = qfmTask
        else:
            qfmTaskConfig = QuickFrameMeasurementTaskConfig()
            self.qfmTask = QuickFrameMeasurementTask(config=qfmTaskConfig)
        self.dataProduct = dataProduct
        self.testData = pd.read_parquet(dataset)
        self.dataIds = [{'day_obs': row['day_obs'], 'seq_num': row['sequence_number'], 'detector': row['detector']}
                        for i, row in self.testData.iterrows()]
        
        self.cuts = {'G': successCut,
                     'QG': nearSuccessCut,
                     'DG': donutCut}
        
        self.resultKey = {'G': "Success", # Centroid is centered on the brightest star
                          'QG': "Near success", # Centroid is near the center of the brightest star
                          'BI': "Bad image", # A tracking issue, for example. Don't expect good fit
                          'WF': "Wrong star", # Centroid is not on the brightest star
                          'OF': "Other failure", # Other source of failure
                          'FG': "Good failure", # Calibration image, so failure is expected
                          'FP': "False positive", # No stars, so fit should have failed
                          'DG': "Success (Donut)", # Donut image, centroid is somewhere on donut
                          'DF': "Failure (Donut)", # Donut image, fit failed
                          'SG': "Success (Giant donut)", # Giant donut, centroid is somewhere on donut
                          'SF': "Failure (Giant donut)", # Giant donut, fit failed
                          'U': "Ambiguous", # Centroid is on a star, but unclear whether it is the brightest one
                          }

    def run(self, nSamples=None):

        if nSamples is not None:
            samples = np.random.choice(range(len(self.dataIds)), size=nSamples, replace=False)
            testSubset = self.testData.iloc[samples]
        else:
            testSubset = self.testData
        print(testSubset['finalTag'].unique())
        qfmResults = pd.DataFrame(index=testSubset.index, columns=self.testData.columns)
        for i, row in testSubset.iterrows():
            dataId = {'day_obs': row['day_obs'], 'seq_num': row['sequence_number'], 'detector': row['detector']}

            exp = self.butler.get(self.dataProduct, dataId=dataId)
            qfmRes = qfmResults.loc[i]

            t1 = time.time()
            result = self.qfmTask.run(exp)
            t2 = time.time()
            try:
                qfmRes['runtime'] = t2 - t1
            except:
                import ipdb; ipdb.set_trace()
            if result.success:
                pixCoord = result.brightestObjCentroid
                qfmRes['centroid_x'] = pixCoord[0]
                qfmRes['centroid_y'] = pixCoord[1]
                qfmRes['finalTag'] = 'P'

            else:
                qfmRes['finalTag'] = 'F'

        # Now compare with past results:
        for key in ['G', 'QG', 'WF', 'DG', 'SG', 'FP', 'U']:
            key_inds = testSubset['finalTag'] == key
            if key_inds.sum() == 0:
                continue
            origResults = testSubset[key_inds]
            newResults = qfmResults[key_inds]

            stillSucceeds = (newResults['finalTag'] == 'P').sum()
            print(f"Results for '{self.resultKey[key]}' cases:")
            print(f"    {stillSucceeds} out of {len(origResults)} still succeed")

            centroid_distances = ((origResults['centroid_x'] - newResults['centroid_x'])**2
                                  + (origResults['centroid_y'] - newResults['centroid_y'])**2)**0.5

            if key in ['G', 'QG', 'DG']:
                inCut = centroid_distances < self.cuts[key]
                print(f"    {inCut.sum()} out of {len(origResults)} centroids are within {self.cuts[key]} "
                      "pixels of the baseline centroid fit.")
            if key in ['U', 'WF', 'QG']:
                print("    Individual exposures:")
                print(f"    {'day_obs':<10}{'sequence_number':<17}{'old centroid':<17}{'new centroid':<17}")
                for i, res in origResults.iterrows():
                    newRes = newResults.loc[i]
                    old_centroid = f"({res['centroid_x']:.1f}, {res['centroid_y']:.1f})"
                    new_centroid = f"({newRes['centroid_x']:.1f}, {newRes['centroid_y']:.1f})"
                    print(f"    {res['day_obs']:<10}{res['sequence_number']:<17}{old_centroid:<17}{new_centroid:<17}")

        for key in ['FG', 'DF', 'SF', 'OF']:
            key_inds = testSubset['finalTag'] == key
            if key_inds.sum() == 0:
                continue
            origResults = testSubset[key_inds]
            newResults = qfmResults[key_inds]

            stillFails = (newResults['finalTag'] == 'F').sum()
            print(f"Results for '{self.resultKey[key]}' cases:")
            print(f"    {stillFails} out of {len(origResults)} still fail")

        print("Runtime comparison:")
        print(f"Baseline: {np.mean(testSubset['runtime']):.2f}+/-{np.std(testSubset['runtime']):.2f} seconds")
        print(f"Current: {np.mean(qfmResults['runtime']):.2f}+/-{np.std(qfmResults['runtime']):.2f} seconds")

if __name__ == '__main__':

    butler = butlerUtils.makeDefaultLatissButler(embargo=True)
    assess = AssessQFM(butler)

    assess.run(nSamples=200)

