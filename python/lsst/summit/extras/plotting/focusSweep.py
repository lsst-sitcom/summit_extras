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

__all__ = [
    "collectSweepData",
    "inferHexapodSweepAxis",
    "fitSweepParabola",
    "plotSweepParabola",
]


import numpy as np
from matplotlib.figure import Figure

from lsst.daf.butler import DimensionRecord
from lsst.summit.utils.efdUtils import efdTimestampToAstropy, getMostRecentRowWithDataBefore

PLATESCALE = 0.2  # arcsec / pixel


def collectSweepData(records, consDbClient, efdClient):
    """
    Populate focus sweep table.

    Parameters
    ----------
    records : list of `DimensionRecord`
        Records spanning focus sweep.
    consDbClient: `ConsDbClient`
        Consolidated database client
    efdClient: `EfdClient`
        Engineering facilities database client

    Returns
    -------
    table : astropy.table.Table
        Table containing hexapod sweep motions and quick look PSF measurements.
    """
    visitString = ",".join(str(r.id) for r in records)
    data = consDbClient.query(
        "SELECT "
        "visit_id as visit_id, "
        "n_inputs as n_inputs, "
        "psf_sigma_median as sigma, "
        "psf_ixx_median as ixx, "
        "psf_iyy_median as iyy, "
        "psf_ixy_median as ixy "
        f"from cdb_lsstcomcamsim.visit1_quicklook WHERE visit_id in ({visitString}) "
        "ORDER BY visit_id"
    )
    data["seqNum"] = data["visit_id"] % 10000
    data["T"] = data["ixx"] + data["iyy"]
    data["e1"] = (data["ixx"] - data["iyy"]) / data["T"]
    data["e2"] = 2 * data["ixy"] / data["T"]
    data["fwhm"] = np.sqrt(np.log(256)) * PLATESCALE * data["sigma"]

    # Placeholder columns for EFD data
    for prefix in ["cam_", "m2_"]:
        for k in ["x", "y", "z", "u", "v", "age"]:
            data[prefix + k] = np.nan

    for row in data:
        rIdx = [r.id for r in records].index(row["visit_id"])
        record = records[rIdx]
        for pid, prefix in zip(["MTHexapod:1", "MTHexapod:2"], ["cam_", "m2_"]):
            try:
                efdData = getMostRecentRowWithDataBefore(
                    efdClient,
                    "lsst.sal.MTHexapod.logevent_compensatedPosition",
                    timeToLookBefore=record.timespan.begin,
                    maxSearchNMinutes=3,
                    where=lambda df: df["private_identity"] == pid,
                )
            except ValueError:
                row[prefix + "age"] = np.nan
                for k in ["x", "y", "z", "u", "v"]:
                    row[prefix + k] = np.nan
            else:
                age = record.timespan.begin - efdTimestampToAstropy(efdData["private_efdStamp"])
                row[prefix + "age"] = age.sec
                for k in ["x", "y", "z", "u", "v"]:
                    row[prefix + k] = efdData[k]
    return data


def inferSweepVariable(data):
    """
    Heuristically determine which variable is being swept during a focus sweep.

    Parameters
    ----------
    data : astropy.table.Table
        Table holding sweep hexapod motions and PSF measurements.

    Returns
    -------
    varName : str | None
        Name of the inferred active hexapod variable or None if inference failed.
    """
    # Examine the ratio of RMS hexapod values to RMS hexapod residuals from a linear fit
    # against seqNum.  If removing the linear term significantly reduces the RMS, that's
    # a good sign this is the active variable.
    stats = {}
    for prefix in ["cam_", "m2_"]:
        for k in ["x", "y", "z", "u", "v"]:
            hexapodValue = data[prefix + k].value.astype(float)
            seqNum = data["seqNum"]
            coefs = np.polyfit(seqNum, hexapodValue, 1)
            resids = np.polyval(coefs, seqNum) - hexapodValue
            stdResids = np.nanstd(resids)
            if stdResids == 0:
                stats[prefix + k] = np.nan
            else:
                stats[prefix + k] = np.nanstd(hexapodValue) / stdResids
    statMax = -np.inf
    varName = None
    for vName, stat in stats.items():
        if stat > statMax:
            varName = vName
            statMax = stat
    return varName


def fitSweepParabola(data, varName):
    fwhms = data["fwhm"]
    e1s = data["e1"]
    e2s = data["e2"]
    xs = data[varName]
    coefs, cov = np.polyfit(xs, fwhms, 2, cov=True)
    a, b, c = coefs
    vertex = -b / (2 * a)
    resids = np.polyval(coefs, xs) - fwhms
    rms = np.sqrt(np.mean(np.square(resids)))
    extremum = np.polyval(coefs, vertex)

    # WARNING!  Trusting ChatGPT with vertex uncertainty
    # propagation for the moment.  Treat with extreme caution!
    da = np.sqrt(cov[0, 0])
    db = np.sqrt(cov[1, 1])
    covAB = cov[0, 1]

    # Uncertainty propagation to the vertex x-coordinate
    vertexUncertainty = np.sqrt((db / (2 * a)) ** 2 + (b * da / (2 * a**2)) ** 2 - (b / (2 * a**2)) * (covAB / a))
    extremumUncertainty = np.sqrt((2 * vertex * da) ** 2 + db**2 + 4 * vertex * covAB)

    e1Rms = np.sqrt(np.mean(np.square(e1s)))
    e2Rms = np.sqrt(np.mean(np.square(e2s)))

    return dict(
        vertex=vertex,
        extremum=extremum,
        rms=rms,
        vertexUncertainty=vertexUncertainty,
        extremumUncertainty=extremumUncertainty,
        e1Rms=e1Rms,
        e2Rms=e2Rms,
        coefs=coefs,
    )


def plotSweepParabola(data, varName, fitDict, filename=None, figAxes=None):
    xs = data[varName]

    if figAxes is None:
        fig = Figure(figsize=(8, 6))
        axes = fig.subplots(nrows=2, ncols=2)
    else:
        fig, axes = figAxes
    fwhmAx, _, e1Ax, e2Ax = axes.ravel()
    fig.delaxes(axes[0, 1])

    e1Ax.scatter(xs, data["e1"], c="r")
    e1Ax.axhline(0, c="k")
    e1Ax.set_ylim(-0.11, 0.11)
    e1Ax.set_ylabel("e1")

    e2Ax.scatter(xs, data["e2"], c="r")
    e2Ax.axhline(0, c="k")
    e2Ax.set_ylim(-0.11, 0.11)
    e2Ax.set_ylabel("e2")

    fwhmAx.scatter(xs, data["fwhm"], c="r")
    xlim = fwhmAx.get_xlim()
    xs = np.linspace(xlim[0], xlim[1], 100)
    ys = np.polyval(fitDict["coefs"], xs)
    fwhmAx.plot(xs, ys, c="k")
    fwhmAx.set_ylabel("fwhm [arcsec]")

    label = varName.replace("_", " ")
    label = label.replace("u", "Rx")
    label = label.replace("v", "Ry")
    unit = "deg" if "r" in label else "mm"

    for ax in axes.ravel():
        labeltext = label
        if unit is not None:
            labeltext = labeltext + " [" + unit + "]"
        ax.set_xlabel(labeltext)

    # Print useful info in the top right
    kwargs = dict(fontsize=10, ha="left", fontfamily="monospace")
    fig.text(0.6, 0.94, "FWHM fit", **kwargs)
    fig.text(0.6, 0.91, "--------", **kwargs)
    fig.text(0.6, 0.88, f"vertex: {fitDict['vertex']:.3f} ± {fitDict['vertexUncertainty']:.3f} {unit}", **kwargs)
    fig.text(0.6, 0.85, f"extremum: {fitDict['extremum']:.3f} ± {fitDict['extremumUncertainty']:.3f} arcsec", **kwargs)
    fig.text(0.6, 0.82, f"RMS resid: {fitDict['rms']:.3f} arcsec", **kwargs)

    fig.text(0.6, 0.70, f"Ellipticity spread", **kwargs)
    fig.text(0.6, 0.67, f"------------------", **kwargs)
    fig.text(0.6, 0.64, f"e1 RMS: {fitDict['e1Rms']:.3f}", **kwargs)
    fig.text(0.6, 0.61, f"e2 RMS: {fitDict['e2Rms']:.3f}", **kwargs)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
