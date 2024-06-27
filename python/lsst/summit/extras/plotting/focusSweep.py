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
    "fitSweepParabola",
    "plotSweepParabola",
]


import numpy as np
from matplotlib.figure import Figure

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
    """Heuristically determine which variable is being swept during a focus
    sweep.

    Parameters
    ----------
    data : astropy.table.Table
        Table holding sweep hexapod motions and PSF measurements.

    Returns
    -------
    varName : str | None
        Name of the inferred active hexapod variable or None if inference
        failed.
    """
    # Examine the ratio of RMS hexapod values to RMS hexapod residuals from a
    # linear fit against seqNum.  If removing the linear term significantly
    # reduces the RMS, that's a good sign this is the active variable.
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
    vertexUncertainty = np.sqrt(
        (db / (2 * a)) ** 2 + (b * da / (2 * a**2)) ** 2 - (b / (2 * a**2)) * (covAB / a)
    )
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


def plotSweepParabola(data, varName, fitDict, saveAs=None, figAxes=None):
    xs = data[varName]

    if figAxes is None:
        fig = Figure(figsize=(12, 9))
        axes = fig.subplots(nrows=3, ncols=4)
    else:
        fig, axes = figAxes

    camZAx, m2ZAx, *_ = axes[0]
    camXyAx, m2XyAx, fwhmSeqAx, fwhmVarAx = axes[1]
    camRAx, m2RAx, ellipSeqAx, ellipVarAx = axes[2]

    fig.delaxes(axes[0, 2])
    fig.delaxes(axes[0, 3])

    seqNum = data["seqNum"]
    camZAx.scatter(seqNum, data["cam_z"], c="r")
    camXyAx.scatter(seqNum, data["cam_x"], c="c", label="x")
    camXyAx.scatter(seqNum, data["cam_y"], c="m", label="y")
    camXyAx.legend()
    camRAx.scatter(seqNum, data["cam_u"], c="c", label="Rx")
    camRAx.scatter(seqNum, data["cam_v"], c="m", label="Ry")
    camRAx.legend()

    m2ZAx.scatter(seqNum, data["m2_z"], c="r")
    m2XyAx.scatter(seqNum, data["m2_x"], c="c", label="x")
    m2XyAx.scatter(seqNum, data["m2_y"], c="m", label="y")
    m2XyAx.legend()
    m2RAx.scatter(seqNum, data["m2_u"], c="c", label="Rx")
    m2RAx.scatter(seqNum, data["m2_v"], c="m", label="Ry")
    m2RAx.legend()

    fwhmSeqAx.scatter(seqNum, data["fwhm"], c="r")
    ellipSeqAx.scatter(seqNum, data["e1"], c="c", label="e1")
    ellipSeqAx.scatter(seqNum, data["e2"], c="m", label="e2")
    ellipSeqAx.legend()

    var = data[varName]
    fwhmVarAx.scatter(var, data["fwhm"], c="r")
    xlim = fwhmVarAx.get_xlim()
    xs = np.linspace(xlim[0], xlim[1], 100)
    ys = np.polyval(fitDict["coefs"], xs)
    fwhmVarAx.plot(xs, ys, c="k")

    ellipVarAx.scatter(var, data["e1"], c="c", label="e1")
    ellipVarAx.scatter(var, data["e2"], c="m", label="e2")
    ellipVarAx.legend()

    label = varName.replace("_", " ")
    label = label.replace("u", "Rx")
    label = label.replace("v", "Ry")
    unit = "deg" if "r" in label else "mm"

    for ax in [fwhmVarAx, fwhmSeqAx]:
        ax.set_ylabel("fwhm [arcsec]")

    for ax in [camZAx, m2ZAx]:
        ax.set_ylabel("z [mm]")
    camZAx.set_title("Camera")
    m2ZAx.set_title("M2")

    for ax in [camXyAx, m2XyAx]:
        ax.set_ylabel("x or y [mm]")

    for ax in [camRAx, m2RAx]:
        ax.set_ylabel("Rx or Ry [deg]")

    for ax in [camZAx, camXyAx, camRAx, m2ZAx, m2XyAx, m2RAx, fwhmSeqAx, ellipSeqAx]:
        ax.set_xlabel("seqnum")
        ax.set_xlim(min(seqNum) - 0.5, max(seqNum) + 0.5)

    for ax in [fwhmVarAx, ellipVarAx]:
        ax.set_xlabel(label + "[" + unit + "]")

    for ax in [ellipSeqAx, ellipVarAx]:
        ax.set_ylim(-0.11, 0.11)
        ax.axhline(0, c="k")
        ax.set_ylabel("e1 or e2")

    # Print useful info in the top right
    kwargs = dict(fontsize=10, ha="left", fontfamily="monospace")
    fig.text(0.7, 0.94, "FWHM fit", **kwargs)
    fig.text(0.7, 0.92, "--------", **kwargs)
    fig.text(
        0.7, 0.90, f"vertex: {fitDict['vertex']:.3f} ± {fitDict['vertexUncertainty']:.3f} {unit}", **kwargs
    )
    fig.text(
        0.7,
        0.88,
        f"extremum: {fitDict['extremum']:.3f} ± {fitDict['extremumUncertainty']:.3f} arcsec",
        **kwargs,
    )
    fig.text(0.7, 0.86, f"RMS resid: {fitDict['rms']:.3f} arcsec", **kwargs)

    fig.text(0.7, 0.80, "Ellipticity spread", **kwargs)
    fig.text(0.7, 0.78, "------------------", **kwargs)
    fig.text(0.7, 0.76, f"e1 RMS: {fitDict['e1Rms']:.3f}", **kwargs)
    fig.text(0.7, 0.74, f"e2 RMS: {fitDict['e2Rms']:.3f}", **kwargs)

    fig.tight_layout()
    if saveAs is not None:
        fig.savefig(saveAs)
