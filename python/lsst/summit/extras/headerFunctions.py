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

import filecmp
import hashlib
import logging
import os
import pickle
import sys

import astropy
import numpy as np
from astropy.io import fits

# redirect logger to stdout so that logger messages appear in notebooks too
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("headerFunctions")


def loadHeaderDictsFromLibrary(libraryFilename):
    """Load the header and hash dicts from a pickle file.

    Parameters
    ----------
    libraryFilename : `str`
        Path of the library file to load from

    Returns
    -------
    headersDict : `dict`
        A dict, keyed by filename, with the values being the full primary
    header, exactly as if it were built by buildHashAndHeaderDicts().

    dataDict : `dict`
        A dict, keyed by filename, with the values being hashes of the data
    sections, exactly as if it were built by buildHashAndHeaderDicts().
    """
    try:
        with open(libraryFilename, "rb") as pickleFile:
            headersDict, dataDict = pickle.load(pickleFile)

        if len(headersDict) != len(dataDict):
            print("Loaded differing numbers of entries for the header and data dicts.")
            print(f"{len(headersDict)} vs {len(dataDict)}")
            print("Something has gone badly wrong - your library seems corrupted!")
        else:
            print(f"Loaded {len(headersDict)} values from pickle files")
    except Exception as e:
        if not os.path.exists(libraryFilename):
            print(
                f"{libraryFilename} not found. If building the header dicts for the first time this"
                " is to be expected.\nOtherwise you've misspecified the path to you library!"
            )
        else:
            print(
                f"Something more sinister went wrong loading headers from {libraryFilename}:\n{e}"
            )
        return {}, {}

    return headersDict, dataDict


def _saveToLibrary(libraryFilename, headersDict, dataDict):
    try:
        with open(libraryFilename, "wb") as dumpFile:
            pickle.dump((headersDict, dataDict), dumpFile, pickle.HIGHEST_PROTOCOL)
    except Exception:
        print(
            "Failed to write pickle file! Here's a debugger so you don't lose all your work:"
        )
        import ipdb as pdb

        pdb.set_trace()


def _findKeyForValue(dictionary, value, warnOnCollision=True, returnCollisions=False):
    listOfKeys = [k for (k, v) in dictionary.items() if v == value]
    if warnOnCollision and len(listOfKeys) != 1:
        logger.warning("Found multiple keys for value! Returning only first.")
    if returnCollisions:
        return listOfKeys
    return listOfKeys[0]


def _hashFile(fileToHash, dataHdu, sliceToUse):
    """Put in place so that if hashing multiple HDUs is desired when one
    is filled with zeros it will be easy to add"""
    data = fileToHash[dataHdu].data[sliceToUse, sliceToUse].tostring()
    h = _hashData(data)
    return h


def _hashData(data):
    h = hashlib.sha256(data).hexdigest()  # hex because we want it readable in the dict
    return h


ZERO_HASH = _hashData(np.zeros((100, 100), dtype=np.int32))


def buildHashAndHeaderDicts(fileList, dataHdu="Segment00", libraryLocation=None):
    """For a list of files, build dicts of hashed data and headers.

    Data is hashed using a currently-hard-coded 100x100 region of the pixels
    i.e. file[dataHdu].data[0:100, 0:100]

    Parameters
    ----------
    fileList : `list` of `str`
        The fully-specified paths of the files to scrape

    dataHdu : `str` or `int`
        The HDU to use for the pixel data to hash.

    Returns
    -------
    headersDict : `dict`
        A dict, keyed by filename, with the values being the full primary
    header.

    dataDict : `dict`
        A dict, keyed by filename, with the values being hashes of the file's
    data section, as defined by the dataSize and dataHdu.

    """
    headersDict = {}
    dataDict = {}

    if libraryLocation:
        headersDict, dataDict = loadHeaderDictsFromLibrary(libraryLocation)

    # don't load files we already know about from the library
    filesToLoad = [f for f in fileList if f not in headersDict.keys()]

    s = slice(0, 100)
    for filenum, filename in enumerate(filesToLoad):
        if len(filesToLoad) > 1000 and filenum % 1000 == 0:
            if libraryLocation:
                logger.info(
                    f"Processed {filenum} of {len(filesToLoad)} files not loaded from library..."
                )
            else:
                logger.info(f"Processed {filenum} of {len(fileList)} files...")
        with fits.open(filename) as f:
            try:
                headersDict[filename] = f[0].header
                h = _hashFile(f, dataHdu, s)
                if h in dataDict.values():
                    collision = _findKeyForValue(dataDict, h, warnOnCollision=False)
                    logger.warning(
                        f"Duplicate file (or hash collision!) for files {filename} and "
                        f"{collision}!"
                    )
                    if filecmp.cmp(filename, collision):
                        logger.warning("Filecmp shows files are identical")
                    else:
                        logger.warning(
                            "Filecmp shows files differ - "
                            "likely just zeros for data (or a genuine hash collision!)"
                        )

                dataDict[filename] = h
            except Exception:
                logger.warning(f"Failed to load {filename} - file is likely corrupted.")

    # we have always added to this, so save it back over the original
    if libraryLocation and len(filesToLoad) > 0:
        _saveToLibrary(libraryLocation, headersDict, dataDict)

    # have to pare these down, as library loaded could be a superset
    headersDict = {k: headersDict[k] for k in fileList if k in headersDict.keys()}
    dataDict = {k: dataDict[k] for k in fileList if k in dataDict.keys()}

    return headersDict, dataDict


def sorted(inlist, replacementValue="<BLANK VALUE>"):
    """Redefinition of sorted() to deal with blank values and str/int mixes"""
    from builtins import sorted as _sorted

    output = [
        (
            str(x)
            if not isinstance(x, astropy.io.fits.card.Undefined)
            else replacementValue
        )
        for x in inlist
    ]
    output = _sorted(output)
    return output


def keyValuesSetFromFiles(
    fileList,
    keys,
    joinKeys,
    noWarn=False,
    printResults=True,
    libraryLocation=None,
    printPerFile=False,
):
    """For a list of FITS files, get the set of values for the given keys.

    Parameters
    ----------
    fileList : `list` of `str`
        The fully-specified paths of the files to scrape

    keys : `list` of `str`
        The header keys to scrape

    joinKeys : `list` of `str`
        List of keys to concatenate when scraping, e.g. for a header with
        FILTER1 = SDSS_u and FILTER2 == NB_640nm
        this would return SDSS_u+NB_640nm
        Useful when looking for the actual set, rather than taking the product
        of all the individual values, as some combinations may never happen.
    """
    print(f"Scraping headers from {len(fileList)} files...")
    if printPerFile and (len(fileList) * len(keys) > 200):
        print(
            f"You asked to print headers per-file, for {len(fileList)} files x {len(keys)} keys."
        )
        cont = input("Are you sure? Press y to continue, anything else to quit:")
        if cont.lower()[0] != "y":
            exit()

    headerDict, hashDict = buildHashAndHeaderDicts(
        fileList, libraryLocation=libraryLocation
    )

    if keys:  # necessary so that -j works on its own
        kValues = {k: set() for k in keys}
    else:
        keys = []
        kValues = None

    if joinKeys:
        joinedValues = set()

    for filename in headerDict.keys():
        header = headerDict[filename]
        for key in keys:
            if key in header:
                kValues[key].add(header[key])
                if printPerFile:
                    print(f"{filename}\t{key}\t{header[key]}")
                    if len(keys) > 1 and key == keys[-1]:
                        # newline between files if multikey
                        print()
            else:
                if not noWarn:
                    logger.warning(f"{key} not found in header of {filename}")

        if joinKeys:
            jVals = None
            # Note that CCS doesn't leave values blank, it misses the whole
            # card out for things like FILTER2 when not being used
            jVals = [header[k] if k in header else "<missing card>" for k in joinKeys]

            # However, we do ALSO get blank cards to, so:
            # substitute <BLANK_VALUE> when there is an undefined card
            # because str(v) will give the address for each blank value
            # too, meaning each blank card looks like a different value
            joinedValues.add(
                "+".join(
                    [
                        (
                            str(v)
                            if not isinstance(v, astropy.io.fits.card.Undefined)
                            else "<BLANK_VALUE>"
                        )
                        for v in jVals
                    ]
                )
            )

    if printResults:
        # Do this first because it's messy
        zeroFiles = _findKeyForValue(
            hashDict, ZERO_HASH, warnOnCollision=False, returnCollisions=True
        )
        if zeroFiles:
            print("\nFiles with zeros for data:")
        for filename in zeroFiles:
            print(f"{filename}")

        if kValues is not None:
            for key in kValues.keys():
                print(f"\nValues found for header key {key}:")
                print(f"{sorted(kValues[key])}")

        if joinKeys:
            print(f"\nValues found when joining {joinKeys}:")
            print(f"{sorted(joinedValues)}")

    if joinKeys:
        return kValues, joinedValues

    return kValues


def compareHeaders(filename1, filename2):
    """Compare the headers of two files in detail.

    First, the two files are confirmed to have the same pixel data to ensure
    the files should be being compared (by hashing the first 100x100 pixels
    in HDU 1).

    It then prints out:
        the keys that appear in A and not B
        the keys that appear in B but not A
        the keys that in common, and of those in common:
            which are the same,
            which differ,
            and where different, what the differing values are

    Parameters
    ----------
    filename1 : str
        Full path to the first of the files to compare

    filename2 : str
        Full path to the second of the files to compare
    """
    assert isinstance(filename1, str)
    assert isinstance(filename2, str)

    headerDict1, hashDict1 = buildHashAndHeaderDicts([filename1])
    headerDict2, hashDict2 = buildHashAndHeaderDicts([filename2])

    if hashDict1[filename1] != hashDict2[filename2]:
        print(
            "Pixel data was not the same - did you really mean to compare these files?"
        )
        print(f"{filename1}\n{filename2}")
        cont = input("Press y to continue, anything else to quit:")
        if cont.lower()[0] != "y":
            exit()

    # you might think you don't want to always call sorted() on the key sets
    # BUT otherwise they seem to be returned in random order each time you run
    # and that can be crazy-making

    h1 = headerDict1[filename1]
    h2 = headerDict2[filename2]
    h1Keys = list(h1.keys())
    h2Keys = list(h2.keys())

    commonKeys = set(h1Keys)
    commonKeys = commonKeys.intersection(h2Keys)

    keysInh1NotInh2 = sorted([_ for _ in h1Keys if _ not in h2Keys])
    keysInh2NotInh1 = sorted([_ for _ in h2Keys if _ not in h1Keys])

    print(f"Keys in {filename1} not in {filename2}:\n{keysInh1NotInh2}\n")
    print(f"Keys in {filename2} not in {filename1}:\n{keysInh2NotInh1}\n")
    print(f"Keys in common:\n{sorted(commonKeys)}\n")

    # put in lists so we can output neatly rather than interleaving
    identical = []
    differing = []
    for key in commonKeys:
        if h1[key] == h2[key]:
            identical.append(key)
        else:
            differing.append(key)

    assert len(identical) + len(differing) == len(commonKeys)

    if len(identical) == len(commonKeys):
        print("All keys in common have identical values :)")
    else:
        print("Of the common keys, the following had identical values:")
        print(f"{sorted(identical)}\n")
        print("Common keys with differing values were:")
        for key in sorted(differing):
            d = "<blank card>".ljust(25)
            v1 = (
                str(h1[key]).ljust(25)
                if not isinstance(h1[key], astropy.io.fits.card.Undefined)
                else d
            )
            v2 = (
                str(h2[key]).ljust(25)
                if not isinstance(h2[key], astropy.io.fits.card.Undefined)
                else d
            )
            print(f"{key.ljust(8)}: {v1} vs {v2}")

    # Finally, check the extension naming has the same ordering.
    # We have to touch the files again, which is pretty lame
    # but not doing so would require the header builder to know about
    # file pairings or return extra info, and that's not ideal either,
    # and also not worth the hassle to optimise as this is only
    # ever for a single file, not bulk file processing
    numbering1, numbering2 = [], []
    with fits.open(filename1) as f1, fits.open(filename2) as f2:
        for hduF1, hduF2 in zip(f1[1:], f2[1:]):  # skip the PDU
            if "EXTNAME" in hduF1.header and "EXTNAME" in hduF2.header:
                numbering1.append(hduF1.header["EXTNAME"])
                numbering2.append(hduF2.header["EXTNAME"])

    if numbering1 != numbering2:
        print("\nSection numbering differs between files!")
        for s1, s2 in zip(numbering1, numbering2):
            print(f"{s1.ljust(12)} vs {s2.ljust(12)}")
    if len(numbering1) != len(numbering2):
        print(
            "The length of those lists was also DIFFERENT! Presumably a non-image HDU was interleaved."
        )
