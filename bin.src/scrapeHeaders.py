#!/usr/bin/env python

import argparse
import glob
import os
import sys
from os.path import abspath

from lsst.summit.extras.headerFunctions import keyValuesSetFromFiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        type=str,
        help=(
            "List of files to scrape. Enclose any glob " "patterns in quotes so they are passed unexpanded"
        ),
    )
    parser.add_argument("-k", metavar="keys", dest="keys", nargs="+", type=str, help="Keys to return")
    parser.add_argument(
        "-j", metavar="joinKeys", dest="joinKeys", nargs="+", type=str, help="Keys to return joined together."
    )
    parser.add_argument(
        "-l",
        metavar="libraryLocation",
        dest="libraryLocation",
        type=str,
        help="Location of library for precomputed results.",
    )
    parser.add_argument(
        "-p", action="store_true", default=False, dest="printPerFile", help="Print all keys for each file?"
    )
    parser.add_argument(
        "--noWarn",
        action="store_true",
        help="Suppress warnings for keys not in header?",
        default=False,
        dest="noWarn",
    )
    parser.add_argument(
        "--walk",
        action="store_true",
        help="Ignore path glob and walk whole tree for" " all fits and fits.gz files",
        default=False,
        dest="walk",
    )
    parser.add_argument(
        "--oneFilePerDir",
        action="store_true",
        help="If walking, only take one file from" " each directory",
        default=False,
        dest="oneFilePerDir",
    )

    args = parser.parse_args()
    keys = args.keys
    joinKeys = args.joinKeys
    noWarn = args.noWarn
    walk = args.walk
    oneFilePerDir = args.oneFilePerDir
    libraryLocation = args.libraryLocation
    printPerFile = args.printPerFile
    # important to use absolute paths always, as these are used to key the
    # dicts and these are also pickled, so need to be the same no matter where
    # this is run from
    files = []
    if walk:
        dirToWalk = os.path.dirname(abspath(args.files))
        print(f"Walking files from {dirToWalk}...")

        for dirpath, dirnames, filenames in os.walk(args.files):
            print(f"Collected {len(files)} files so far...")
            for filename in [f for f in filenames if f.endswith(".fits") or f.endswith(".fits.gz")]:
                files.append(abspath(os.path.join(dirpath, filename)))
                if oneFilePerDir:
                    break
    else:
        files = [abspath(f) for f in glob.glob(args.files)]
    print(f"Collected {len(files)} in total")

    if not keys and not joinKeys:
        print(
            (
                "No keys requested for scraping! Specify with e.g. -k KEY1 KEY2, "
                "or with e.g. -j FILTER FILTER2 for keys to join"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    if not files:
        print("Found no files matching: " + args.files, file=sys.stderr)
        sys.exit(1)

    keyValuesSetFromFiles(
        files, keys, joinKeys, noWarn, libraryLocation=libraryLocation, printPerFile=printPerFile
    )


if __name__ == "__main__":
    main()
