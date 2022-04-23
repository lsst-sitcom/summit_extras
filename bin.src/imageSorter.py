#!/usr/bin/env python

import argparse
import glob
import sys
import os
from os.path import abspath
from lsst.summit.extras import ImageSorter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, help=("List of files to scrape. Enclose any glob "
                                                 "patterns in quotes so they are passed unexpanded"))
    parser.add_argument("-o", metavar='outputFile', dest='outputFile', nargs='+', type=str,
                        help="Full path and filename to write results to")

    args = parser.parse_args()
    files = args.files
    outputFile = args.outputFile[0]

    files = sorted([abspath(f) for f in glob.glob(args.files)])
    if not files:
        print('Found no files matching: ' + args.files, file=sys.stderr)
        sys.exit(1)

    dirname = os.path.dirname(outputFile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print(f"Annotating {len(files)} files, writing data to {outputFile}...")

    sorter = ImageSorter(files, outputFile)
    sorter.sortImages()


if __name__ == '__main__':
    main()
