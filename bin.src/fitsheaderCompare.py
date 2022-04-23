#!/usr/bin/env python

import argparse
from lsst.summit.extras.headerFunctions import compareHeaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs='+',
                        help=("List of files to scrape. Enclose any glob "
                              "patterns in quotes so they are passed unexpanded"))

    args = parser.parse_args()
    files = args.files

    if len(files) != 2:
        print("Must provide 2 and only 2 files for comparison")
        print(f"Got {files} for files to compare")

    compareHeaders(files[0], files[1])


if __name__ == '__main__':
    main()
