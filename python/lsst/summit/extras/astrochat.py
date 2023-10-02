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

import logging
import os

INSTALL_NEEDED = False
LOG = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    INSTALL_NEEDED = True
    LOG.warning("openai package not found. Please install openai: pip install openai")


def _checkInstallation():
    if INSTALL_NEEDED:
        raise RuntimeError("openai package not found. Please install openai: pip install openai")


def setApiKey(filename="~/.openaikey.txt"):
    """Set the OpenAI API key from a file.

    Set the OpenAI API key from a file. The file should contain a single line
    with the API key. The file name can be specified as an argument. If the
    API key is already set, it will be overwritten, with a warning issues.

    Parameters
    ----------
    filename : `str`
        Name of the file containing the API key.
    """
    _checkInstallation()

    currentKey = os.getenv('OPENAI_API_KEY')
    if currentKey:
        LOG.warning(f"OPENAI_API_KEY is already set. Overwriting with key from {filename}")

    filename = os.path.expanduser(filename)
    with open(filename, 'r') as file:
        apiKey = file.readline().strip()

    openai.api_key = apiKey
    os.environ["OPENAI_API_KEY"] = apiKey
