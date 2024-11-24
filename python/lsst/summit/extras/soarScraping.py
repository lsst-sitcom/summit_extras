import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime

import easyocr
import numpy as np
import requests
from packaging import version

# Check Pillow version to determine the correct resampling filter
from PIL import Image, ImageEnhance
from PIL import __version__ as PILLOW_VERSION

if version.parse(PILLOW_VERSION) >= version.parse("9.1.0"):
    # For Pillow >= 9.1.0
    resample_filter = Image.LANCZOS
else:
    # For older versions of Pillow
    resample_filter = Image.ANTIALIAS


USER_COORDINATES = {
    "dateImage": ((42, 164), (222, 186)),
    "seeingImage": ((180, 128), (233, 154)),
    "freeAtmSeeingImage": ((180, 103), (233, 128)),
    "groundLayerImage": ((180, 80), (233, 103)),
}

SOAR_IMAGE_URL = "http://www.ctio.noirlab.edu/~soarsitemon/soar_seeing_monitor.png"


@dataclass
class SeeingConditions:
    timestamp: datetime
    seeing: float
    freeAtmSeeing: float
    groundLayer: float

    def __repr__(self):
        return (
            f"SeeingConditions @ {self.timestamp.isoformat()}\n"
            f"  Seeing          = {self.seeing}\n"
            f"  Free Atm Seeing = {self.freeAtmSeeing}\n"
            f"  Ground layer    = {self.groundLayer}\n"
        )


class SoarScraper:

    def __init__(self):
        logging.getLogger("easyocr").setLevel(logging.ERROR)
        self.reader = easyocr.Reader(["en"])

    @staticmethod
    def adjust_coords(coords, image_height):
        (x0, y0), (x1, y1) = coords
        # Adjust y coordinates
        y0_new = image_height - y0
        y1_new = image_height - y1
        # Ensure that upper < lower for the crop function
        upper = min(y0_new, y1_new)
        lower = max(y0_new, y1_new)
        return (x0, upper, x1, lower)

    def getSeeingConditionsFromFile(self, filename):
        input_image = Image.open(filename)
        image_width, image_height = input_image.size  # Get image dimensions
        coordinates = {
            name: self.adjust_coords(coords, image_height) for name, coords in USER_COORDINATES.items()
        }

        # Crop and store sub-images in variables
        dateImage = input_image.crop(coordinates["dateImage"])
        seeingImage = input_image.crop(coordinates["seeingImage"])
        freeAtmSeeingImage = input_image.crop(coordinates["freeAtmSeeingImage"])
        groundLayerImage = input_image.crop(coordinates["groundLayerImage"])

        date = self._getDateTime(dateImage)
        seeing = self._getSeeingNumber(seeingImage)
        freeAtmSeeing = self._getSeeingNumber(freeAtmSeeingImage)
        groundLayer = self._getSeeingNumber(groundLayerImage)

        return SeeingConditions(date, seeing, freeAtmSeeing, groundLayer)

    def getCurrentSeeing(self):
        with tempfile.NamedTemporaryFile() as temp_file:
            with requests.get(SOAR_IMAGE_URL, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            # Call the parsing method within the same context
            seeing_conditions = self.getSeeingConditionsFromFile(temp_file.name)

        return seeing_conditions

    def _getDateTime(self, dateImage):
        dateImage_np = np.array(dateImage)
        results = self.reader.readtext(dateImage_np, detail=0)
        date_string = " ".join(results).strip()
        date_string = date_string.replace(".", ":")  # Replace dots with colons for parsing

        try:
            date_obj = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # If the format doesn't match, try alternative formats
            # For example, if there's an underscore instead of a space
            date_obj = datetime.strptime(date_string, "%Y-%m-%d_%H:%M:%S")

        return date_obj

    def _getSeeingNumber(self, image):
        scale_factor = 4  # You can adjust this factor as needed
        new_size = (image.width * scale_factor, image.height * scale_factor)
        image = image.resize(new_size, resample=resample_filter)

        image_gray = image.convert("L")

        enhancer = ImageEnhance.Contrast(image_gray)
        seeingImage_contrast = enhancer.enhance(2.0)  # Increase contrast by a factor of 2

        threshold = 128  # You can adjust this threshold as needed

        def threshold_image(image, threshold):
            return image.point(lambda x: 0 if x < threshold else 255, "L")

        seeingImage_bw = threshold_image(seeingImage_contrast, threshold)
        seeingImage_np = np.array(seeingImage_bw).astype(np.uint8)
        results = self.reader.readtext(seeingImage_np, detail=0, contrast_ths=0.1, adjust_contrast=0.5)
        seeing_text = " ".join(results).strip()

        match = re.search(r"[-+]?\d*\.\d+|\d+", seeing_text)
        if match:
            seeing_value = float(match.group())
            return seeing_value
        else:
            print("No numerical value found in the OCR result.")
            return float("nan")
